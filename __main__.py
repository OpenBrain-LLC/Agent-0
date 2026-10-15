#!/usr/bin/env python3
"""Agent-0 — autonomous text-tool AI agent with Ollama + Groq backends.

Deployment
----------
* Config file:     db/local/.config.json   (override with $AGENT0_CONFIG)
* Memory file:     db/local/memory.md      (override with $AGENT0_MEMORY)
* Ollama model:    deepseek-r1:1.5b        (override with $AGENT0_MODEL)
* Groq API key:    set via /api <key>, or export $AGENT0_GROQ_API_KEY
* Requires:        pip install ollama         (ollama backend)
                   pip install groq           (groq backend)
* Needs an interactive TTY.

Autonomy levels (see /auto):
  manual  — confirm every tool call
  smart   — auto-approve read-only tools
  auto    — also auto-approve file edits / memory / todos  (DEFAULT)
  yolo    — auto-approve everything, including shell and network

Backends use native function calling:
  * Ollama — ollama.chat(..., tools=[...]) and message.tool_calls.
  * Groq   — client.chat.completions.create(..., tools=[...]) and delta.tool_calls.
"""

from __future__ import annotations

import datetime
import fnmatch
import inspect
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

try:
    import ollama  # type: ignore
    _OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None  # type: ignore
    _OLLAMA_AVAILABLE = False

try:
    from groq import Groq as _GroqClient  # type: ignore
    _GROQ_AVAILABLE = True
except ImportError:
    _GroqClient = None  # type: ignore
    _GROQ_AVAILABLE = False


__version__ = "3.0.0"

# ---------------- ANSI ----------------

def _colors_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


if _colors_enabled():
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    GRAY   = "\033[90m"
    CYAN   = "\033[36m"
    YELLOW = "\033[33m"
    GREEN  = "\033[32m"
    RED    = "\033[31m"
    MAGENTA = "\033[35m"
    RESET  = "\033[0m"
else:
    BOLD = DIM = GRAY = CYAN = YELLOW = GREEN = RED = MAGENTA = RESET = ""


# ---------------- Config ----------------

SCRIPT_DIR = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else default


CONFIG_PATH = _env_path("AGENT0_CONFIG", SCRIPT_DIR / "db" / "local" / ".config.json")
MEMORY_PATH = _env_path("AGENT0_MEMORY", SCRIPT_DIR / "db" / "local" / "memory.md")
DEFAULT_OLLAMA_MODEL = os.environ.get("AGENT0_MODEL") or "deepseek-r1:1.5b"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

DEFAULT_CONFIG = {
    "model": DEFAULT_OLLAMA_MODEL,
    "provider": "ollama",
    "autonomy": "auto",           # manual | smart | auto | yolo
    "groq": {"api_key": "", "model": DEFAULT_GROQ_MODEL},
    "thinking": {"show": "True", "think": "True", "fast": "False"},
}

MAX_STEPS = 24
TOOL_OUTPUT_LIMIT = 20000
TOOL_PREVIEW_LIMIT = 500
MEMORY_INJECT_LIMIT = 6000
MEMORY_MAX_BYTES = 200_000
SHELL_TIMEOUT = 60
FETCH_TIMEOUT = 20
FETCH_MAX_BYTES = 500_000
TEMPERATURE = 0.25

CTRL_C_HINT = f"{GRAY}Ctrl-C is disabled — type /bye to exit{RESET}"
WELCOME = f"Welcome to Agent-0 AI [v{__version__}]\nBy: OpenBrain-LLC"


def warn(msg: str) -> None:
    print(f"{GRAY}[warn] {msg}{RESET}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"{GRAY}[info] {msg}{RESET}", file=sys.stderr)


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _as_bool(v: Any, default: bool = False) -> bool:
    return _to_bool(v, default)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def load_config() -> dict:
    cfg = {
        "model": DEFAULT_OLLAMA_MODEL,
        "provider": "ollama",
        "autonomy": "auto",
        "groq": {"api_key": "", "model": DEFAULT_GROQ_MODEL},
        "thinking": dict(DEFAULT_CONFIG["thinking"]),
    }

    if not CONFIG_PATH.exists():
        warn(f"{CONFIG_PATH} not found; creating with defaults")
        try:
            save_config(cfg)
        except OSError as e:
            warn(f"could not create config: {e}")
        return cfg

    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("config is not a JSON object")

        if isinstance(data.get("model"), str):
            cfg["model"] = data["model"]
        if data.get("provider") in ("ollama", "groq"):
            cfg["provider"] = data["provider"]
        if data.get("autonomy") in ("manual", "smart", "auto", "yolo"):
            cfg["autonomy"] = data["autonomy"]

        g = data.get("groq")
        if isinstance(g, dict):
            if isinstance(g.get("api_key"), str):
                cfg["groq"]["api_key"] = g["api_key"]
            if isinstance(g.get("model"), str):
                cfg["groq"]["model"] = g["model"]

        th = data.get("thinking")
        if isinstance(th, dict):
            for k in ("show", "think", "fast"):
                if k in th:
                    cfg["thinking"][k] = th[k]
    except (json.JSONDecodeError, OSError, ValueError) as e:
        warn(f"could not read config ({e}); using defaults")

    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=4) + "\n", encoding="utf-8")


def _mask_api_key(key: str) -> str:
    if not key:
        return "(not set)"
    if len(key) < 12:
        return "*" * len(key)
    prefix_len = min(7, len(key) - 5)
    return f"{key[:prefix_len]}****...**{key[-5:]}"


class State:
    def __init__(self) -> None:
        self.config = load_config()

        self.model = self.config["model"]
        self.provider = self.config.get("provider", "ollama")
        self.autonomy = self.config.get("autonomy", "auto")

        g = self.config.get("groq") or {}
        self.groq_model = g.get("model", DEFAULT_GROQ_MODEL)

        env_key = os.environ.get("AGENT0_GROQ_API_KEY", "")
        self._key_from_env = bool(env_key)
        self._persisted_groq_key = g.get("api_key", "")
        self.groq_api_key = env_key or self._persisted_groq_key

        th = self.config["thinking"]
        self.show_thinking = _to_bool(th.get("show"), False)
        self.think_enabled = _to_bool(th.get("think"), True)
        self.fast_think    = _to_bool(th.get("fast"), False)

        self._sync_config()

    def _sync_config(self) -> None:
        self.config["model"] = self.model
        self.config["provider"] = self.provider
        self.config["autonomy"] = self.autonomy

        groq_cfg: dict = {"model": self.groq_model}
        if self._key_from_env:
            if self._persisted_groq_key:
                groq_cfg["api_key"] = self._persisted_groq_key
        else:
            groq_cfg["api_key"] = self.groq_api_key
        self.config["groq"] = groq_cfg

        self.config["thinking"] = {
            "show":  "True" if self.show_thinking else "False",
            "think": "True" if self.think_enabled else "False",
            "fast":  "True" if self.fast_think else "False",
        }

    def _persist(self) -> None:
        try:
            save_config(self.config)
        except OSError as e:
            warn(f"could not persist config to {CONFIG_PATH}: {e}")

    def set_model(self, name: str) -> None:
        self.model = name
        self._sync_config()
        self._persist()

    def set_groq_model(self, name: str) -> None:
        self.groq_model = name
        self._sync_config()
        self._persist()

    def set_groq_key(self, key: str) -> None:
        self.groq_api_key = key
        self._key_from_env = False
        self._persisted_groq_key = key
        self._sync_config()
        self._persist()

    def set_provider(self, provider: str) -> None:
        if provider not in ("ollama", "groq"):
            raise ValueError(f"unknown provider: {provider}")
        self.provider = provider
        self._sync_config()
        self._persist()

    def set_autonomy(self, level: str) -> None:
        if level not in AUTONOMY_LEVELS:
            raise ValueError(f"unknown autonomy level: {level}")
        self.autonomy = level
        self._sync_config()
        self._persist()

    def set_thinking(self, *, show=None, think=None, fast=None) -> None:
        if show is not None:
            self.show_thinking = bool(show)
        if think is not None:
            self.think_enabled = bool(think)
        if fast is not None:
            self.fast_think = bool(fast)
        self._sync_config()
        self._persist()

    def current_model(self) -> str:
        return self.groq_model if self.provider == "groq" else self.model

    def think_param(self):
        if not self.think_enabled:
            return False
        if self.fast_think:
            return "low"
        return True


STATE = State()


# ---------------- Autonomy policy ----------------

SAFE_TOOLS = {"read_file", "list_dir", "glob", "grep", "todo_read", "todo_write", "memory_read"}
MEDIUM_TOOLS = {"write_file", "edit_file", "memory_write", "memory_append"}
DANGEROUS_TOOLS = {"shell", "fetch_url"}

TOOL_POLICY: dict[str, str] = {}
for _n in SAFE_TOOLS:      TOOL_POLICY[_n] = "safe"
for _n in MEDIUM_TOOLS:    TOOL_POLICY[_n] = "medium"
for _n in DANGEROUS_TOOLS: TOOL_POLICY[_n] = "dangerous"

AUTONOMY_LEVELS: dict[str, dict[str, bool]] = {
    "manual": {"safe": False, "medium": False, "dangerous": False},
    "smart":  {"safe": True,  "medium": False, "dangerous": False},
    "auto":   {"safe": True,  "medium": True,  "dangerous": False},
    "yolo":   {"safe": True,  "medium": True,  "dangerous": True},
}

AUTONOMY_ORDER = ["manual", "smart", "auto", "yolo"]


def _tool_policy(name: str) -> str:
    return TOOL_POLICY.get(name, "dangerous")


def _autonomy_allows(name: str) -> bool:
    """True if the current autonomy level auto-approves this tool."""
    level = AUTONOMY_LEVELS.get(STATE.autonomy, AUTONOMY_LEVELS["auto"])
    return bool(level.get(_tool_policy(name), False))


# ---------------- Todos ----------------

_TODOS: list[dict] = []


def _reset_todos() -> None:
    _TODOS.clear()


# ---------------- Memory ----------------

_MEMORY_DEFAULT = """# Agent-0 Memory

## User
<!-- durable facts about the user: name, preferences, timezone, projects -->

## Projects
<!-- ongoing work, repo layout notes, decisions -->

## Notes
<!-- gotchas, commands that worked, follow-ups -->
"""


def _memory_init() -> None:
    if MEMORY_PATH.exists():
        return
    try:
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_PATH.write_text(_MEMORY_DEFAULT, encoding="utf-8")
    except OSError as e:
        warn(f"could not create memory file {MEMORY_PATH}: {e}")


def _load_memory_text() -> str:
    try:
        if not MEMORY_PATH.exists():
            return ""
        text = MEMORY_PATH.read_text(encoding="utf-8").strip()
    except OSError as e:
        return f"(could not read memory: {e})"
    if not text:
        return ""
    if len(text) > MEMORY_INJECT_LIMIT:
        text = text[:MEMORY_INJECT_LIMIT] + f"\n\n... [memory truncated at {MEMORY_INJECT_LIMIT} chars]"
    return text


def tool_memory_read() -> str:
    try:
        if not MEMORY_PATH.exists():
            return "(memory file does not exist yet)"
        text = MEMORY_PATH.read_text(encoding="utf-8")
        if not text.strip():
            return "(memory file is empty)"
        return _truncate(text)
    except Exception as e:
        return f"Error: {e}"


def tool_memory_write(content: str) -> str:
    try:
        if not isinstance(content, str):
            return f"Error: content must be a string, got {type(content).__name__}"
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        encoded = content.encode("utf-8")
        if len(encoded) > MEMORY_MAX_BYTES:
            return f"Error: memory content is too large ({len(encoded)} bytes > {MEMORY_MAX_BYTES})"
        MEMORY_PATH.write_text(content, encoding="utf-8")
        return f"Wrote {len(encoded)} bytes to {MEMORY_PATH}"
    except Exception as e:
        return f"Error: {e}"


def tool_memory_append(section: str, entry: str) -> str:
    try:
        section = (section or "").strip().lstrip("#").strip()
        entry = (entry or "").strip()
        if not section:
            return "Error: 'section' must be a non-empty string"
        if not entry:
            return "Error: 'entry' must be a non-empty string"

        if not entry.startswith(("-", "*", "#")):
            entry = "- " + entry
        entry = entry.replace("\n", " ").strip()

        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        if MEMORY_PATH.exists():
            text = MEMORY_PATH.read_text(encoding="utf-8")
        else:
            text = "# Agent-0 Memory\n"

        lines = text.rstrip("\n").split("\n")
        header = f"## {section}"

        section_idx: int | None = None
        next_idx: int | None = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if section_idx is None and stripped == header:
                section_idx = i
                continue
            if section_idx is not None and stripped.startswith("## "):
                next_idx = i
                break

        if section_idx is None:
            if lines and lines[-1].strip() != "":
                lines.append("")
            lines.append(header)
            lines.append(entry)
        else:
            insert_at = next_idx if next_idx is not None else len(lines)
            while insert_at > section_idx + 1 and lines[insert_at - 1].strip() == "":
                insert_at -= 1
            lines.insert(insert_at, entry)

        new_text = "\n".join(lines).rstrip("\n") + "\n"
        encoded = new_text.encode("utf-8")
        if len(encoded) > MEMORY_MAX_BYTES:
            return f"Error: memory would grow past {MEMORY_MAX_BYTES} bytes; use memory_write to consolidate"
        MEMORY_PATH.write_text(new_text, encoding="utf-8")
        return f"Appended to memory under '{section}'."
    except Exception as e:
        return f"Error: {e}"


# ---------------- Tool schema ----------------

def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _fn(
        "shell",
        "Run a shell command. Returns combined stdout+stderr and exit code. "
        "Prefer read-only commands (ls, git status, git diff) unless the user "
        f"asked for a change. Times out after {SHELL_TIMEOUT} seconds.",
        {"command": {"type": "string", "description": "The shell command to run."}},
        ["command"],
    ),
    _fn(
        "read_file",
        "Return the text contents of a file. Read a file before editing it.",
        {"path": {"type": "string", "description": "Path to the file."}},
        ["path"],
    ),
    _fn(
        "write_file",
        "Write content to a path, overwriting it. Creates parent directories. "
        "Use for NEW files or full rewrites; for small edits prefer edit_file.",
        {
            "path": {"type": "string", "description": "Path to write."},
            "content": {"type": "string", "description": "Full file contents."},
        },
        ["path", "content"],
    ),
    _fn(
        "edit_file",
        "Replace old_string with new_string in a file. old_string must appear "
        "EXACTLY ONCE unless replace_all is true. Include enough surrounding "
        "context to make old_string unique.",
        {
            "path": {"type": "string", "description": "Path to edit."},
            "old_string": {"type": "string", "description": "Exact text to find."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean",
                            "description": "Replace every occurrence (default false)."},
        },
        ["path", "old_string", "new_string"],
    ),
    _fn(
        "list_dir",
        "List a directory's contents as a tree. Ignores .git, node_modules, "
        "__pycache__, venvs, build dirs.",
        {
            "path": {"type": "string", "description": "Directory (default '.')."},
            "recursive": {"type": "boolean",
                          "description": "List recursively (default false)."},
        },
        [],
    ),
    _fn(
        "glob",
        "Find files by glob pattern, e.g. '**/*.py' or 'src/**/*.ts'.",
        {
            "pattern": {"type": "string", "description": "Glob pattern."},
            "path": {"type": "string", "description": "Base directory (default '.')."},
        },
        ["pattern"],
    ),
    _fn(
        "grep",
        "Regex search inside files. Optionally filter by filename glob.",
        {
            "pattern": {"type": "string", "description": "Regex to search for."},
            "path": {"type": "string", "description": "Root dir or file (default '.')."},
            "include": {"type": "string", "description": "Filename glob filter, e.g. '*.py'."},
            "case_insensitive": {"type": "boolean", "description": "Ignore case."},
        },
        ["pattern"],
    ),
    _fn(
        "fetch_url",
        "HTTP GET a URL and return the text body (up to ~500 KB). Only http/https.",
        {"url": {"type": "string", "description": "URL to fetch."}},
        ["url"],
    ),
    _fn(
        "todo_write",
        "Replace the session todo list. Each item: "
        '{"content": str, "status": "pending"|"in_progress"|"completed", '
        '"activeForm": optional str}. At most one item may be in_progress.',
        {
            "todos": {
                "type": "array",
                "description": "The full todo list.",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string"},
                        "activeForm": {"type": "string"},
                    },
                    "required": ["content", "status"],
                },
            }
        },
        ["todos"],
    ),
    _fn("todo_read", "Return the current session todo list.", {}, []),
    _fn(
        "memory_read",
        "Return the contents of the persistent memory file (db/local/memory.md). "
        "Read this when the user references prior context or ongoing projects.",
        {},
        [],
    ),
    _fn(
        "memory_append",
        "Append a single bullet to a section of the persistent memory file. "
        "Creates the section if it doesn't exist. Use for one durable fact.",
        {
            "section": {"type": "string", "description": "Section name, e.g. 'User' or 'Projects'."},
            "entry":   {"type": "string", "description": "The bullet text to append."},
        },
        ["section", "entry"],
    ),
    _fn(
        "memory_write",
        "Replace the ENTIRE memory file. Use to reorganize or consolidate. "
        "Content should be markdown with '## Section' headings.",
        {"content": {"type": "string", "description": "Full new memory file contents."}},
        ["content"],
    ),
]

TOOL_DESCRIPTIONS = {
    "shell":         "run a shell command (returns stdout+stderr)",
    "read_file":     "return the file's text contents",
    "write_file":    "write content to a path (overwrites)",
    "edit_file":     "replace old_string with new_string in a file",
    "list_dir":      "list a directory's contents (tree)",
    "glob":          "find files by glob pattern",
    "grep":          "regex search inside files",
    "fetch_url":     "HTTP GET a URL, return the text body",
    "todo_write":    "replace the session todo list",
    "todo_read":     "return the session todo list",
    "memory_read":   "return the persistent memory file",
    "memory_append": "append one bullet to a memory section",
    "memory_write":  "replace the entire memory file",
}


# ---------------- Prompts ----------------

SYSTEM_PROMPT = """You are Agent-0, a highly autonomous command-line coding and system agent running inside an interactive terminal on the user's machine. You work through multi-step tasks on your own, use tools aggressively, and remember durable facts across sessions.

# Environment
{environment}

# Autonomy
Your current autonomy level is `{autonomy}`.
- manual: every tool call is confirmed by the user.
- smart:  read-only tools run automatically; edits and commands are confirmed.
- auto:   read-only AND file edits run automatically; shell/network are confirmed.
- yolo:   everything runs automatically (user opted into this).
Act as if your actions will be approved. Do not ask the user for permission in
prose — the harness handles that. Do not stop mid-task to check in; keep going
until the task is done or you genuinely need user input.

# Memory
You have a persistent memory file at {memory_path}. It survives across sessions
and is injected below. Use it the way a careful engineer uses a notebook.
- Read it with `memory_read` when the user references prior work, or when you
  need context you don't have.
- Save DURABLE facts: user preferences, project names and layouts, decisions
  with reasons, gotchas, useful commands.
- Prefer `memory_append` for a single fact; `memory_write` only when
  reorganizing the whole file.
- Never store: secrets, API keys, credentials, transient tool output, or
  anything that will be stale in a week.
- Save proactively — if you learn something the user will want you to
  remember, write it down WITHOUT being asked.

### Current memory
{memory}

# How you reply
You have native tools available. Each turn, pick exactly one of:
- CALL one or more tools, or
- REPLY in prose (when the answer is ready, or no tool is needed).
Never do both in one turn. Never describe a tool call in prose — just call it.

# Tone
- Concise. No preamble ("Sure!", "Of course!"). No postamble ("Let me know if…").
- No narration ("Let me…", "I'll now…", "First I'll…").
- Answer in prose unless the user asked for code.
- If the user asks you to do something, DO it.
- Never speculate about code you haven't read. If a file was truncated, say so —
  do not fill in the rest from imagination.

# Your tools — exactly these, nothing else
shell, read_file, write_file, edit_file, list_dir, glob, grep, fetch_url,
todo_write, todo_read, memory_read, memory_append, memory_write.

When asked what you can do, describe ONLY those tools. Do NOT invent tools. If
you read source code that mentions other function names, they are still not
your tools unless they appear in the list above.

# Tool selection
- Find files by name → glob.
- Find code/text by content → grep.
- Understand a directory's shape → list_dir.
- Read a specific file → read_file (before editing it).
- Small targeted edits → edit_file (make old_string unique).
- Full rewrite or new file → write_file.
- Run something → shell.
- Fetch a URL → fetch_url.
- Persist a fact → memory_append.

# Batching
- Multiple INDEPENDENT calls in ONE turn (e.g. read two files, or git status + README).
- Calls that DEPEND on another's output go in a LATER turn.
- Never call the same tool with the same arguments twice in a row.

# Task management
For tasks with 3 or more distinct steps, use todo_write to plan before you start.
- Keep exactly ONE task in_progress at a time.
- Mark tasks completed the moment they are done — don't batch completions.
- Skip this for trivial work.

# Doing tasks
1. UNDERSTAND. Search before editing. Read files before editing them. Do not
   guess at APIs, file paths, or signatures.
2. PLAN. Write todos for multi-step work.
3. IMPLEMENT. Prefer edit_file with enough context that old_string is unique.
4. VERIFY. If the project has tests or a build, run them. Read your edits back.
5. REMEMBER. Save durable facts to memory. Clean up the todo list.

# Safety — never do these without an explicit user instruction
git commit, git push, git reset --hard, git clean -fd, rm -rf, chmod -R,
force-pushing, package installs that modify the system, or anything that
sends data off the machine. Autonomy does not mean recklessness.

# Examples
- "list my cwd"              → list_dir(path=".") → describe in prose.
- "what's in notes.txt?"     → read_file(path="notes.txt") → describe.
- "change port 8000→8080"    → grep → read_file → edit_file → confirm.
- "hi"                       → prose reply, no tools.
- "remember I use UV"        → memory_append(section="User", entry="Uses UV as the Python package manager.")
"""

FAST_THINK_NOTE = (
    "\n\n# Fast mode\nKeep your reasoning extremely brief — one or two short sentences maximum."
)

SUMMARY_SYSTEM_PROMPT = """You are a safety reviewer for a command-line AI agent.
You will receive ONE tool call as JSON. In ONE short sentence, say exactly what it will do on the user's machine.

- Be concrete: name the command, file paths, or side effects.
- Flag plainly if it is destructive, irreversible, or touches the network:
  - shell: delete, overwrite, force-push, chmod -R, kill, install packages.
  - edit_file / write_file / memory_write: mention the path.
  - fetch_url: mention the hostname.
- Output ONLY the sentence. No quotes, no "The tool call...", no preamble."""


def _environment_block() -> str:
    try:
        cwd = Path.cwd()
    except Exception:
        cwd = "(unknown)"
    try:
        plat = f"{platform.system()} {platform.release()}"
    except Exception:
        plat = "(unknown)"
    return (
        f"- Working directory: {cwd}\n"
        f"- Platform: {plat}\n"
        f"- Python: {platform.python_version()}\n"
        f"- Date: {datetime.date.today().isoformat()}"
    )


def build_system_prompt() -> str:
    mem = _load_memory_text() or "(empty)"
    p = SYSTEM_PROMPT.format(
        environment=_environment_block(),
        autonomy=STATE.autonomy,
        memory_path=MEMORY_PATH,
        memory=mem,
    )
    if STATE.fast_think:
        p += FAST_THINK_NOTE
    return p


# ---------------- Tool implementations ----------------

IGNORE_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__", "node_modules", ".venv", "venv", "env",
    ".idea", ".vscode", ".vs",
    "dist", "build", ".next", ".nuxt", ".cache", ".parcel-cache",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".eggs",
    "target", "vendor", "coverage", "htmlcov",
}


def _truncate(s: str, limit: int = TOOL_OUTPUT_LIMIT) -> str:
    if len(s) > limit:
        extra = len(s) - limit
        return s[:limit] + f"\n... [truncated, {extra} more chars]"
    return s


def _should_ignore(p: Path) -> bool:
    return any(part in IGNORE_DIRS for part in p.parts)


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
    except (ProcessLookupError, PermissionError, OSError):
        pass


def tool_shell(command: str) -> str:
    if not isinstance(command, str):
        return f"Error: 'command' must be a string, got {type(command).__name__}"
    posix = os.name == "posix"
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=posix,
        )
    except Exception as e:
        return f"Error: {e}"

    try:
        stdout, stderr = proc.communicate(timeout=SHELL_TIMEOUT)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        return f"Error: command timed out after {SHELL_TIMEOUT}s"
    except BaseException:
        _kill_tree(proc)
        raise

    out = ((stdout or "") + (stderr or "")).strip()
    rc = proc.returncode
    if not out:
        return f"(no output, exit code {rc})"
    out = _truncate(out)
    if rc != 0:
        out += f"\n[exit code {rc}]"
    return out


def tool_read_file(path: str) -> str:
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: file not found: {p}"
        if p.is_dir():
            return f"Error: {p} is a directory (use list_dir)"
        return _truncate(p.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        return f"Error: {e}"


def tool_write_file(path: str, content: str) -> str:
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        text = content if isinstance(content, str) else json.dumps(content, indent=2)
        p.write_text(text, encoding="utf-8")
        return f"Wrote {len(text.encode('utf-8'))} bytes to {p}"
    except Exception as e:
        return f"Error: {e}"


def tool_edit_file(path: str, old_string: str, new_string: str = "", replace_all: Any = False) -> str:
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: file not found: {p}"
        if p.is_dir():
            return f"Error: {p} is a directory"

        original = p.read_text(encoding="utf-8", errors="replace")

        if not isinstance(old_string, str) or old_string == "":
            return "Error: 'old_string' must be a non-empty string"

        replace_all = _as_bool(replace_all, False)

        count = original.count(old_string)
        if count == 0:
            return ("Error: 'old_string' not found in file. "
                    "Read the file first and copy the exact text (including whitespace).")
        if count > 1 and not replace_all:
            return (f"Error: 'old_string' occurs {count} times. "
                    f"Add surrounding context to make it unique, or pass replace_all=true.")

        updated = original.replace(old_string, new_string, -1 if replace_all else 1)
        p.write_text(updated, encoding="utf-8")

        n = count if replace_all else 1
        return f"Edited {p}: replaced {n} occurrence(s)."
    except Exception as e:
        return f"Error: {e}"


def _format_tree(root: Path, max_entries: int = 400) -> str:
    lines: list[str] = []
    count = 0

    def walk(d: Path, prefix: str) -> None:
        nonlocal count
        try:
            entries = sorted(
                (e for e in d.iterdir() if e.name not in IGNORE_DIRS),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except (PermissionError, OSError):
            return
        for i, e in enumerate(entries):
            if count >= max_entries:
                return
            count += 1
            last = i == len(entries) - 1
            branch = "└── " if last else "├── "
            suffix = "/" if e.is_dir() else ""
            lines.append(f"{prefix}{branch}{e.name}{suffix}")
            if e.is_dir():
                walk(e, prefix + ("    " if last else "│   "))

    walk(root, "")
    if count >= max_entries:
        lines.append(f"... [listing capped at {max_entries} entries]")
    return "\n".join(lines) if lines else "(empty)"


def tool_list_dir(path: str = ".", recursive: Any = False) -> str:
    try:
        root = Path(path).expanduser().resolve()
        if not root.exists():
            return f"Error: not found: {root}"
        if not root.is_dir():
            return f"Error: {root} is not a directory"

        header = f"{root}\n"

        if _as_bool(recursive, False):
            out: list[str] = []
            count = 0
            for p in sorted(root.rglob("*"), key=lambda p: str(p).lower()):
                if count >= 400:
                    out.append("... [listing capped at 400 entries]")
                    break
                rel = p.relative_to(root)
                if _should_ignore(rel):
                    continue
                count += 1
                marker = "/" if p.is_dir() else ""
                out.append(f"{rel}{marker}")
            body = "\n".join(out) if out else "(empty)"
            return _truncate(header + body)

        return _truncate(header + _format_tree(root))
    except Exception as e:
        return f"Error: {e}"


def tool_glob(pattern: str, path: str = ".") -> str:
    try:
        root = Path(path).expanduser().resolve()
        if not root.exists():
            return f"Error: not found: {root}"

        matches: list[Path] = []
        for p in root.glob(pattern):
            try:
                rel = p.relative_to(root)
            except ValueError:
                rel = p
            if _should_ignore(rel if isinstance(rel, Path) else Path(str(rel))):
                continue
            matches.append(p)
            if len(matches) >= 200:
                break

        if not matches:
            return f"No files matched '{pattern}' under {root}"

        matches.sort(key=lambda p: str(p).lower())
        lines = []
        for p in matches:
            try:
                lines.append(str(p.relative_to(root)))
            except ValueError:
                lines.append(str(p))
        if len(matches) >= 200:
            lines.append("... [capped at 200 results]")
        return _truncate("\n".join(lines))
    except Exception as e:
        return f"Error: {e}"


def tool_grep(pattern: str, path: str = ".", include: str = "", case_insensitive: Any = False) -> str:
    try:
        root = Path(path).expanduser()
        try:
            rx = re.compile(pattern, re.IGNORECASE if _as_bool(case_insensitive, False) else 0)
        except re.error as e:
            return f"Error: invalid regex: {e}"

        if root.is_file():
            files: Iterator[Path] = iter([root])
            base = root.parent
        elif root.is_dir():
            files = (p for p in root.rglob("*") if p.is_file())
            base = root
        else:
            return f"Error: not found: {root}"

        hits: list[str] = []
        max_hits = 200
        files_scanned = 0
        files_capped = 2000

        for f in files:
            try:
                rel = f.relative_to(base)
            except ValueError:
                rel = f
            if _should_ignore(rel if isinstance(rel, Path) else Path(str(rel))):
                continue
            if include and not fnmatch.fnmatch(f.name, include) and not fnmatch.fnmatch(str(rel), include):
                continue

            files_scanned += 1
            if files_scanned > files_capped:
                break

            try:
                if f.stat().st_size > 2_000_000:
                    continue
                data = f.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError):
                continue

            if "\x00" in data[:1024]:
                continue

            for i, line in enumerate(data.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{rel}:{i}: {line.strip()}")
                    if len(hits) >= max_hits:
                        break
            if len(hits) >= max_hits:
                break

        if not hits:
            return f"No matches for '{pattern}'"
        if len(hits) >= max_hits:
            hits.append(f"... [capped at {max_hits} matches]")
        return _truncate("\n".join(hits))
    except Exception as e:
        return f"Error: {e}"


def tool_fetch_url(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"Error: only http/https URLs are supported, got '{parsed.scheme}'"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": f"Agent-0/{__version__}"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
            raw = resp.read(FETCH_MAX_BYTES + 1)
            truncated = len(raw) > FETCH_MAX_BYTES
            raw = raw[:FETCH_MAX_BYTES]

        charset = resp.headers.get_content_charset() or "utf-8"
        try:
            text = raw.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            text = raw.decode("utf-8", errors="replace")

        head = f"HTTP {resp.status} {url}\nContent-Type: {resp.headers.get('Content-Type', '?')}\n\n"
        body = _truncate(text, TOOL_OUTPUT_LIMIT - len(head) - 50)
        if truncated:
            body += f"\n... [response truncated at {FETCH_MAX_BYTES} bytes]"
        return head + body
    except urllib.error.HTTPError as e:
        return f"Error: HTTP {e.code} {e.reason} for {url}"
    except urllib.error.URLError as e:
        return f"Error: could not reach {url}: {e.reason}"
    except Exception as e:
        return f"Error: {e}"


_VALID_TODO_STATUS = {"pending", "in_progress", "completed"}


def _format_todos() -> str:
    if not _TODOS:
        return "(no todos)"
    marks = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
    lines = []
    for t in _TODOS:
        mark = marks.get(t["status"], "[?]")
        content = t["content"]
        if t["status"] == "in_progress" and t.get("activeForm"):
            content = t["activeForm"]
        lines.append(f"{mark} {content}")
    return "\n".join(lines)


def tool_todo_write(todos: Any) -> str:
    global _TODOS
    if not isinstance(todos, list):
        return f"Error: 'todos' must be a list, got {type(todos).__name__}"

    normalized: list[dict] = []
    for i, item in enumerate(todos):
        if not isinstance(item, dict):
            return f"Error: todos[{i}] must be an object"
        content = item.get("content")
        status = item.get("status", "pending")
        if not isinstance(content, str) or not content.strip():
            return f"Error: todos[{i}].content must be a non-empty string"
        if status not in _VALID_TODO_STATUS:
            return f"Error: todos[{i}].status must be one of {sorted(_VALID_TODO_STATUS)}"
        entry = {"content": content.strip(), "status": status}
        af = item.get("activeForm")
        if isinstance(af, str) and af.strip():
            entry["activeForm"] = af.strip()
        normalized.append(entry)

    in_progress = sum(1 for t in normalized if t["status"] == "in_progress")
    if in_progress > 1:
        return f"Error: at most ONE todo may be 'in_progress' (got {in_progress})"

    _TODOS = normalized
    return "Todo list updated:\n" + _format_todos()


def tool_todo_read() -> str:
    return _format_todos()


TOOLS = {
    "shell": tool_shell,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
    "list_dir": tool_list_dir,
    "glob": tool_glob,
    "grep": tool_grep,
    "fetch_url": tool_fetch_url,
    "todo_write": tool_todo_write,
    "todo_read": tool_todo_read,
    "memory_read": tool_memory_read,
    "memory_append": tool_memory_append,
    "memory_write": tool_memory_write,
}


def format_tool_call(call: dict) -> str:
    name = call.get("name")
    args = call.get("args") or {}
    if not isinstance(args, dict):
        return f"{name}: {args!r}"
    if name == "shell":
        return f"shell: {args.get('command', '')}"
    if name == "read_file":
        return f"read_file: {args.get('path', '')}"
    if name == "write_file":
        path = args.get("path", "")
        content = args.get("content", "")
        size = len(str(content).encode("utf-8"))
        return f"write_file: {path} ({size} bytes)"
    if name == "edit_file":
        path = args.get("path", "")
        old = str(args.get("old_string", ""))
        new = str(args.get("new_string", ""))
        ra = ", replace_all" if _as_bool(args.get("replace_all"), False) else ""
        return f"edit_file: {path}  ({len(old)} chars → {len(new)} chars{ra})"
    if name == "list_dir":
        p = args.get("path") or "."
        rec = _as_bool(args.get("recursive"), False)
        return f"list_dir: {p}  recursive={rec}"
    if name == "glob":
        p = args.get("path") or "."
        return f"glob: {args.get('pattern', '')}  in {p}"
    if name == "grep":
        p = args.get("path") or "."
        inc = args.get("include") or ""
        inc_s = f"  include={inc}" if inc else ""
        return f"grep: /{args.get('pattern', '')}/  in {p}{inc_s}"
    if name == "fetch_url":
        return f"fetch_url: {args.get('url', '')}"
    if name == "todo_write":
        todos = args.get("todos")
        n = len(todos) if isinstance(todos, list) else "?"
        return f"todo_write: {n} item(s)"
    if name == "todo_read":
        return "todo_read"
    if name == "memory_read":
        return "memory_read"
    if name == "memory_append":
        section = args.get("section", "")
        entry = str(args.get("entry", ""))
        short = entry if len(entry) <= 80 else entry[:77] + "..."
        return f"memory_append: [{section}] {short}"
    if name == "memory_write":
        content = str(args.get("content", ""))
        return f"memory_write: {len(content)} chars"
    return f"{name}: {json.dumps(args, default=str)}"


# ---------------- ToolCall representation ----------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str = ""
    parse_error: str | None = None

    def to_openai(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.raw_arguments or json.dumps(self.arguments),
            },
        }

    def to_ollama(self) -> dict:
        return {"function": {"name": self.name, "arguments": self.arguments}}


def _make_tool_call(call_id: str, name: str, raw_arguments: str) -> ToolCall:
    try:
        args = json.loads(raw_arguments, strict=False) if raw_arguments.strip() else {}
        if not isinstance(args, dict):
            raise ValueError(f"arguments must be a JSON object, got {type(args).__name__}")
        return ToolCall(id=call_id, name=name, arguments=args, raw_arguments=raw_arguments)
    except (json.JSONDecodeError, ValueError) as e:
        return ToolCall(id=call_id, name=name, arguments={}, raw_arguments=raw_arguments,
                        parse_error=str(e))


# ---------------- Validation ----------------

_ALLOWED_ARG_TYPES = (str, bool, int, float, list, dict)


def _normalize_args(name: str, args: dict) -> dict:
    """Drop null / empty-string arguments whose parameter has a default.

    Small models routinely emit {"path": null} or {"path": ""} for optional
    parameters they don't want to set. Let the tool function's own default
    apply instead of failing validation.
    """
    try:
        sig = inspect.signature(TOOLS[name])
    except (KeyError, TypeError, ValueError):
        return args

    out: dict = {}
    for k, v in args.items():
        param = sig.parameters.get(k)
        has_default = param is not None and param.default is not inspect.Parameter.empty
        if has_default and (v is None or v == ""):
            continue
        out[k] = v
    return out


def validate_call(call: dict) -> str | None:
    name = call.get("name")
    if not isinstance(name, str) or name not in TOOLS:
        return f"Error: unknown tool '{name}'. Available: {', '.join(TOOLS)}"
    args = call.get("args")
    if not isinstance(args, dict):
        return f"Error: 'args' for {name} must be a JSON object, got {type(args).__name__}"
    for k, v in args.items():
        if v is None:
            return f"Error: argument '{k}' for {name} is null — omit it or give a value"
        if not isinstance(v, _ALLOWED_ARG_TYPES):
            return (f"Error: argument '{k}' for {name} has unsupported type "
                    f"{type(v).__name__}")
    try:
        inspect.signature(TOOLS[name]).bind(**args)
    except TypeError as e:
        return f"Error: bad arguments for {name}: {e}"
    return None


def execute_tool(call: dict) -> str:
    name = call.get("name")
    if isinstance(name, str) and name in TOOLS and isinstance(call.get("args"), dict):
        call = {"name": name, "args": _normalize_args(name, call["args"])}
    err = validate_call(call)
    if err:
        return err
    try:
        return TOOLS[call["name"]](**call["args"])
    except Exception as e:
        return f"Error running {call['name']}: {e}"


# ---------------- Error hints ----------------

def _hint_for_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()

    if "connection" in low or "refused" in low or "unreachable" in low or "connect" in low:
        return (f"{msg}\n  hint: cannot reach the backend. For ollama, is "
                f"`ollama serve` running and the model pulled? For groq, check "
                f"your network and API key with `/api -s`.")
    if "timed out" in low or "timeout" in low:
        return f"{msg}\n  hint: request timed out — retry, or check your connection."
    if "401" in msg or "unauthorized" in low or "invalid api key" in low or "authentication" in low:
        return (f"{msg}\n  hint: authentication failed. Set or update the Groq "
                f"key with `/api <key>`.")
    if "403" in msg or "forbidden" in low:
        return f"{msg}\n  hint: access forbidden — check your Groq account / model access."
    if "429" in msg or "rate limit" in low or "too many requests" in low:
        return (f"{msg}\n  hint: rate limited. Wait a moment and retry, or "
                f"switch model/provider with `/model` or `/provider`.")
    if "model" in low and ("not found" in low or "does not exist" in low or "no such" in low or "unknown model" in low):
        return (f"{msg}\n  hint: model not available. List models with `/model` "
                f"(no args) or switch with `/model <name>`.")
    if "context length" in low or "too long" in low or "maximum context" in low or "context_length" in low:
        return (f"{msg}\n  hint: conversation is too long. Clear it with `/reset` "
                f"to start fresh.")
    if "tool" in low and ("not supported" in low or "unsupported" in low or "does not support" in low):
        return (f"{msg}\n  hint: this model doesn't support native tool calling. "
                f"Try a different model — for groq: `/model llama-3.3-70b-versatile`; "
                f"for ollama: a tool-capable model like `qwen2.5` or `llama3.1`.")
    if "think" in low:
        return f"{msg}\n  hint: `think` unsupported by this model — /think off."
    return msg


# ---------------- Backends: Ollama ----------------

_THINK_RESOLVED: dict[tuple[str, str], Any] = {}
_MISSING = object()


def _think_candidates(requested):
    cands = []
    cached = _THINK_RESOLVED.get((STATE.model, repr(requested)), _MISSING)
    if cached is not _MISSING:
        cands.append(cached)
    cands.append(requested)
    if isinstance(requested, str):
        cands.append(True)
    cands.append(None)

    out = []
    for c in cands:
        if c not in out:
            out.append(c)
    return out


def _remember_think(requested, think) -> None:
    _THINK_RESOLVED[(STATE.model, repr(requested))] = think


def _is_think_error(e: Exception) -> bool:
    if isinstance(e, (TypeError, ValueError)):
        return True
    resp_err = getattr(ollama, "ResponseError", None) if ollama is not None else None
    if resp_err is not None and isinstance(e, resp_err):
        return "think" in str(e).lower()
    return False


def _require_ollama() -> None:
    if not _OLLAMA_AVAILABLE:
        raise RuntimeError(
            "The 'ollama' package is not installed. Run: pip install ollama"
        )


def _ollama_kwargs(messages, stream: bool, think, tools):
    kw: dict = dict(
        model=STATE.model,
        messages=messages,
        stream=stream,
        options={"temperature": TEMPERATURE},
    )
    if think is not None:
        kw["think"] = think
    if tools:
        kw["tools"] = tools
    return kw


def _ollama_stream_raw(messages, tools):
    requested = STATE.think_param()
    last_err = None
    for think in _think_candidates(requested):
        try:
            it = iter(ollama.chat(**_ollama_kwargs(messages, True, think, tools)))
            first = next(it)
        except StopIteration:
            _remember_think(requested, think)
            return
        except Exception as e:
            if _is_think_error(e):
                last_err = e
                continue
            raise
        _remember_think(requested, think)
        yield first
        yield from it
        return
    if last_err is not None:
        raise last_err


def _ollama_once(messages, tools=None):
    requested = STATE.think_param()
    last_err = None
    for think in _think_candidates(requested):
        try:
            resp = ollama.chat(**_ollama_kwargs(messages, False, think, tools))
        except Exception as e:
            if _is_think_error(e):
                last_err = e
                continue
            raise
        _remember_think(requested, think)
        return resp
    if last_err is not None:
        raise last_err
    return None


def _to_ollama_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args, strict=False)
                    except json.JSONDecodeError:
                        args = {}
                tcs.append({"function": {"name": fn.get("name", ""), "arguments": args}})
            out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs})
        elif role == "tool":
            out.append({"role": "tool", "content": m.get("content", ""),
                        "tool_name": m.get("tool_name", "")})
        else:
            out.append(m)
    return out


def _iter_ollama(messages: list[dict]):
    _require_ollama()
    ollama_msgs = _to_ollama_messages(messages)
    for chunk in _ollama_stream_raw(ollama_msgs, tools=TOOL_SCHEMAS):
        msg = _get(chunk, "message") or {}
        t = _get(msg, "thinking")
        if t:
            yield ("thinking", t)
        c = _get(msg, "content")
        if c:
            yield ("content", c)
        raw_tcs = _get(msg, "tool_calls") or []
        for i, raw_tc in enumerate(raw_tcs):
            fn = _get(raw_tc, "function") or {}
            name = _get(fn, "name") or ""
            args = _get(fn, "arguments")
            if isinstance(args, str):
                raw = args
            elif isinstance(args, dict):
                raw = json.dumps(args)
            else:
                raw = "{}"
            cid = _get(raw_tc, "id") or f"call_{i}"
            yield ("tool_call", _make_tool_call(cid, name, raw))


# ---------------- Backends: Groq ----------------

class _GroqStreamError(RuntimeError):
    """Groq aborted a stream mid-flight."""


def _require_groq() -> None:
    if not _GROQ_AVAILABLE:
        raise RuntimeError(
            "The 'groq' package is not installed. Run: pip install groq"
        )
    if not STATE.groq_api_key:
        raise RuntimeError(
            "No Groq API key is set. Use /api <key> to set one, "
            "or export $AGENT0_GROQ_API_KEY."
        )


def _groq_error_message(e: Exception) -> str:
    low = str(e).lower()
    if "tool_choice" in low and "none" in low:
        return ("Groq rejected the model's tool call (tool_choice mismatch). "
                "Try a different model with `/model llama-3.3-70b-versatile`.")
    if "tool" in low and ("did not call" in low or "no tool" in low):
        return ("Groq rejected the response because the model produced no tool "
                "call where one was required. Retry, or switch models.")
    return str(e)


def _groq_create(messages, *, stream: bool, tools=None):
    client = _GroqClient(api_key=STATE.groq_api_key)
    kw: dict = dict(
        model=STATE.groq_model,
        messages=messages,
        temperature=TEMPERATURE,
        stream=stream,
    )
    if tools:
        kw["tools"] = tools
        kw["tool_choice"] = "auto"
    return client.chat.completions.create(**kw)


def _to_groq_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m.get("role") == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id", ""),
                "content": m.get("content", ""),
            })
        else:
            out.append(m)
    return out


def _iter_groq(messages: list[dict]):
    _require_groq()
    groq_msgs = _to_groq_messages(messages)
    try:
        stream = _groq_create(groq_msgs, stream=True, tools=TOOL_SCHEMAS)
    except Exception as e:
        raise _GroqStreamError(_groq_error_message(e)) from e

    pending: dict[int, dict] = {}

    try:
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            reasoning = getattr(delta, "reasoning", None)
            if reasoning:
                yield ("thinking", reasoning)

            content = getattr(delta, "content", None)
            if content:
                yield ("content", content)

            for tc_delta in getattr(delta, "tool_calls", None) or []:
                idx = getattr(tc_delta, "index", 0)
                idx = 0 if idx is None else idx
                entry = pending.setdefault(idx, {"id": None, "name": None, "args": []})
                tcid = getattr(tc_delta, "id", None)
                if tcid:
                    entry["id"] = tcid
                fn = getattr(tc_delta, "function", None)
                if fn is not None:
                    fn_name = getattr(fn, "name", None)
                    if fn_name:
                        entry["name"] = fn_name
                    fn_args = getattr(fn, "arguments", None)
                    if fn_args:
                        entry["args"].append(fn_args)
    except _GroqStreamError:
        raise
    except Exception as e:
        raise _GroqStreamError(_groq_error_message(e)) from e

    for idx in sorted(pending):
        entry = pending[idx]
        raw = "".join(entry["args"]) or "{}"
        yield ("tool_call", _make_tool_call(
            entry["id"] or f"call_{idx}", entry["name"] or "", raw,
        ))


def _groq_once(messages: list[dict]):
    _require_groq()
    try:
        resp = _groq_create(_to_groq_messages(messages), stream=False, tools=None)
    except Exception as e:
        raise _GroqStreamError(_groq_error_message(e)) from e
    text = ""
    if getattr(resp, "choices", None):
        msg = getattr(resp.choices[0], "message", None)
        if msg is not None:
            text = getattr(msg, "content", "") or ""
    return {"message": {"content": text}}


# ---------------- Dispatchers ----------------

def _iter_stream(messages: list[dict]):
    if STATE.provider == "groq":
        yield from _iter_groq(messages)
    else:
        yield from _iter_ollama(messages)


def _chat_once(messages: list[dict]) -> dict:
    if STATE.provider == "groq":
        return _groq_once(messages)
    resp = _ollama_once(messages, tools=None)
    return resp or {"message": {"content": ""}}


def stream_chat(messages: list[dict]) -> tuple[str, str, list[ToolCall], bool]:
    thinking_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    in_thinking = False
    thinking_done = False
    printed = False

    def _close_thinking() -> None:
        nonlocal thinking_done
        if in_thinking and not thinking_done:
            if STATE.show_thinking:
                print(RESET, flush=True)
            else:
                print(f" {GREEN}Done!{RESET}", flush=True)
            thinking_done = True

    try:
        for kind, payload in _iter_stream(messages):
            if kind == "thinking":
                if not in_thinking:
                    if STATE.show_thinking:
                        print(f"{GRAY}Thinking:{RESET} {DIM}", end="", flush=True)
                    else:
                        print(f"{GRAY}Thinking...{RESET}", end="", flush=True)
                    in_thinking = True
                if STATE.show_thinking:
                    print(payload, end="", flush=True)
                thinking_parts.append(payload)
            elif kind == "content":
                if in_thinking and not thinking_done:
                    _close_thinking()
                elif not in_thinking and not thinking_done:
                    thinking_done = True
                content_parts.append(payload)
                print(payload, end="", flush=True)
                printed = True
            elif kind == "tool_call":
                tool_calls.append(payload)
    except _GroqStreamError as e:
        _close_thinking()
        content = f"(groq error) {_hint_for_error(e)}"
        print(f"\n{RED}{content}{RESET}", flush=True)
        return "".join(thinking_parts), content, [], True

    _close_thinking()
    if printed:
        print()

    content = "".join(content_parts).strip()
    return "".join(thinking_parts), content, tool_calls, printed


_THINK_BLOCK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text)


def summarize_tool_call(call: dict) -> str:
    name = call.get("name")
    args = call.get("args") or {}
    payload = json.dumps({"tool": name, "args": args}, indent=2, ensure_ascii=False, default=str)

    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Tool call:\n{payload}"},
    ]
    try:
        resp = _chat_once(messages)
        text = (_get(_get(resp, "message") or {}, "content") or "").strip()
        text = _strip_think(text).strip()
        if text:
            return " ".join(text.split())
    except KeyboardInterrupt:
        return "(summary interrupted)"
    except Exception as e:
        return f"(could not summarize: {e})"

    return f"Will run {name} with {args}"


# ---------------- Confirmation ----------------

def confirm_tool_call(call: dict, *, label: str = "") -> bool:
    print(f"{YELLOW}{label}{format_tool_call(call)}{RESET}")
    summary = summarize_tool_call(call)
    print(f"{CYAN}{label}{summary}{RESET}")

    while True:
        try:
            ans = input(f"{BOLD}{label}[Y]es/[N]o? {RESET}").strip().lower()
        except KeyboardInterrupt:
            print(f"\n{CTRL_C_HINT}")
            continue
        except EOFError:
            print(f"\n{CTRL_C_HINT}")
            continue

        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


# ---------------- Agent loop ----------------

def _assistant_message(content: str, tool_calls: list[ToolCall]) -> dict:
    msg: dict = {"role": "assistant", "content": content or ""}
    if tool_calls:
        msg["tool_calls"] = [tc.to_openai() for tc in tool_calls]
    return msg


def _tool_message(tc: ToolCall, content: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tc.id,
        "tool_name": tc.name,
        "content": content,
    }


def _handle_one_call(tc: ToolCall, label: str) -> str:
    if tc.parse_error:
        msg = (f"Error: could not parse arguments for '{tc.name}': {tc.parse_error}. "
               f"Raw was: {tc.raw_arguments!r}")
        print(f"{RED}{label}{msg}{RESET}")
        return msg

    args = _normalize_args(tc.name, tc.arguments) if tc.name in TOOLS else tc.arguments
    call = {"name": tc.name, "args": args}

    err = validate_call(call)
    if err:
        print(f"{RED}{label}{err}{RESET}")
        return err

    needs_confirm = not _autonomy_allows(tc.name)

    if needs_confirm:
        if not confirm_tool_call(call, label=label):
            print(f"{GRAY}{label}Denied.{RESET}")
            return "User DENIED this tool call. It was NOT executed."
    else:
        policy = _tool_policy(tc.name)
        print(f"{MAGENTA}{label}[auto:{policy}] {format_tool_call(call)}{RESET}")

    try:
        result = TOOLS[tc.name](**args)
    except Exception as e:
        result = f"Error running {tc.name}: {e}"

    preview = (
        result if len(result) <= TOOL_PREVIEW_LIMIT
        else result[:TOOL_PREVIEW_LIMIT] + " ..."
    )
    print(f"{GRAY}{label}{preview}{RESET}")
    return result


def run_agent(user_input: str, history: list) -> None:
    history[0] = {"role": "system", "content": build_system_prompt()}
    history.append({"role": "user", "content": user_input})

    for _ in range(1, MAX_STEPS + 1):
        try:
            _, content, tool_calls, printed = stream_chat(history)
        except KeyboardInterrupt:
            print(f"\n{CTRL_C_HINT}")
            return
        except Exception as e:
            print(f"{RED}[error] {_hint_for_error(e)}{RESET}")
            return

        if not content and not tool_calls:
            if not printed:
                print(f"{GRAY}[empty reply — stopping]{RESET}")
            return

        history.append(_assistant_message(content, tool_calls))

        if not tool_calls:
            if not printed and content:
                print(content)
            return

        batch = len(tool_calls)
        for idx, tc in enumerate(tool_calls, 1):
            label = f"[{idx}/{batch}] " if batch > 1 else ""
            result = _handle_one_call(tc, label)
            history.append(_tool_message(tc, result))

    print(f"{RED}Hit max steps ({MAX_STEPS}); stopping this turn.{RESET}")


# ---------------- Slash commands ----------------

def _list_local_models() -> list[str]:
    if not _OLLAMA_AVAILABLE:
        return []
    try:
        resp = ollama.list()
    except Exception as e:
        warn(f"could not list local models: {e}")
        return []

    entries = getattr(resp, "models", None)
    if entries is None and isinstance(resp, dict):
        entries = resp.get("models", [])

    out: list[str] = []
    for m in entries or []:
        name = getattr(m, "model", None) or getattr(m, "name", None)
        if name is None and isinstance(m, dict):
            name = m.get("model") or m.get("name")
        if name:
            out.append(name)
    return out


def _list_groq_models() -> list[str]:
    if not _GROQ_AVAILABLE or not STATE.groq_api_key:
        return []
    try:
        client = _GroqClient(api_key=STATE.groq_api_key)
        resp = client.models.list()
        return [m.id for m in getattr(resp, "data", []) if getattr(m, "id", None)]
    except Exception as e:
        warn(f"could not list groq models: {e}")
        return []


def cmd_model(arg: str) -> None:
    arg = arg.strip()
    if not arg:
        print(f"{GRAY}provider: {CYAN}{STATE.provider}{RESET}")
        print(f"{GRAY}model   : {CYAN}{STATE.current_model()}{RESET}")
        print(f"{GRAY}usage: /model <name>{RESET}")
        return

    name = arg

    if STATE.provider == "groq":
        available = _list_groq_models()
        if available and name not in available:
            print(f"{YELLOW}note:{RESET} '{name}' is not in the Groq model list.")
            print(f"{GRAY}available: {', '.join(available[:12])}"
                  f"{' …' if len(available) > 12 else ''}{RESET}")
        STATE.set_groq_model(name)
        print(f"{GREEN}groq model set to {CYAN}{name}{RESET} "
              f"{GRAY}(saved to {CONFIG_PATH}){RESET}")
    else:
        available = _list_local_models()
        if available and name not in available and f"{name}:latest" not in available:
            print(f"{YELLOW}note:{RESET} '{name}' is not in the local Ollama model list.")
            print(f"{GRAY}available: {', '.join(available)}{RESET}")
        STATE.set_model(name)
        print(f"{GREEN}ollama model set to {CYAN}{name}{RESET} "
              f"{GRAY}(saved to {CONFIG_PATH}){RESET}")


def cmd_provider(arg: str) -> None:
    arg = arg.strip().lower()
    if not arg:
        print(f"{GRAY}current provider: {CYAN}{STATE.provider}{RESET}")
        print(f"{GRAY}usage: /provider ollama|groq{RESET}")
        return
    if arg not in ("ollama", "groq"):
        print(f"{YELLOW}expected 'ollama' or 'groq'{RESET}")
        return
    if arg == "groq":
        if not _GROQ_AVAILABLE:
            print(f"{RED}The 'groq' package is not installed. Run: pip install groq{RESET}")
            return
        if not STATE.groq_api_key:
            print(f"{RED}No Groq API key set. Use /api <key> first.{RESET}")
            return
    if arg == "ollama" and not _OLLAMA_AVAILABLE:
        print(f"{RED}The 'ollama' package is not installed. Run: pip install ollama{RESET}")
        return

    STATE.set_provider(arg)
    print(f"{GREEN}provider set to {CYAN}{arg}{RESET} "
          f"{GRAY}(model: {STATE.current_model()}){RESET}")


AUTO_USAGE = (
    f"{GRAY}usage:\n"
    f"  /auto manual    confirm every tool call\n"
    f"  /auto smart     auto read-only tools\n"
    f"  /auto auto      auto read-only + file edits (default)\n"
    f"  /auto yolo      auto everything, including shell and network\n"
    f"  /auto           show current level{RESET}"
)

_AUTO_DESCRIPTIONS = {
    "manual": "confirm every tool call",
    "smart":  "auto read-only tools",
    "auto":   "auto read-only + file edits",
    "yolo":   "auto everything (including shell and network)",
}


def cmd_auto(arg: str) -> None:
    arg = arg.strip().lower()
    if not arg:
        print(f"{GRAY}autonomy: {CYAN}{STATE.autonomy}{RESET} "
              f"{GRAY}— {_AUTO_DESCRIPTIONS.get(STATE.autonomy, '')}{RESET}")
        print(AUTO_USAGE)
        return
    if arg not in AUTONOMY_LEVELS:
        print(f"{YELLOW}expected one of: {', '.join(AUTONOMY_ORDER)}{RESET}")
        print(AUTO_USAGE)
        return
    if arg == "yolo":
        print(f"{YELLOW}warning:{RESET} 'yolo' auto-approves ALL tool calls, "
              f"including shell commands and network access. Dangerous.")
        try:
            ans = input(f"{BOLD}Enable yolo? [y/N] {RESET}").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if ans not in ("y", "yes"):
            print(f"{GRAY}cancelled{RESET}")
            return
    STATE.set_autonomy(arg)
    print(f"{GREEN}autonomy set to {CYAN}{arg}{RESET} "
          f"{GRAY}({_AUTO_DESCRIPTIONS[arg]}){RESET}")


MEMORY_USAGE = (
    f"{GRAY}usage:\n"
    f"  /memory        show the memory file's path and contents\n"
    f"  /memory -r     clear the memory file (with confirmation)\n"
    f"  /memory -e     open the memory file in $EDITOR{RESET}"
)


def cmd_memory(arg: str) -> None:
    arg = arg.strip()

    if not arg:
        print(f"{GRAY}memory file: {CYAN}{MEMORY_PATH}{RESET}")
        text = _load_memory_text()
        if not text:
            print(f"{GRAY}(empty){RESET}")
        else:
            print()
            print(text)
        return

    if arg == "-r":
        try:
            ans = input(f"{BOLD}Clear memory? [y/N] {RESET}").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if ans not in ("y", "yes"):
            print(f"{GRAY}cancelled{RESET}")
            return
        try:
            MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            MEMORY_PATH.write_text(_MEMORY_DEFAULT, encoding="utf-8")
            print(f"{GREEN}memory reset to default{RESET}")
        except OSError as e:
            print(f"{RED}could not reset memory: {e}{RESET}")
        return

    if arg == "-e":
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
        if not editor:
            for cand in ("nano", "vim", "vi", "code"):
                if shutil.which(cand):
                    editor = cand
                    break
        if not editor:
            print(f"{RED}no $EDITOR set and no common editor found{RESET}")
            return
        try:
            MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            if not MEMORY_PATH.exists():
                MEMORY_PATH.write_text(_MEMORY_DEFAULT, encoding="utf-8")
            subprocess.run([*shlex.split(editor), str(MEMORY_PATH)], check=False)
        except Exception as e:
            print(f"{RED}could not open editor: {e}{RESET}")
        return

    print(f"{YELLOW}unknown flag: {arg}{RESET}")
    print(MEMORY_USAGE)


API_USAGE = (
    f"{GRAY}usage:\n"
    f"  /api <key>       set the Groq API key and switch to the groq provider\n"
    f"  /api -s          show the masked key and current provider\n"
    f"  /api -r          remove the key and switch back to ollama\n"
    f"  /api             show this help{RESET}"
)


def _show_api_status() -> None:
    if not _GROQ_AVAILABLE:
        print(f"{YELLOW}note:{RESET} the 'groq' package is not installed. "
              f"Run: {CYAN}pip install groq{RESET}")
    print(f"{GRAY}provider      : {CYAN}{STATE.provider}{RESET}")
    print(f"{GRAY}groq api key  : {CYAN}{_mask_api_key(STATE.groq_api_key)}{RESET}")
    print(f"{GRAY}groq model    : {CYAN}{STATE.groq_model}{RESET}")
    if STATE._key_from_env:
        print(f"{GRAY}(key sourced from $AGENT0_GROQ_API_KEY){RESET}")
    print()
    print(API_USAGE)


def cmd_api(arg: str) -> None:
    arg = arg.strip()

    if not arg or arg == "-s":
        _show_api_status()
        return

    if arg == "-r":
        if not STATE.groq_api_key:
            print(f"{GRAY}No API key is set.{RESET}")
            return
        STATE.set_groq_key("")
        if STATE.provider == "groq":
            STATE.set_provider("ollama")
        print(f"{GREEN}Groq API key cleared.{RESET}")
        return

    if arg.startswith("-"):
        print(f"{YELLOW}unknown flag: {arg}{RESET}")
        print(API_USAGE)
        return

    if not _GROQ_AVAILABLE:
        print(f"{RED}The 'groq' package is not installed. Run: pip install groq{RESET}")
        return

    STATE.set_groq_key(arg)
    if STATE.provider != "groq":
        STATE.set_provider("groq")
    print(f"{GREEN}Groq API key saved: {CYAN}{_mask_api_key(arg)}{RESET}")
    print(f"{GRAY}provider is now {CYAN}groq{GRAY}, model {CYAN}{STATE.groq_model}{RESET}")


THINK_USAGE = (
    f"{GRAY}usage:\n"
    f"  /think on|off        enable/disable thinking\n"
    f"  /think -s on|off     show/hide the thinking text\n"
    f"  /think -f on|off     fast thinking (low-effort)\n"
    f"  /think               show current settings{RESET}"
)


def cmd_think(arg: str) -> None:
    parts = arg.strip().split()

    if not parts:
        print(f"{GRAY}thinking       : {CYAN}{'on' if STATE.think_enabled else 'off'}{RESET}")
        print(f"{GRAY}show thinking  : {CYAN}{'on' if STATE.show_thinking else 'off'}{RESET}")
        print(f"{GRAY}fast thinking  : {CYAN}{'on' if STATE.fast_think else 'off'}{RESET}")
        print(THINK_USAGE)
        return

    if len(parts) == 1:
        v = parts[0].lower()
        if v in ("on", "off"):
            STATE.set_thinking(think=(v == "on"))
            print(f"{GREEN}thinking {'on' if STATE.think_enabled else 'off'}{RESET} "
                  f"{GRAY}(saved){RESET}")
            return
        print(THINK_USAGE)
        return

    if len(parts) == 2:
        flag, v = parts[0], parts[1].lower()
        if v not in ("on", "off"):
            print(f"{YELLOW}expected on|off{RESET}")
            print(THINK_USAGE)
            return
        b = (v == "on")
        if flag == "-s":
            STATE.set_thinking(show=b)
            print(f"{GREEN}show thinking {'on' if b else 'off'}{RESET} {GRAY}(saved){RESET}")
            return
        if flag == "-f":
            STATE.set_thinking(fast=b)
            print(f"{GREEN}fast thinking {'on' if b else 'off'}{RESET} {GRAY}(saved){RESET}")
            return

    print(THINK_USAGE)


def cmd_tools() -> None:
    print(f"{BOLD}Available tools{RESET}")
    for name, desc in TOOL_DESCRIPTIONS.items():
        policy = _tool_policy(name)
        policy_color = {"safe": GREEN, "medium": YELLOW, "dangerous": RED}.get(policy, GRAY)
        auto = "auto" if _autonomy_allows(name) else "ask "
        print(f"  {CYAN}{name:<14}{RESET}{policy_color}{policy:<10}{RESET}"
              f"{GRAY}{auto}  {desc}{RESET}")


def cmd_todos() -> None:
    print(f"{BOLD}Session todos{RESET}")
    print(_format_todos())


def show_help() -> None:
    print(f"""{BOLD}Commands{RESET}
  {CYAN}/model{RESET} <name>          switch model (for the current provider) and persist
  {CYAN}/model{RESET}                 show current provider and model
  {CYAN}/provider{RESET} ollama|groq   switch backend
  {CYAN}/api{RESET} <key>             set the Groq API key and switch to groq
  {CYAN}/api -s{RESET}                show the masked Groq API key
  {CYAN}/api -r{RESET}                remove the Groq API key and switch back to ollama
  {CYAN}/auto{RESET} <level>          set autonomy (manual|smart|auto|yolo)
  {CYAN}/auto{RESET}                  show current autonomy level
  {CYAN}/memory{RESET}                show the memory file
  {CYAN}/memory -r{RESET}             reset the memory file
  {CYAN}/memory -e{RESET}             edit the memory file in $EDITOR
  {CYAN}/think{RESET} on|off           enable/disable thinking (ollama only)
  {CYAN}/think -s{RESET} on|off        show/hide the thinking text
  {CYAN}/think -f{RESET} on|off        fast thinking (low-effort)
  {CYAN}/think{RESET}                  show current thinking settings
  {CYAN}/tools{RESET}                  list the tools the agent can call
  {CYAN}/todos{RESET}                  show the session todo list
  {CYAN}/reset{RESET}                 clear the conversation history and todos
  {CYAN}/help{RESET}                  show this help
  {CYAN}/bye{RESET}                   quit (Ctrl-C / Ctrl-D do nothing)
""")


# ---------------- Entrypoint ----------------

def main() -> None:
    if not sys.stdin.isatty():
        print("This script needs an interactive terminal.", file=sys.stderr)
        sys.exit(1)

    _memory_init()

    print(WELCOME)
    print(f"{GRAY}autonomy: {STATE.autonomy}    "
          f"provider: {STATE.provider}    model: {STATE.current_model()}{RESET}")
    print()

    history: list = [{"role": "system", "content": build_system_prompt()}]

    while True:
        try:
            user_input = input(f"{BOLD}>>> {RESET}")
        except KeyboardInterrupt:
            print(f"\n{CTRL_C_HINT}")
            continue
        except EOFError:
            print(f"\n{CTRL_C_HINT}")
            continue

        stripped = user_input.strip()
        if not stripped:
            continue

        if stripped.startswith("/"):
            parts = stripped.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/bye", "/exit", "/quit"):
                break
            if cmd == "/help":
                show_help()
                continue
            if cmd == "/tools":
                cmd_tools()
                continue
            if cmd == "/todos":
                cmd_todos()
                continue
            if cmd == "/reset":
                history = [{"role": "system", "content": build_system_prompt()}]
                _reset_todos()
                print(f"{GRAY}history and todos cleared (memory kept){RESET}")
                continue
            if cmd == "/model":
                cmd_model(arg)
                continue
            if cmd == "/provider":
                cmd_provider(arg)
                continue
            if cmd == "/api":
                cmd_api(arg)
                continue
            if cmd == "/auto":
                cmd_auto(arg)
                continue
            if cmd == "/memory":
                cmd_memory(arg)
                continue
            if cmd == "/think":
                cmd_think(arg)
                continue

            print(f"{YELLOW}unknown command:{RESET} {cmd}  {GRAY}(try /help){RESET}")
            continue

        try:
            run_agent(stripped, history)
        except KeyboardInterrupt:
            print(f"\n{CTRL_C_HINT}")
        print()

    print(f"{GRAY}bye.{RESET}")


if __name__ == "__main__":
    main()