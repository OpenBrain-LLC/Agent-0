#!/usr/bin/env python3
"""Agent-0 — autonomous text-tool AI agent with Ollama + Groq backends.

Deployment
----------
* Config file:     db/local/.config.json   (override with $AGENT0_CONFIG)
* Memory file:     db/local/memory.md      (override with $AGENT0_MEMORY)
* Session file:    db/local/session.json   (override with $AGENT0_SESSION)
* Ollama model:    deepseek-r1:1.5b        (override with $AGENT0_MODEL)
* Groq API key:    set via /api <key>, or export $AGENT0_GROQ_API_KEY
* Brave key:       /search key brave <key> or $AGENT0_BRAVE_KEY
* Tavily key:      /search key tavily <key> or $AGENT0_TAVILY_KEY
* Requires:        pip install ollama groq
* Needs an interactive TTY.

Web search
----------
* DuckDuckGo (default) — no key required; scrapes html.duckduckgo.com with
  a fallback to lite.duckduckgo.com.
* Brave Search — free tier 2k queries/month. Set key: /search key brave <key>.
* Tavily — free tier 1k queries/month, LLM-optimized. Set: /search key tavily <key>.

Security (v5.0.1)
-----------------
* Agent-0's own state files (session.json, .config.json, and anything in
  db/local/ except memory.md) are protected: file tools refuse to read,
  write, or edit them. Prevents the model from reading its own system
  prompt or the API keys in config.
* State files are written with mode 0600 (owner read/write only).
* The system prompt is stripped from persisted session history so a shared
  session.json cannot leak the full instructions.

Autonomy levels (see /auto): manual / smart / auto (default) / yolo.
"""

from __future__ import annotations

import datetime
import fnmatch
import html as _html
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
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
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


__version__ = "1.1.0"


# ---------------- ANSI ----------------

def _colors_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


if _colors_enabled():
    BOLD, DIM, GRAY = "\033[1m", "\033[2m", "\033[90m"
    CYAN, YELLOW, GREEN, RED, MAGENTA, BLUE = (
        "\033[36m", "\033[33m", "\033[32m", "\033[31m", "\033[35m", "\033[34m",
    )
    RESET = "\033[0m"
else:
    BOLD = DIM = GRAY = CYAN = YELLOW = GREEN = RED = MAGENTA = BLUE = RESET = ""


# ---------------- Config ----------------

SCRIPT_DIR = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else default


CONFIG_PATH = _env_path("AGENT0_CONFIG", SCRIPT_DIR / "db" / "local" / ".config.json")
MEMORY_PATH = _env_path("AGENT0_MEMORY", SCRIPT_DIR / "db" / "local" / "memory.md")
SESSION_PATH = _env_path("AGENT0_SESSION", SCRIPT_DIR / "db" / "local" / "session.json")

# ---------------- Protected paths ----------------
# Agent-0's own state files. The file tools refuse to touch these so the
# model cannot exfiltrate its own system prompt or the API keys in .config.json.

PROTECTED_PATHS: set[Path] = {SESSION_PATH.resolve(), CONFIG_PATH.resolve()}

# Filenames that appear inside db/local/ and must never be touched by tools.
_PROTECTED_NAMES = {"session.json", ".config.json", "goals.db", "audit.jsonl"}


def _is_protected(p: Path) -> bool:
    """True if `p` resolves to an Agent-0 internal state file that tools must not touch.

    memory.md is deliberately allowed — the model accesses it via memory_read /
    memory_append, and the underlying file is fine to expose.
    """
    try:
        resolved = p.expanduser().resolve()
    except (OSError, ValueError):
        return False

    if resolved in PROTECTED_PATHS:
        return True

    # Anything else inside db/local/, except memory.md, is off-limits.
    try:
        base = (SCRIPT_DIR / "db" / "local").resolve()
    except (OSError, ValueError):
        return False
    if resolved == base or base in resolved.parents:
        if resolved.name == "memory.md":
            return False
        return True

    # Fallback: match by known filename anywhere on disk.
    if resolved.name in _PROTECTED_NAMES:
        return True

    return False


def _secure_write(path: Path, content: str) -> None:
    """Write state files with 0600 permissions on POSIX."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


DEFAULT_OLLAMA_MODEL = os.environ.get("AGENT0_MODEL") or "deepseek-r1:1.5b"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

DEFAULT_CONFIG = {
    "model": DEFAULT_OLLAMA_MODEL,
    "provider": "ollama",
    "autonomy": "auto",
    "budget": {"max_steps": 50, "max_tool_calls": 200, "max_seconds": 900},
    "groq": {"api_key": "", "model": DEFAULT_GROQ_MODEL},
    "search": {
        "backend": "duckduckgo",     # duckduckgo | brave | tavily
        "brave_api_key": "",
        "tavily_api_key": "",
        "max_results": 8,
    },
    "thinking": {"show": "True", "think": "True", "fast": "False"},
}

MAX_STEPS = 24
TOOL_OUTPUT_LIMIT = 20000
TOOL_PREVIEW_LIMIT = 500
MEMORY_INJECT_LIMIT = 6000
MEMORY_MAX_BYTES = 200_000
SESSION_HISTORY_CAP = 60
SHELL_TIMEOUT = 60
FETCH_TIMEOUT = 20
FETCH_MAX_BYTES = 500_000
SEARCH_TIMEOUT = 15
SEARCH_MAX_BYTES = 2_000_000
TEMPERATURE = 0.25
MAX_SUBAGENT_DEPTH = 2

CTRL_C_HINT = f"{GRAY}Ctrl-C is disabled — type /bye to exit{RESET}"
WELCOME = f"Welcome to Agent-0 AI [v{__version__}]\nBy: OpenBrain-LLC"

_USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


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


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


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
        "budget": dict(DEFAULT_CONFIG["budget"]),
        "groq": {"api_key": "", "model": DEFAULT_GROQ_MODEL},
        "search": dict(DEFAULT_CONFIG["search"]),
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
        b = data.get("budget")
        if isinstance(b, dict):
            for k in ("max_steps", "max_tool_calls", "max_seconds"):
                if k in b:
                    cfg["budget"][k] = _as_int(b[k], cfg["budget"][k])
        g = data.get("groq")
        if isinstance(g, dict):
            if isinstance(g.get("api_key"), str):
                cfg["groq"]["api_key"] = g["api_key"]
            if isinstance(g.get("model"), str):
                cfg["groq"]["model"] = g["model"]
        s = data.get("search")
        if isinstance(s, dict):
            if s.get("backend") in ("duckduckgo", "brave", "tavily"):
                cfg["search"]["backend"] = s["backend"]
            if isinstance(s.get("brave_api_key"), str):
                cfg["search"]["brave_api_key"] = s["brave_api_key"]
            if isinstance(s.get("tavily_api_key"), str):
                cfg["search"]["tavily_api_key"] = s["tavily_api_key"]
            if "max_results" in s:
                cfg["search"]["max_results"] = _as_int(s["max_results"], 8)
        th = data.get("thinking")
        if isinstance(th, dict):
            for k in ("show", "think", "fast"):
                if k in th:
                    cfg["thinking"][k] = th[k]
    except (json.JSONDecodeError, OSError, ValueError) as e:
        warn(f"could not read config ({e}); using defaults")
    return cfg


def save_config(cfg: dict) -> None:
    _secure_write(CONFIG_PATH, json.dumps(cfg, indent=4) + "\n")


def _mask_api_key(key: str) -> str:
    if not key:
        return "(not set)"
    if len(key) < 12:
        return "*" * len(key)
    return f"{key[:min(7, len(key) - 5)]}****...**{key[-5:]}"


class State:
    def __init__(self) -> None:
        self.config = load_config()
        self.model = self.config["model"]
        self.provider = self.config.get("provider", "ollama")
        self.autonomy = self.config.get("autonomy", "auto")

        b = self.config.get("budget") or {}
        self.budget_max_steps = _as_int(b.get("max_steps"), 50)
        self.budget_max_tools = _as_int(b.get("max_tool_calls"), 200)
        self.budget_max_seconds = _as_int(b.get("max_seconds"), 900)

        g = self.config.get("groq") or {}
        self.groq_model = g.get("model", DEFAULT_GROQ_MODEL)
        env_key = os.environ.get("AGENT0_GROQ_API_KEY", "")
        self._key_from_env = bool(env_key)
        self._persisted_groq_key = g.get("api_key", "")
        self.groq_api_key = env_key or self._persisted_groq_key

        s = self.config.get("search") or {}
        self.search_backend = s.get("backend", "duckduckgo")
        env_brave = os.environ.get("AGENT0_BRAVE_KEY", "")
        env_tavily = os.environ.get("AGENT0_TAVILY_KEY", "")
        self.brave_api_key = env_brave or s.get("brave_api_key", "")
        self.tavily_api_key = env_tavily or s.get("tavily_api_key", "")
        self.search_max_results = _as_int(s.get("max_results"), 8)
        self._brave_from_env = bool(env_brave)
        self._tavily_from_env = bool(env_tavily)

        th = self.config["thinking"]
        self.show_thinking = _to_bool(th.get("show"), False)
        self.think_enabled = _to_bool(th.get("think"), True)
        self.fast_think = _to_bool(th.get("fast"), False)

        self._sync_config()

    def _sync_config(self) -> None:
        self.config["model"] = self.model
        self.config["provider"] = self.provider
        self.config["autonomy"] = self.autonomy
        self.config["budget"] = {
            "max_steps": self.budget_max_steps,
            "max_tool_calls": self.budget_max_tools,
            "max_seconds": self.budget_max_seconds,
        }
        groq_cfg: dict = {"model": self.groq_model}
        if self._key_from_env:
            if self._persisted_groq_key:
                groq_cfg["api_key"] = self._persisted_groq_key
        else:
            groq_cfg["api_key"] = self.groq_api_key
        self.config["groq"] = groq_cfg

        search_cfg: dict = {
            "backend": self.search_backend,
            "max_results": self.search_max_results,
        }
        if not self._brave_from_env:
            search_cfg["brave_api_key"] = self.brave_api_key
        elif self.config.get("search", {}).get("brave_api_key"):
            search_cfg["brave_api_key"] = self.config["search"]["brave_api_key"]
        if not self._tavily_from_env:
            search_cfg["tavily_api_key"] = self.tavily_api_key
        elif self.config.get("search", {}).get("tavily_api_key"):
            search_cfg["tavily_api_key"] = self.config["search"]["tavily_api_key"]
        self.config["search"] = search_cfg

        self.config["thinking"] = {
            "show": "True" if self.show_thinking else "False",
            "think": "True" if self.think_enabled else "False",
            "fast": "True" if self.fast_think else "False",
        }

    def _persist(self) -> None:
        try:
            save_config(self.config)
        except OSError as e:
            warn(f"could not persist config to {CONFIG_PATH}: {e}")

    def set_model(self, name: str) -> None:
        self.model = name; self._sync_config(); self._persist()

    def set_groq_model(self, name: str) -> None:
        self.groq_model = name; self._sync_config(); self._persist()

    def set_groq_key(self, key: str) -> None:
        self.groq_api_key = key
        self._key_from_env = False
        self._persisted_groq_key = key
        self._sync_config(); self._persist()

    def set_provider(self, provider: str) -> None:
        if provider not in ("ollama", "groq"):
            raise ValueError(f"unknown provider: {provider}")
        self.provider = provider; self._sync_config(); self._persist()

    def set_autonomy(self, level: str) -> None:
        if level not in AUTONOMY_LEVELS:
            raise ValueError(f"unknown autonomy level: {level}")
        self.autonomy = level; self._sync_config(); self._persist()

    def set_budget(self, *, steps=None, tools=None, seconds=None) -> None:
        if steps is not None: self.budget_max_steps = max(1, int(steps))
        if tools is not None: self.budget_max_tools = max(1, int(tools))
        if seconds is not None: self.budget_max_seconds = max(5, int(seconds))
        self._sync_config(); self._persist()

    def set_search_backend(self, name: str) -> None:
        if name not in ("duckduckgo", "brave", "tavily"):
            raise ValueError(f"unknown search backend: {name}")
        self.search_backend = name; self._sync_config(); self._persist()

    def set_brave_key(self, key: str) -> None:
        self.brave_api_key = key; self._brave_from_env = False
        self._sync_config(); self._persist()

    def set_tavily_key(self, key: str) -> None:
        self.tavily_api_key = key; self._tavily_from_env = False
        self._sync_config(); self._persist()

    def set_search_max_results(self, n: int) -> None:
        self.search_max_results = max(1, min(20, n))
        self._sync_config(); self._persist()

    def set_thinking(self, *, show=None, think=None, fast=None) -> None:
        if show is not None: self.show_thinking = bool(show)
        if think is not None: self.think_enabled = bool(think)
        if fast is not None: self.fast_think = bool(fast)
        self._sync_config(); self._persist()

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

SAFE_TOOLS = {"read_file", "list_dir", "glob", "grep", "todo_read",
              "todo_write", "memory_read"}
MEDIUM_TOOLS = {"write_file", "edit_file", "memory_write", "memory_append",
                "web_search", "fetch_url"}
DANGEROUS_TOOLS = {"shell", "spawn_subagent"}

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
    level = AUTONOMY_LEVELS.get(STATE.autonomy, AUTONOMY_LEVELS["auto"])
    return bool(level.get(_tool_policy(name), False))


# ---------------- Guardrails ----------------

BLOCKED_SHELL_PATTERNS = [
    (r"\brm\s+-rf?\s+/(?:\s|$)", "rm -rf /"),
    (r":\(\)\s*\{\s*:\|:&\s*\}\s*;:", "fork bomb"),
    (r"\bdd\b[^\n]*\bof=/dev/(?!null)", "raw device write"),
    (r"\bmkfs\.", "filesystem format"),
    (r"\bgit\s+push\b[^\n]*--force[^\n]*\b(main|master|trunk)\b",
     "force-push to protected branch"),
    (r"\bgit\s+push\b[^\n]*\s-f\b[^\n]*\b(main|master|trunk)\b",
     "force-push to protected branch"),
    (r"\bchmod\s+-R\s+777\s+/(?:\s|$)", "chmod -R 777 /"),
    (r"\b(?:curl|wget)\b[^\n]*\|\s*(?:sh|bash|zsh|fish)\b", "pipe-to-shell"),
    (r"\bsudo\s+rm\s+-rf?\s+/(?:\s|$)", "sudo rm -rf /"),
]

BLOCKED_WRITE_ROOTS = [
    Path("/etc"), Path("/usr"), Path("/bin"), Path("/sbin"),
    Path("/boot"), Path("/sys"), Path("/proc"), Path("/dev"),
]


def guard_tool_call(tc: "ToolCall") -> str | None:
    if tc.name == "shell":
        cmd = str(tc.arguments.get("command", ""))
        for pat, label in BLOCKED_SHELL_PATTERNS:
            try:
                if re.search(pat, cmd, re.IGNORECASE):
                    return f"guardrail blocked: {label}"
            except re.error:
                continue
    if tc.name in ("write_file", "edit_file"):
        raw = tc.arguments.get("path")
        if raw:
            try:
                p = Path(str(raw)).expanduser().resolve()
                for root in BLOCKED_WRITE_ROOTS:
                    if p == root or root in p.parents or str(p).startswith(str(root) + os.sep):
                        return f"guardrail blocked: writes under {root} are not permitted"
            except (OSError, ValueError):
                pass
    return None


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
        _secure_write(MEMORY_PATH, _MEMORY_DEFAULT)
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
        encoded = content.encode("utf-8")
        if len(encoded) > MEMORY_MAX_BYTES:
            return f"Error: memory too large ({len(encoded)} > {MEMORY_MAX_BYTES})"
        _secure_write(MEMORY_PATH, content)
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

        text = MEMORY_PATH.read_text(encoding="utf-8") if MEMORY_PATH.exists() else "# Agent-0 Memory\n"

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
            return f"Error: memory would grow past {MEMORY_MAX_BYTES} bytes"
        _secure_write(MEMORY_PATH, new_text)
        return f"Appended to memory under '{section}'."
    except Exception as e:
        return f"Error: {e}"


# ---------------- Session persistence ----------------

def save_session(history: list) -> None:
    """Persist session state.

    The system prompt is stripped before writing so a shared session.json
    cannot leak the agent's full instructions. It is regenerated on load.
    """
    try:
        trimmed = [m for m in history[-SESSION_HISTORY_CAP:]
                   if m.get("role") != "system"]
        data = {
            "saved_at": datetime.datetime.now().isoformat(),
            "provider": STATE.provider,
            "model": STATE.current_model(),
            "todos": _TODOS,
            "history": trimmed,
        }
        _secure_write(SESSION_PATH,
                      json.dumps(data, indent=2, default=str))
    except OSError as e:
        warn(f"could not save session: {e}")


def load_session() -> list | None:
    if not SESSION_PATH.exists():
        return None
    try:
        data = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        warn(f"could not load session ({e})")
        return None
    history = data.get("history")
    if not isinstance(history, list):
        return None
    # Always prepend a fresh system prompt; saved sessions never contain one.
    if history and isinstance(history[0], dict) and history[0].get("role") == "system":
        history[0] = {"role": "system", "content": build_system_prompt()}
    else:
        history.insert(0, {"role": "system", "content": build_system_prompt()})
    todos = data.get("todos")
    if isinstance(todos, list):
        _TODOS.clear()
        _TODOS.extend(t for t in todos if isinstance(t, dict))
    return history


def delete_session() -> bool:
    try:
        if SESSION_PATH.exists():
            SESSION_PATH.unlink()
            return True
    except OSError as e:
        warn(f"could not delete session: {e}")
    return False


# ---------------- Loop detection ----------------

class LoopDetector:
    def __init__(self, window: int = 4):
        self.window = window
        self.recent: list[str] = []

    def observe(self, fingerprint: str) -> bool:
        self.recent.append(fingerprint)
        if len(self.recent) > self.window:
            self.recent.pop(0)
        if len(self.recent) == self.window and len(set(self.recent)) == 1:
            return True
        return False

    def reset(self) -> None:
        self.recent.clear()


def _fingerprint(tool_calls: list["ToolCall"]) -> str:
    if not tool_calls:
        return "no-tools"
    return "|".join(
        f"{tc.name}({json.dumps(tc.arguments, sort_keys=True, default=str)})"
        for tc in tool_calls
    )


# ---------------- Budget ----------------

@dataclass
class Budget:
    max_steps: int = 50
    max_tool_calls: int = 200
    max_seconds: int = 900
    steps: int = 0
    tool_calls: int = 0
    started: float = field(default_factory=time.monotonic)

    def exhausted(self) -> bool:
        return (self.steps >= self.max_steps
                or self.tool_calls >= self.max_tool_calls
                or time.monotonic() - self.started > self.max_seconds)

    def status(self) -> str:
        return (f"steps {self.steps}/{self.max_steps}  "
                f"tools {self.tool_calls}/{self.max_tool_calls}  "
                f"{int(time.monotonic() - self.started)}s/{self.max_seconds}s")

    @classmethod
    def from_state(cls) -> "Budget":
        return cls(
            max_steps=STATE.budget_max_steps,
            max_tool_calls=STATE.budget_max_tools,
            max_seconds=STATE.budget_max_seconds,
        )


# ---------------- Schedule ----------------

@dataclass
class ScheduledTask:
    when: datetime.datetime
    prompt: str
    repeat_minutes: int | None = None


_SCHEDULE: list[ScheduledTask] = []


def _check_schedule(history: list) -> None:
    now = datetime.datetime.now()
    for task in list(_SCHEDULE):
        if task.when <= now:
            print(f"\n{BLUE}[scheduled]{RESET} {task.prompt}")
            try:
                run_agent(task.prompt, history)
            except KeyboardInterrupt:
                print(f"\n{CTRL_C_HINT}")
            if task.repeat_minutes:
                task.when = now + datetime.timedelta(minutes=task.repeat_minutes)
            else:
                _SCHEDULE.remove(task)


def _parse_duration(s: str) -> int | None:
    m = re.fullmatch(r"\s*(\d+)\s*([smhd])\s*", s.lower())
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    return n * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


# ---------------- Web search backends ----------------

def _http_get(url: str, headers: dict | None = None,
              timeout: int = SEARCH_TIMEOUT,
              max_bytes: int = SEARCH_MAX_BYTES) -> tuple[bytes, str, int]:
    hdrs = {"User-Agent": _USER_AGENT, "Accept": "*/*"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(max_bytes)
        charset = resp.headers.get_content_charset() or "utf-8"
        return raw, charset, resp.status


def _http_post_json(url: str, payload: dict, headers: dict | None = None,
                    timeout: int = SEARCH_TIMEOUT) -> tuple[bytes, int]:
    data = json.dumps(payload).encode("utf-8")
    hdrs = {"User-Agent": _USER_AGENT,
            "Content-Type": "application/json",
            "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(SEARCH_MAX_BYTES), resp.status


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)


def _clean_text(s: str) -> str:
    return " ".join(_html.unescape(_strip_tags(s)).split())


_DDG_REDIRECT_PREFIXES = (
    "//duckduckgo.com/l/",
    "https://duckduckgo.com/l/",
    "http://duckduckgo.com/l/",
)


def _ddg_unwrap(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    for pref in _DDG_REDIRECT_PREFIXES:
        if href.startswith(pref):
            try:
                qs = urllib.parse.urlparse(href).query
                params = urllib.parse.parse_qs(qs)
                real = params.get("uddg", [""])[0]
                if real:
                    return urllib.parse.unquote(real)
            except Exception:
                pass
    return href


_DDG_RESULT_BLOCK = re.compile(
    r'<div[^>]*class="[^"]*\bresult\b[^"]*"[^>]*>(.*?)(?=<div[^>]*class="[^"]*\bresult\b|$)',
    re.DOTALL | re.IGNORECASE,
)
_DDG_LINK = re.compile(
    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_DDG_SNIPPET = re.compile(
    r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)


def _parse_ddg_html(html_text: str, limit: int) -> list[dict]:
    results: list[dict] = []
    for block_match in _DDG_RESULT_BLOCK.finditer(html_text):
        block = block_match.group(1)
        link_match = _DDG_LINK.search(block)
        if not link_match:
            continue
        href = _ddg_unwrap(link_match.group(1).strip())
        title = _clean_text(link_match.group(2))
        snippet = ""
        sm = _DDG_SNIPPET.search(block)
        if sm:
            snippet = _clean_text(sm.group(1))
        if href and title:
            results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


_DDG_LITE_LINK = re.compile(
    r'<a[^>]*class="[^"]*result-link[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_DDG_LITE_SNIPPET = re.compile(
    r'<td[^>]*class="[^"]*result-snippet[^"]*"[^>]*>(.*?)</td>',
    re.DOTALL | re.IGNORECASE,
)


def _parse_ddg_lite(html_text: str, limit: int) -> list[dict]:
    results: list[dict] = []
    links = list(_DDG_LITE_LINK.finditer(html_text))
    snippets = list(_DDG_LITE_SNIPPET.finditer(html_text))
    for i, lm in enumerate(links[:limit]):
        href = _ddg_unwrap(lm.group(1).strip())
        title = _clean_text(lm.group(2))
        snippet = ""
        if i < len(snippets):
            snippet = _clean_text(snippets[i].group(1))
        if href and title:
            results.append({"title": title, "url": href, "snippet": snippet})
    return results


def _search_duckduckgo(query: str, limit: int) -> tuple[str, str]:
    q = urllib.parse.quote_plus(query)
    endpoints = [
        (f"https://html.duckduckgo.com/html/?q={q}", _parse_ddg_html),
        (f"https://lite.duckduckgo.com/lite/?q={q}", _parse_ddg_lite),
    ]
    last_err = "no attempt made"
    for url, parser in endpoints:
        try:
            raw, charset, status = _http_get(url)
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code} {e.reason}"
            continue
        except urllib.error.URLError as e:
            last_err = f"network error: {e.reason}"
            continue
        except Exception as e:
            last_err = str(e)
            continue
        try:
            body = raw.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            body = raw.decode("utf-8", errors="replace")
        results = parser(body, limit)
        if results:
            return _format_search_results(query, "DuckDuckGo", results), ""
        last_err = "no results parsed (layout may have changed or rate limit hit)"
    return "", (f"Error: DuckDuckGo search failed ({last_err}). "
                f"Try again in a moment, or switch backends: "
                f"`/search backend brave` (needs key) or `/search backend tavily`.")


def _search_brave(query: str, limit: int, api_key: str) -> tuple[str, str]:
    q = urllib.parse.quote_plus(query)
    url = f"https://api.search.brave.com/res/v1/web/search?q={q}&count={limit}"
    try:
        raw, charset, _ = _http_get(
            url, headers={"Accept": "application/json",
                          "X-Subscription-Token": api_key},
            timeout=SEARCH_TIMEOUT, max_bytes=SEARCH_MAX_BYTES,
        )
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return "", "Error: Brave API key rejected (401). Check `/search -s`."
        if e.code == 429:
            return "", "Error: Brave rate limit hit (429). Wait or switch backend."
        return "", f"Error: Brave HTTP {e.code} {e.reason}"
    except Exception as e:
        return "", f"Error: Brave request failed: {e}"

    try:
        data = json.loads(raw.decode(charset or "utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        return "", f"Error: could not parse Brave response: {e}"

    web = ((data.get("web") or {}).get("results") or [])
    results: list[dict] = []
    for r in web[:limit]:
        title = r.get("title", "")
        u = r.get("url", "")
        desc = r.get("description", "")
        if title and u:
            results.append({"title": _clean_text(title), "url": u,
                            "snippet": _clean_text(desc)})
    if not results:
        return "", "Error: Brave returned no results"
    return _format_search_results(query, "Brave", results), ""


def _search_tavily(query: str, limit: int, api_key: str) -> tuple[str, str]:
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": limit,
        "include_answer": False,
        "search_depth": "basic",
    }
    try:
        raw, _ = _http_post_json("https://api.tavily.com/search", payload)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return "", "Error: Tavily API key rejected (401). Check `/search -s`."
        if e.code == 429:
            return "", "Error: Tavily rate limit hit (429). Wait or switch backend."
        return "", f"Error: Tavily HTTP {e.code} {e.reason}"
    except Exception as e:
        return "", f"Error: Tavily request failed: {e}"

    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        return "", f"Error: could not parse Tavily response: {e}"

    results: list[dict] = []
    for r in (data.get("results") or [])[:limit]:
        title = r.get("title", "")
        u = r.get("url", "")
        content = r.get("content", "")
        if title and u:
            results.append({"title": _clean_text(title), "url": u,
                            "snippet": _clean_text(content)[:400]})
    if not results:
        return "", "Error: Tavily returned no results"
    return _format_search_results(query, "Tavily", results), ""


def _format_search_results(query: str, backend: str, results: list[dict]) -> str:
    lines = [f"Web search ({backend}) for: {query}", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   {r['url']}")
        if r.get("snippet"):
            lines.append(f"   {r['snippet']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def tool_web_search(query: str, num_results: Any = None) -> str:
    if not isinstance(query, str) or not query.strip():
        return "Error: 'query' must be a non-empty string"
    n = _as_int(num_results, STATE.search_max_results)
    n = max(1, min(20, n))
    q = query.strip()

    backend = STATE.search_backend
    if backend == "duckduckgo":
        text, err = _search_duckduckgo(q, n)
    elif backend == "brave":
        if not STATE.brave_api_key:
            return ("Error: Brave API key not set. "
                    "Use `/search key brave <key>` or switch to DuckDuckGo with "
                    "`/search backend duckduckgo`.")
        text, err = _search_brave(q, n, STATE.brave_api_key)
    elif backend == "tavily":
        if not STATE.tavily_api_key:
            return ("Error: Tavily API key not set. "
                    "Use `/search key tavily <key>` or switch to DuckDuckGo with "
                    "`/search backend duckduckgo`.")
        text, err = _search_tavily(q, n, STATE.tavily_api_key)
    else:
        return f"Error: unknown search backend '{backend}'"

    if err:
        return err
    return _truncate(text)


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
    _fn("shell",
        f"Run a shell command. Returns combined stdout+stderr and exit code. "
        f"Times out after {SHELL_TIMEOUT} seconds.",
        {"command": {"type": "string", "description": "The shell command."}},
        ["command"]),
    _fn("read_file", "Return the text contents of a file.",
        {"path": {"type": "string"}}, ["path"]),
    _fn("write_file",
        "Write content to a path, overwriting it. Creates parent dirs. "
        "Use for NEW files or full rewrites; prefer edit_file for small edits.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"]),
    _fn("edit_file",
        "Replace old_string with new_string in a file. old_string must appear "
        "EXACTLY ONCE unless replace_all is true.",
        {"path": {"type": "string"},
         "old_string": {"type": "string"},
         "new_string": {"type": "string"},
         "replace_all": {"type": "boolean"}},
        ["path", "old_string", "new_string"]),
    _fn("list_dir",
        "List a directory's contents as a tree. Ignores .git, node_modules, etc.",
        {"path": {"type": "string"}, "recursive": {"type": "boolean"}}, []),
    _fn("glob", "Find files by glob pattern, e.g. '**/*.py'.",
        {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
    _fn("grep", "Regex search inside files.",
        {"pattern": {"type": "string"}, "path": {"type": "string"},
         "include": {"type": "string"}, "case_insensitive": {"type": "boolean"}},
        ["pattern"]),
    _fn("fetch_url",
        "HTTP GET a URL and return the text body (up to ~500 KB).",
        {"url": {"type": "string"}}, ["url"]),
    _fn("web_search",
        "Search the web and return a list of results (title, URL, snippet). "
        "Use this to find current information, then `fetch_url` specific "
        "results for full content. Backend is configurable via /search.",
        {"query": {"type": "string", "description": "The search query."},
         "num_results": {"type": "integer",
                         "description": "How many results (1-20, default 8)."}},
        ["query"]),
    _fn("todo_write",
        "Replace the session todo list. Each item: "
        '{"content": str, "status": "pending"|"in_progress"|"completed", '
        '"activeForm": optional str}. At most one may be in_progress.',
        {"todos": {"type": "array", "items": {"type": "object"}}}, ["todos"]),
    _fn("todo_read", "Return the current session todo list.", {}, []),
    _fn("memory_read", "Return the persistent memory file contents.", {}, []),
    _fn("memory_append",
        "Append one bullet to a section of the persistent memory file.",
        {"section": {"type": "string"}, "entry": {"type": "string"}},
        ["section", "entry"]),
    _fn("memory_write", "Replace the ENTIRE memory file.",
        {"content": {"type": "string"}}, ["content"]),
    _fn("spawn_subagent",
        "Delegate an independent subgoal to a nested agent. Returns its prose summary. "
        "Use ONLY for work that can be done without the current conversation's context.",
        {"goal": {"type": "string"}, "max_steps": {"type": "integer"}},
        ["goal"]),
]

TOOL_DESCRIPTIONS = {
    "shell": "run a shell command",
    "read_file": "return a file's text contents",
    "write_file": "write content to a path (overwrites)",
    "edit_file": "replace old_string with new_string in a file",
    "list_dir": "list a directory's contents (tree)",
    "glob": "find files by glob pattern",
    "grep": "regex search inside files",
    "fetch_url": "HTTP GET a URL",
    "web_search": "search the web for a query",
    "todo_write": "replace the session todo list",
    "todo_read": "return the session todo list",
    "memory_read": "return the persistent memory file",
    "memory_append": "append one bullet to a memory section",
    "memory_write": "replace the entire memory file",
    "spawn_subagent": "delegate an independent subgoal to a nested agent",
}


# ---------------- Prompts ----------------

SYSTEM_PROMPT = """You are Agent-0, a highly autonomous command-line coding and system agent running inside an interactive terminal on the user's machine. You work through multi-step tasks on your own, use tools aggressively, search the web when you need current information, and remember durable facts across sessions.

# Environment
{environment}

# Autonomy
Your current autonomy level is `{autonomy}`.
- manual: every tool call is confirmed by the user.
- smart:  read-only tools run automatically; edits, web, and commands are confirmed.
- auto:   read-only AND file/web tools run automatically; shell is confirmed.
- yolo:   everything runs automatically (user opted in).
Act as if your actions will be approved. Do NOT ask for permission in prose —
the harness handles that. Do NOT stop mid-task to check in; keep going until
the task is done or you genuinely need user input.

# Memory
You have a persistent memory file at {memory_path}. It survives across sessions
and is injected below.
- Read it with `memory_read` when the user references prior work.
- Save DURABLE facts: preferences, project layouts, decisions, gotchas.
- Prefer `memory_append` for one fact; `memory_write` only to reorganize.
- NEVER store: secrets, API keys, credentials, transient output.
- Save proactively — do not wait to be asked.

### Current memory
{memory}

# Your own files — do not read or write these
Agent-0 stores its own state under `db/local/`. These are agent internals,
NOT part of the user's project:
- `db/local/session.json`  — conversation history (system prompt is stripped)
- `db/local/.config.json`  — API keys, preferences
- `db/local/goals.db`, `db/local/audit.jsonl` — if present
The file tools refuse to touch them. Do NOT attempt `read_file` / `write_file`
on them, and do not use `shell` (e.g. `cat`, `grep`) to read them either — it
is a policy violation even if the tool would technically work. If you need
your own memory, use `memory_read` / `memory_append` / `memory_write`.

# How you reply
You have native tools. Each turn, pick exactly ONE of:
- CALL one or more tools, or
- REPLY in prose (when the answer is ready or no tool is needed).
Never do both in one turn. Never describe a tool call in prose — just call it.

# Tone
- Concise. No preamble ("Sure!", "Of course!"). No postamble ("Let me know if…").
- No narration ("Let me…", "I'll now…").
- Answer in prose unless the user asked for code.
- If the user asks you to do something, DO it.
- Never speculate about code you have not read. If a file was truncated, say so.

# Your tools — exactly these, nothing else
shell, read_file, write_file, edit_file, list_dir, glob, grep, fetch_url,
web_search, todo_write, todo_read, memory_read, memory_append, memory_write,
spawn_subagent.

When asked what you can do, describe ONLY those tools. Do NOT invent tools.

# When to search the web
- User asks about current events, recent releases, news, prices, or anything
  time-sensitive → search.
- User references a specific library, API, or tool whose interface may have
  changed → search.
- You are about to write code that depends on a version-specific detail you
  don't have in memory → search.
- The user says "find", "look up", "search for", "google", "what is X" about
  something external to the machine → search.
- DO NOT search for: file operations, local code questions, or things you can
  answer from the repo, memory, or general knowledge that does not change.

Search workflow: use `web_search` first to get titles and URLs, then
`fetch_url` the most promising results to get full content. Do NOT paste the
raw search output back to the user — synthesize.

# Tool selection
- Find files by name → glob.
- Find code/text by content → grep.
- Understand a directory's shape → list_dir.
- Read a specific file → read_file (before editing).
- Small targeted edits → edit_file (make old_string unique).
- Full rewrite or new file → write_file.
- Run something → shell.
- Fetch a known URL → fetch_url.
- Search the web → web_search.
- Persist a fact → memory_append.
- Delegate independent work → spawn_subagent.

# Batching
- Multiple INDEPENDENT calls in ONE turn (e.g. read two files, search + read).
- Calls that DEPEND on each other go in separate turns.
- Never call the same tool with the same arguments twice in a row.

# Task management
For tasks with 3+ distinct steps, use todo_write to plan before starting.
- Keep exactly ONE task in_progress at a time.
- Mark tasks completed the moment they are done.

# Doing tasks
1. UNDERSTAND. Search before editing. Read files before editing them. If the
   task depends on external information, web_search first.
2. PLAN. Write todos for multi-step work.
3. IMPLEMENT. Prefer edit_file with enough context that old_string is unique.
4. VERIFY. If tests exist, run them. Read your edits back.
5. REMEMBER. Save durable facts to memory. Clean up the todo list.

# Safety — never do these without explicit user instruction
git commit, git push, git reset --hard, git clean -fd, rm -rf, chmod -R,
force-pushing, package installs, or anything that sends data off-machine.
Autonomy does not mean recklessness.
"""

FAST_THINK_NOTE = "\n\n# Fast mode\nKeep reasoning extremely brief."

SUMMARY_SYSTEM_PROMPT = """You are a safety reviewer for a command-line AI agent.
You will receive ONE tool call as JSON. In ONE short sentence, say exactly what it will do on the user's machine.

- Be concrete: name the command, file paths, side effects, or search query.
- Flag plainly if destructive, irreversible, or touches the network.
- Output ONLY the sentence. No quotes, no preamble."""

CRITIC_PROMPT = """You are evaluating whether a CLI agent achieved its goal.
Read the goal and the agent's recent work. Reply with EXACTLY one word on the
first line:

  DONE      - the goal is fully achieved
  CONTINUE  - real progress, but more work is needed
  STUCK     - the agent is looping, blocked, or repeating itself

On the next line, one sentence of justification. Nothing else."""

REFLECT_PROMPT = """You appear to be stuck or looping. Before trying again:
1. One sentence: what were you trying to do?
2. One sentence: why did the last attempt fail?
3. One sentence: a DIFFERENT approach you have not tried yet.
Then continue working toward the goal. Do not repeat the same tool call."""


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
    ".git", ".hg", ".svn", "__pycache__", "node_modules",
    ".venv", "venv", "env", ".idea", ".vscode", ".vs",
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
    if any(part in IGNORE_DIRS for part in p.parts):
        return True
    # Hide agent-internal state from directory/file listings.
    try:
        if p.name in _PROTECTED_NAMES:
            return True
    except Exception:
        pass
    return False


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def tool_shell(command: str) -> str:
    if not isinstance(command, str):
        return f"Error: 'command' must be a string, got {type(command).__name__}"
    posix = os.name == "posix"
    try:
        proc = subprocess.Popen(
            command, shell=True, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors="replace", start_new_session=posix,
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
        if _is_protected(p):
            return (f"Error: {p} is an Agent-0 internal state file "
                    f"and cannot be read by tools.")
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
        if _is_protected(p):
            return (f"Error: {p} is an Agent-0 internal state file "
                    f"and cannot be written by tools.")
        p.parent.mkdir(parents=True, exist_ok=True)
        text = content if isinstance(content, str) else json.dumps(content, indent=2)
        p.write_text(text, encoding="utf-8")
        return f"Wrote {len(text.encode('utf-8'))} bytes to {p}"
    except Exception as e:
        return f"Error: {e}"


def tool_edit_file(path: str, old_string: str, new_string: str = "",
                   replace_all: Any = False) -> str:
    try:
        p = Path(path).expanduser()
        if _is_protected(p):
            return (f"Error: {p} is an Agent-0 internal state file "
                    f"and cannot be edited by tools.")
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
            return ("Error: 'old_string' not found. Read the file first "
                    "and copy the exact text (including whitespace).")
        if count > 1 and not replace_all:
            return (f"Error: 'old_string' occurs {count} times. "
                    f"Add context or pass replace_all=true.")
        updated = original.replace(old_string, new_string, -1 if replace_all else 1)
        p.write_text(updated, encoding="utf-8")
        return f"Edited {p}: replaced {count if replace_all else 1} occurrence(s)."
    except Exception as e:
        return f"Error: {e}"


def _format_tree(root: Path, max_entries: int = 400) -> str:
    lines: list[str] = []
    count = 0

    def walk(d: Path, prefix: str) -> None:
        nonlocal count
        try:
            entries = sorted(
                (e for e in d.iterdir() if not _should_ignore(e)),
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
                try:
                    rel = p.relative_to(root)
                except ValueError:
                    rel = p
                if _should_ignore(rel if isinstance(rel, Path) else Path(str(rel))):
                    continue
                if _is_protected(p):
                    continue
                count += 1
                marker = "/" if p.is_dir() else ""
                out.append(f"{rel}{marker}")
            return _truncate(header + ("\n".join(out) if out else "(empty)"))
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
            if _is_protected(p):
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


def tool_grep(pattern: str, path: str = ".", include: str = "",
              case_insensitive: Any = False) -> str:
    try:
        root = Path(path).expanduser()
        try:
            rx = re.compile(pattern, re.IGNORECASE if _as_bool(case_insensitive, False) else 0)
        except re.error as e:
            return f"Error: invalid regex: {e}"
        if root.is_file():
            if _is_protected(root):
                return (f"Error: {root} is an Agent-0 internal state file "
                        f"and cannot be searched by tools.")
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
        for f in files:
            try:
                rel = f.relative_to(base)
            except ValueError:
                rel = f
            if _should_ignore(rel if isinstance(rel, Path) else Path(str(rel))):
                continue
            if _is_protected(f):
                continue
            if include and not fnmatch.fnmatch(f.name, include) and \
               not fnmatch.fnmatch(str(rel), include):
                continue
            files_scanned += 1
            if files_scanned > 2000:
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
            return f"Error: only http/https, got '{parsed.scheme}'"
        req = urllib.request.Request(url,
                                     headers={"User-Agent": f"Agent-0/{__version__}"},
                                     method="GET")
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
    if sum(1 for t in normalized if t["status"] == "in_progress") > 1:
        return "Error: at most ONE todo may be 'in_progress'"
    _TODOS = normalized
    return "Todo list updated:\n" + _format_todos()


def tool_todo_read() -> str:
    return _format_todos()


# ---------------- Subagent ----------------

_SUBAGENT_DEPTH = 0


def spawn_subagent(goal: str, max_steps: Any = 10) -> str:
    global _SUBAGENT_DEPTH
    if _SUBAGENT_DEPTH >= MAX_SUBAGENT_DEPTH:
        return (f"Error: subagent recursion depth {_SUBAGENT_DEPTH} >= "
                f"{MAX_SUBAGENT_DEPTH}; refusing to nest further")
    if not isinstance(goal, str) or not goal.strip():
        return "Error: 'goal' must be a non-empty string"
    steps = max(1, min(20, _as_int(max_steps, 10)))

    _SUBAGENT_DEPTH += 1
    try:
        sub_history: list = [{"role": "system", "content": build_system_prompt()}]
        sub_budget = Budget(max_steps=steps, max_tool_calls=steps * 6, max_seconds=300)
        print(f"{MAGENTA}[subagent:{_SUBAGENT_DEPTH}] {goal}{RESET}")
        ok = run_goal(goal, sub_history, sub_budget, quiet=True)
        for m in reversed(sub_history):
            if m.get("role") == "assistant" and m.get("content"):
                summary = str(m["content"])[:2000]
                return f"Subagent {'completed' if ok else 'did not complete'}:\n{summary}"
        return f"Subagent produced no output (completed={ok})."
    finally:
        _SUBAGENT_DEPTH -= 1


TOOLS = {
    "shell": tool_shell,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
    "list_dir": tool_list_dir,
    "glob": tool_glob,
    "grep": tool_grep,
    "fetch_url": tool_fetch_url,
    "web_search": tool_web_search,
    "todo_write": tool_todo_write,
    "todo_read": tool_todo_read,
    "memory_read": tool_memory_read,
    "memory_append": tool_memory_append,
    "memory_write": tool_memory_write,
    "spawn_subagent": spawn_subagent,
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
        return (f"write_file: {args.get('path', '')} "
                f"({len(str(args.get('content', '')).encode())} bytes)")
    if name == "edit_file":
        return (f"edit_file: {args.get('path', '')} "
                f"({len(str(args.get('old_string', '')))}→"
                f"{len(str(args.get('new_string', '')))} chars"
                f"{', replace_all' if _as_bool(args.get('replace_all'), False) else ''})")
    if name == "list_dir":
        return f"list_dir: {args.get('path') or '.'}  recursive={_as_bool(args.get('recursive'), False)}"
    if name == "glob":
        return f"glob: {args.get('pattern', '')}  in {args.get('path') or '.'}"
    if name == "grep":
        p = args.get("path") or "."
        inc = args.get("include") or ""
        return f"grep: /{args.get('pattern', '')}/  in {p}" + (f"  include={inc}" if inc else "")
    if name == "fetch_url":
        return f"fetch_url: {args.get('url', '')}"
    if name == "web_search":
        q = str(args.get("query", ""))
        n = args.get("num_results")
        suffix = f"  (n={n})" if n is not None else ""
        return f"web_search: {q}{suffix}"
    if name == "todo_write":
        todos = args.get("todos")
        return f"todo_write: {len(todos) if isinstance(todos, list) else '?'} item(s)"
    if name in ("todo_read", "memory_read"):
        return name
    if name == "memory_append":
        entry = str(args.get("entry", ""))
        short = entry if len(entry) <= 80 else entry[:77] + "..."
        return f"memory_append: [{args.get('section', '')}] {short}"
    if name == "memory_write":
        return f"memory_write: {len(str(args.get('content', '')))} chars"
    if name == "spawn_subagent":
        goal = str(args.get("goal", ""))
        short = goal if len(goal) <= 100 else goal[:97] + "..."
        return f"spawn_subagent: {short}"
    return f"{name}: {json.dumps(args, default=str)}"


# ---------------- ToolCall ----------------

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
        return ToolCall(id=call_id, name=name, arguments={},
                        raw_arguments=raw_arguments, parse_error=str(e))


# ---------------- Validation ----------------

_ALLOWED_ARG_TYPES = (str, bool, int, float, list, dict)


def _normalize_args(name: str, args: dict) -> dict:
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
            return f"Error: argument '{k}' for {name} has unsupported type {type(v).__name__}"
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
    if any(w in low for w in ("connection", "refused", "unreachable", "connect")):
        return (f"{msg}\n  hint: cannot reach the backend. Ollama: is `ollama serve` running? "
                f"Groq: check network / `/api -s`.")
    if "timed out" in low or "timeout" in low:
        return f"{msg}\n  hint: request timed out."
    if "401" in msg or "unauthorized" in low or "invalid api key" in low:
        return f"{msg}\n  hint: auth failed. Set key with `/api <key>` or `/search key …`."
    if "403" in msg or "forbidden" in low:
        return f"{msg}\n  hint: access forbidden."
    if "429" in msg or "rate limit" in low or "too many requests" in low:
        return f"{msg}\n  hint: rate limited. Wait, or switch model/provider."
    if "model" in low and any(w in low for w in ("not found", "does not exist", "unknown model")):
        return f"{msg}\n  hint: model not available. `/model` to list, `/model <name>` to switch."
    if "context" in low and ("length" in low or "too long" in low or "maximum" in low):
        return f"{msg}\n  hint: context full. `/reset` to clear history."
    if "tool" in low and any(w in low for w in ("not supported", "unsupported", "does not support")):
        return (f"{msg}\n  hint: model doesn't support native tool calling. "
                f"Try `/model llama-3.3-70b-versatile` (groq) or `qwen2.5`/`llama3.1` (ollama).")
    if "think" in low:
        return f"{msg}\n  hint: `think` unsupported. `/think off`."
    return msg


# ---------------- Ollama backend ----------------

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
        raise RuntimeError("The 'ollama' package is not installed. Run: pip install ollama")


def _ollama_kwargs(messages, stream: bool, think, tools):
    kw: dict = dict(model=STATE.model, messages=messages, stream=stream,
                    options={"temperature": TEMPERATURE})
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
            return ollama.chat(**_ollama_kwargs(messages, False, think, tools))
        except Exception as e:
            if _is_think_error(e):
                last_err = e
                continue
            raise
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
            out.append({"role": "assistant", "content": m.get("content") or "",
                        "tool_calls": tcs})
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
        for i, raw_tc in enumerate(_get(msg, "tool_calls") or []):
            fn = _get(raw_tc, "function") or {}
            name = _get(fn, "name") or ""
            args = _get(fn, "arguments")
            if isinstance(args, str):
                raw = args
            elif isinstance(args, dict):
                raw = json.dumps(args)
            else:
                raw = "{}"
            yield ("tool_call", _make_tool_call(_get(raw_tc, "id") or f"call_{i}", name, raw))


# ---------------- Groq backend ----------------

class _GroqStreamError(RuntimeError):
    pass


def _require_groq() -> None:
    if not _GROQ_AVAILABLE:
        raise RuntimeError("The 'groq' package is not installed. Run: pip install groq")
    if not STATE.groq_api_key:
        raise RuntimeError("No Groq API key. Use /api <key> or $AGENT0_GROQ_API_KEY.")


def _groq_error_message(e: Exception) -> str:
    low = str(e).lower()
    if "tool_choice" in low and "none" in low:
        return "Groq rejected a tool call (tool_choice mismatch). Retry or switch model."
    return str(e)


def _groq_create(messages, *, stream: bool, tools=None):
    client = _GroqClient(api_key=STATE.groq_api_key)
    kw: dict = dict(model=STATE.groq_model, messages=messages,
                    temperature=TEMPERATURE, stream=stream)
    if tools:
        kw["tools"] = tools
        kw["tool_choice"] = "auto"
    return client.chat.completions.create(**kw)


def _to_groq_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m.get("role") == "tool":
            out.append({"role": "tool",
                        "tool_call_id": m.get("tool_call_id", ""),
                        "content": m.get("content", "")})
        else:
            out.append(m)
    return out


def _iter_groq(messages: list[dict]):
    _require_groq()
    try:
        stream = _groq_create(_to_groq_messages(messages), stream=True, tools=TOOL_SCHEMAS)
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
                idx = getattr(tc_delta, "index", 0) or 0
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
        yield ("tool_call", _make_tool_call(entry["id"] or f"call_{idx}",
                                            entry["name"] or "", raw))


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
    return "".join(thinking_parts), "".join(content_parts).strip(), tool_calls, printed


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


# ---------------- Goal evaluation ----------------

def _evaluate_goal(goal: str, history: list) -> tuple[str, str]:
    recent = history[-12:] if len(history) > 12 else history[1:]
    parts = []
    for m in recent:
        role = m.get("role", "?")
        content = str(m.get("content", ""))[:1500]
        tcs = m.get("tool_calls")
        if tcs:
            names = ", ".join((t.get("function") or {}).get("name", "?") for t in tcs)
            parts.append(f"{role}: [called: {names}]\n{content}")
        else:
            parts.append(f"{role}: {content}")
    text = "\n\n".join(parts)
    messages = [
        {"role": "system", "content": CRITIC_PROMPT},
        {"role": "user", "content": f"GOAL: {goal}\n\nRECENT WORK:\n{text}"},
    ]
    try:
        resp = _chat_once(messages)
        raw = (_get(_get(resp, "message") or {}, "content") or "").strip()
        raw = _strip_think(raw).strip()
        upper = raw.upper()[:300]
        if "DONE" in upper:
            verdict = "DONE"
        elif "STUCK" in upper:
            verdict = "STUCK"
        else:
            verdict = "CONTINUE"
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        reason = lines[1].strip() if len(lines) > 1 else ""
        return verdict, reason
    except Exception as e:
        return "CONTINUE", f"(critic failed: {e})"


# ---------------- Confirmation ----------------

def confirm_tool_call(call: dict, *, label: str = "") -> bool:
    print(f"{YELLOW}{label}{format_tool_call(call)}{RESET}")
    summary = summarize_tool_call(call)
    print(f"{CYAN}{label}{summary}{RESET}")
    while True:
        try:
            ans = input(f"{BOLD}{label}[Y]es/[N]o? {RESET}").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{CTRL_C_HINT}")
            continue
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


# ---------------- Agent loop primitives ----------------

def _assistant_message(content: str, tool_calls: list[ToolCall]) -> dict:
    msg: dict = {"role": "assistant", "content": content or ""}
    if tool_calls:
        msg["tool_calls"] = [tc.to_openai() for tc in tool_calls]
    return msg


def _tool_message(tc: ToolCall, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tc.id, "tool_name": tc.name, "content": content}


def _handle_one_call(tc: ToolCall, label: str, *, auto_override: bool = False) -> str:
    if tc.parse_error:
        msg = f"Error: could not parse arguments for '{tc.name}': {tc.parse_error}."
        print(f"{RED}{label}{msg}{RESET}")
        return msg

    block = guard_tool_call(tc)
    if block:
        print(f"{RED}{label}{block}{RESET}")
        return f"Error: {block}. Try a different approach."

    args = _normalize_args(tc.name, tc.arguments) if tc.name in TOOLS else tc.arguments
    call = {"name": tc.name, "args": args}

    err = validate_call(call)
    if err:
        print(f"{RED}{label}{err}{RESET}")
        return err

    needs_confirm = not auto_override and not _autonomy_allows(tc.name)
    if needs_confirm:
        if not confirm_tool_call(call, label=label):
            print(f"{GRAY}{label}Denied.{RESET}")
            return "User DENIED this tool call. It was NOT executed."
    else:
        policy = "auto" if auto_override else f"auto:{_tool_policy(tc.name)}"
        print(f"{MAGENTA}{label}[{policy}] {format_tool_call(call)}{RESET}")

    try:
        result = TOOLS[tc.name](**args)
    except Exception as e:
        result = f"Error running {tc.name}: {e}"

    preview = result if len(result) <= TOOL_PREVIEW_LIMIT else result[:TOOL_PREVIEW_LIMIT] + " ..."
    print(f"{GRAY}{label}{preview}{RESET}")
    return result


# ---------------- Regular agent run ----------------

def run_agent(user_input: str, history: list, *, quiet: bool = False) -> None:
    history[0] = {"role": "system", "content": build_system_prompt()}
    history.append({"role": "user", "content": user_input})

    loop = LoopDetector()

    for _ in range(1, MAX_STEPS + 1):
        try:
            _, content, tool_calls, printed = stream_chat(history)
        except KeyboardInterrupt:
            print(f"\n{CTRL_C_HINT}")
            break
        except Exception as e:
            print(f"{RED}[error] {_hint_for_error(e)}{RESET}")
            break

        if not content and not tool_calls:
            if not printed:
                print(f"{GRAY}[empty reply — stopping]{RESET}")
            break

        history.append(_assistant_message(content, tool_calls))

        if not tool_calls:
            if not printed and content:
                print(content)
            break

        if loop.observe(_fingerprint(tool_calls)):
            print(f"{YELLOW}[loop detected]{RESET} injecting reflection")
            history.append({"role": "user", "content": REFLECT_PROMPT})
            loop.reset()
            continue

        batch = len(tool_calls)
        for idx, tc in enumerate(tool_calls, 1):
            label = f"[{idx}/{batch}] " if batch > 1 else ""
            result = _handle_one_call(tc, label)
            history.append(_tool_message(tc, result))
    else:
        print(f"{RED}Hit max steps ({MAX_STEPS}); stopping this turn.{RESET}")

    if not quiet:
        save_session(history)


# ---------------- Goal-driven autonomous run ----------------

def run_goal(goal: str, history: list, budget: Budget, *, quiet: bool = False) -> bool:
    if not quiet:
        print(f"{BOLD}{CYAN}goal:{RESET} {goal}")
        print(f"{GRAY}budget: {budget.status()}{RESET}")

    history[0] = {"role": "system", "content": build_system_prompt()}
    history.append({
        "role": "user",
        "content": (
            f"GOAL: {goal}\n\n"
            "Plan first with todo_write if there are 3+ steps, then execute. "
            "Work autonomously — do not ask for permission or clarification unless "
            "you are truly blocked. Search the web if you need current information. "
            "When you believe the goal is complete, reply with the final answer in prose."
        ),
    })

    loop = LoopDetector()
    stuck_count = 0
    consecutive_continue = 0

    while not budget.exhausted():
        budget.steps += 1
        try:
            _, content, tool_calls, printed = stream_chat(history)
        except KeyboardInterrupt:
            print(f"\n{CTRL_C_HINT}")
            return False
        except Exception as e:
            print(f"{RED}[error] {_hint_for_error(e)}{RESET}")
            return False

        if not content and not tool_calls:
            if not printed:
                print(f"{GRAY}[empty reply — stopping]{RESET}")
            return False

        history.append(_assistant_message(content, tool_calls))

        if tool_calls:
            budget.tool_calls += len(tool_calls)
            if loop.observe(_fingerprint(tool_calls)):
                print(f"{YELLOW}[loop detected]{RESET} injecting reflection")
                history.append({"role": "user", "content": REFLECT_PROMPT})
                loop.reset()
                continue
            batch = len(tool_calls)
            for idx, tc in enumerate(tool_calls, 1):
                label = f"[{idx}/{batch}] " if batch > 1 else ""
                result = _handle_one_call(tc, label)
                history.append(_tool_message(tc, result))
            continue

        verdict, reason = _evaluate_goal(goal, history)
        print(f"{GRAY}critic: {verdict}"
              + (f" — {reason}" if reason else "")
              + f"  ({budget.status()}){RESET}")

        if verdict == "DONE":
            if not printed and content:
                print(content)
            print(f"{GREEN}goal complete.{RESET}")
            if not quiet:
                save_session(history)
            return True

        if verdict == "STUCK":
            stuck_count += 1
            if stuck_count >= 3:
                print(f"{RED}stuck 3 times; pausing goal.{RESET}")
                if not quiet:
                    save_session(history)
                return False
            history.append({"role": "user", "content": REFLECT_PROMPT})
        else:
            consecutive_continue += 1
            if consecutive_continue >= 5:
                print(f"{RED}5 continues without progress; pausing goal.{RESET}")
                if not quiet:
                    save_session(history)
                return False
            history.append({
                "role": "user",
                "content": "Continue toward the goal. What is the NEXT concrete step?",
            })

    print(f"{RED}budget exhausted ({budget.status()}); pausing goal.{RESET}")
    if not quiet:
        save_session(history)
    return False


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
    if STATE.provider == "groq":
        avail = _list_groq_models()
        if avail and arg not in avail:
            print(f"{YELLOW}note:{RESET} '{arg}' not in Groq list. "
                  f"{GRAY}available: {', '.join(avail[:12])}{RESET}")
        STATE.set_groq_model(arg)
        print(f"{GREEN}groq model set to {CYAN}{arg}{RESET}")
    else:
        avail = _list_local_models()
        if avail and arg not in avail and f"{arg}:latest" not in avail:
            print(f"{YELLOW}note:{RESET} '{arg}' not in local Ollama list. "
                  f"{GRAY}available: {', '.join(avail)}{RESET}")
        STATE.set_model(arg)
        print(f"{GREEN}ollama model set to {CYAN}{arg}{RESET}")


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
            print(f"{RED}Run: pip install groq{RESET}"); return
        if not STATE.groq_api_key:
            print(f"{RED}No Groq API key. Use /api <key>.{RESET}"); return
    if arg == "ollama" and not _OLLAMA_AVAILABLE:
        print(f"{RED}Run: pip install ollama{RESET}"); return
    STATE.set_provider(arg)
    print(f"{GREEN}provider: {CYAN}{arg}{RESET} "
          f"{GRAY}(model: {STATE.current_model()}){RESET}")


AUTO_USAGE = f"{GRAY}usage: /auto manual|smart|auto|yolo  (no arg shows current){RESET}"
_AUTO_DESCRIPTIONS = {
    "manual": "confirm every tool call",
    "smart":  "auto read-only tools",
    "auto":   "auto read-only + file edits + web",
    "yolo":   "auto everything (including shell)",
}


def cmd_auto(arg: str) -> None:
    arg = arg.strip().lower()
    if not arg:
        print(f"{GRAY}autonomy: {CYAN}{STATE.autonomy}{RESET} "
              f"{GRAY}— {_AUTO_DESCRIPTIONS.get(STATE.autonomy, '')}{RESET}")
        print(AUTO_USAGE); return
    if arg not in AUTONOMY_LEVELS:
        print(f"{YELLOW}expected one of: {', '.join(AUTONOMY_ORDER)}{RESET}")
        print(AUTO_USAGE); return
    if arg == "yolo":
        print(f"{YELLOW}warning:{RESET} yolo auto-approves ALL tool calls "
              f"(including shell). Guardrails still fire.")
        try:
            ans = input(f"{BOLD}Enable yolo? [y/N] {RESET}").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print(); return
        if ans not in ("y", "yes"):
            print(f"{GRAY}cancelled{RESET}"); return
    STATE.set_autonomy(arg)
    print(f"{GREEN}autonomy: {CYAN}{arg}{RESET} "
          f"{GRAY}({_AUTO_DESCRIPTIONS[arg]}){RESET}")


SEARCH_USAGE = (
    f"{GRAY}usage:\n"
    f"  /search                          show current search settings\n"
    f"  /search backend duckduckgo|brave|tavily\n"
    f"  /search key brave <key>          set the Brave API key\n"
    f"  /search key tavily <key>         set the Tavily API key\n"
    f"  /search max <n>                  default results per query (1-20)\n"
    f"  /search -r                       remove stored API keys{RESET}"
)


def _show_search_status() -> None:
    print(f"{GRAY}backend       : {CYAN}{STATE.search_backend}{RESET}")
    print(f"{GRAY}max results   : {CYAN}{STATE.search_max_results}{RESET}")
    print(f"{GRAY}brave api key : {CYAN}{_mask_api_key(STATE.brave_api_key)}{RESET}")
    print(f"{GRAY}tavily api key: {CYAN}{_mask_api_key(STATE.tavily_api_key)}{RESET}")
    if STATE._brave_from_env:
        print(f"{GRAY}(brave key from $AGENT0_BRAVE_KEY){RESET}")
    if STATE._tavily_from_env:
        print(f"{GRAY}(tavily key from $AGENT0_TAVILY_KEY){RESET}")
    print()
    print(SEARCH_USAGE)


def cmd_search(arg: str) -> None:
    arg = arg.strip()

    if not arg:
        _show_search_status(); return

    if arg == "-r":
        STATE.set_brave_key("")
        STATE.set_tavily_key("")
        print(f"{GREEN}search API keys cleared{RESET}")
        return

    parts = arg.split(maxsplit=2)
    sub = parts[0].lower()

    if sub == "backend":
        if len(parts) < 2:
            print(f"{YELLOW}usage: /search backend duckduckgo|brave|tavily{RESET}")
            return
        name = parts[1].lower()
        if name not in ("duckduckgo", "brave", "tavily"):
            print(f"{YELLOW}unknown backend: {name}{RESET}")
            return
        if name == "brave" and not STATE.brave_api_key:
            print(f"{YELLOW}note:{RESET} no Brave API key set. "
                  f"Add one with `/search key brave <key>`.")
        if name == "tavily" and not STATE.tavily_api_key:
            print(f"{YELLOW}note:{RESET} no Tavily API key set. "
                  f"Add one with `/search key tavily <key>`.")
        STATE.set_search_backend(name)
        print(f"{GREEN}search backend: {CYAN}{name}{RESET}")
        return

    if sub == "key":
        if len(parts) < 3:
            print(f"{YELLOW}usage: /search key brave|tavily <key>{RESET}")
            return
        service = parts[1].lower()
        key = parts[2].strip()
        if not key:
            print(f"{YELLOW}key is empty{RESET}"); return
        if service == "brave":
            STATE.set_brave_key(key)
            print(f"{GREEN}Brave key saved: {CYAN}{_mask_api_key(key)}{RESET}")
        elif service == "tavily":
            STATE.set_tavily_key(key)
            print(f"{GREEN}Tavily key saved: {CYAN}{_mask_api_key(key)}{RESET}")
        else:
            print(f"{YELLOW}unknown service: {service} (expected brave or tavily){RESET}")
        return

    if sub == "max":
        if len(parts) < 2:
            print(f"{YELLOW}usage: /search max <n>{RESET}"); return
        try:
            n = int(parts[1])
        except ValueError:
            print(f"{YELLOW}expected an integer{RESET}"); return
        STATE.set_search_max_results(n)
        print(f"{GREEN}max results: {CYAN}{STATE.search_max_results}{RESET}")
        return

    print(f"{YELLOW}unknown subcommand: {sub}{RESET}")
    print(SEARCH_USAGE)


BUDGET_USAGE = (
    f"{GRAY}usage:\n"
    f"  /budget                          show current limits\n"
    f"  /budget steps <n>\n"
    f"  /budget tools <n>\n"
    f"  /budget seconds <n>{RESET}"
)


def cmd_budget(arg: str) -> None:
    parts = arg.strip().split()
    if not parts:
        print(f"{GRAY}max steps     : {CYAN}{STATE.budget_max_steps}{RESET}")
        print(f"{GRAY}max tool calls: {CYAN}{STATE.budget_max_tools}{RESET}")
        print(f"{GRAY}max seconds   : {CYAN}{STATE.budget_max_seconds}{RESET}")
        print(BUDGET_USAGE); return
    if len(parts) != 2:
        print(BUDGET_USAGE); return
    kind, val = parts[0].lower(), parts[1]
    try:
        n = int(val)
    except ValueError:
        print(f"{YELLOW}expected an integer{RESET}"); return
    if kind == "steps": STATE.set_budget(steps=n)
    elif kind == "tools": STATE.set_budget(tools=n)
    elif kind == "seconds": STATE.set_budget(seconds=n)
    else:
        print(BUDGET_USAGE); return
    print(f"{GREEN}budget updated{RESET}")


MEMORY_USAGE = (
    f"{GRAY}usage: /memory | /memory -r (reset) | /memory -e (edit in $EDITOR){RESET}"
)


def cmd_memory(arg: str) -> None:
    arg = arg.strip()
    if not arg:
        print(f"{GRAY}memory file: {CYAN}{MEMORY_PATH}{RESET}")
        text = _load_memory_text()
        print(text if text else f"{GRAY}(empty){RESET}")
        return
    if arg == "-r":
        try:
            ans = input(f"{BOLD}Clear memory? [y/N] {RESET}").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print(); return
        if ans not in ("y", "yes"):
            print(f"{GRAY}cancelled{RESET}"); return
        try:
            _secure_write(MEMORY_PATH, _MEMORY_DEFAULT)
            print(f"{GREEN}memory reset{RESET}")
        except OSError as e:
            print(f"{RED}{e}{RESET}")
        return
    if arg == "-e":
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
        if not editor:
            for cand in ("nano", "vim", "vi", "code"):
                if shutil.which(cand):
                    editor = cand; break
        if not editor:
            print(f"{RED}no $EDITOR and no common editor found{RESET}"); return
        try:
            if not MEMORY_PATH.exists():
                _secure_write(MEMORY_PATH, _MEMORY_DEFAULT)
            subprocess.run([*shlex.split(editor), str(MEMORY_PATH)], check=False)
        except Exception as e:
            print(f"{RED}{e}{RESET}")
        return
    print(MEMORY_USAGE)


def cmd_forget() -> None:
    try:
        ans = input(f"{BOLD}Delete session file AND reset memory? [y/N] {RESET}").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print(); return
    if ans not in ("y", "yes"):
        print(f"{GRAY}cancelled{RESET}"); return
    deleted = delete_session()
    try:
        _secure_write(MEMORY_PATH, _MEMORY_DEFAULT)
    except OSError as e:
        print(f"{RED}{e}{RESET}")
    print(f"{GREEN}session {'deleted' if deleted else 'not present'}; memory reset{RESET}")


def cmd_resume(history: list) -> list:
    loaded = load_session()
    if loaded is None:
        print(f"{GRAY}no saved session found{RESET}")
        return history
    history.clear()
    history.extend(loaded)
    print(f"{GREEN}resumed {len(history)-1} messages{RESET}")
    if _TODOS:
        print(_format_todos())
    return history


def cmd_goal(arg: str, history: list) -> None:
    if not arg.strip():
        print(f"{YELLOW}usage: /goal <description>{RESET}")
        return
    if STATE.autonomy not in ("auto", "yolo"):
        print(f"{YELLOW}note:{RESET} goal mode works best with `/auto auto` or `/auto yolo`. "
              f"Currently: {STATE.autonomy}.")
    budget = Budget.from_state()
    run_goal(arg.strip(), history, budget)


def cmd_schedule(arg: str) -> None:
    parts = arg.strip().split(maxsplit=2)
    if not parts or parts[0] == "list":
        if not _SCHEDULE:
            print(f"{GRAY}(no scheduled tasks){RESET}")
        for i, t in enumerate(_SCHEDULE):
            rep = f" every {t.repeat_minutes}m" if t.repeat_minutes else ""
            print(f"  [{i}] {t.when.isoformat()}{rep}  {t.prompt}")
        return
    if parts[0] == "clear":
        _SCHEDULE.clear(); print(f"{GREEN}schedule cleared{RESET}"); return
    if parts[0] == "cancel":
        if len(parts) < 2 or not parts[1].isdigit():
            print(f"{YELLOW}usage: /schedule cancel <index>{RESET}"); return
        i = int(parts[1])
        if 0 <= i < len(_SCHEDULE):
            _SCHEDULE.pop(i); print(f"{GREEN}cancelled{RESET}")
        else:
            print(f"{YELLOW}index out of range{RESET}")
        return

    when: datetime.datetime | None = None
    repeat: int | None = None
    prompt: str = ""

    if parts[0] == "in" and len(parts) >= 3:
        secs = _parse_duration(parts[1])
        if secs is None:
            print(f"{YELLOW}could not parse duration '{parts[1]}' (try 5m, 30s, 2h){RESET}")
            return
        when = datetime.datetime.now() + datetime.timedelta(seconds=secs)
        prompt = parts[2]
    elif parts[0] == "at" and len(parts) >= 3:
        try:
            hh, mm = parts[1].split(":")
            now = datetime.datetime.now()
            when = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
            if when <= now:
                when += datetime.timedelta(days=1)
        except (ValueError, IndexError):
            print(f"{YELLOW}usage: /schedule at HH:MM <prompt>{RESET}"); return
        prompt = parts[2]
    else:
        print(f"{GRAY}usage:\n"
              f"  /schedule in <duration> <prompt>     e.g. /schedule in 5m check the build\n"
              f"  /schedule at HH:MM <prompt>          e.g. /schedule at 14:30 summarize commits\n"
              f"  /schedule every <duration> <prompt>  repeating\n"
              f"  /schedule list\n"
              f"  /schedule cancel <index>\n"
              f"  /schedule clear{RESET}")
        return

    if prompt.startswith("every "):
        rest = prompt[6:].split(maxsplit=1)
        if len(rest) == 2:
            rep_secs = _parse_duration(rest[0])
            if rep_secs and rep_secs >= 60:
                repeat = rep_secs // 60
                prompt = rest[1]

    _SCHEDULE.append(ScheduledTask(when=when, prompt=prompt, repeat_minutes=repeat))
    print(f"{GREEN}scheduled for {when.isoformat()}"
          + (f" (every {repeat}m)" if repeat else "")
          + f": {prompt}{RESET}")
    print(f"{GRAY}note: scheduled tasks fire when the REPL is idle between prompts.{RESET}")


API_USAGE = f"{GRAY}usage: /api <key> | /api -s | /api -r | /api{RESET}"


def cmd_api(arg: str) -> None:
    arg = arg.strip()
    if not arg or arg == "-s":
        if not _GROQ_AVAILABLE:
            print(f"{YELLOW}note:{RESET} 'groq' not installed. Run: pip install groq")
        print(f"{GRAY}provider      : {CYAN}{STATE.provider}{RESET}")
        print(f"{GRAY}groq api key  : {CYAN}{_mask_api_key(STATE.groq_api_key)}{RESET}")
        print(f"{GRAY}groq model    : {CYAN}{STATE.groq_model}{RESET}")
        if STATE._key_from_env:
            print(f"{GRAY}(key from $AGENT0_GROQ_API_KEY){RESET}")
        print(); print(API_USAGE)
        return
    if arg == "-r":
        if not STATE.groq_api_key:
            print(f"{GRAY}no key set{RESET}"); return
        STATE.set_groq_key("")
        if STATE.provider == "groq":
            STATE.set_provider("ollama")
        print(f"{GREEN}Groq key cleared{RESET}")
        return
    if arg.startswith("-"):
        print(f"{YELLOW}unknown flag: {arg}{RESET}"); print(API_USAGE); return
    if not _GROQ_AVAILABLE:
        print(f"{RED}Run: pip install groq{RESET}"); return
    STATE.set_groq_key(arg)
    if STATE.provider != "groq":
        STATE.set_provider("groq")
    print(f"{GREEN}Groq key saved: {CYAN}{_mask_api_key(arg)}{RESET}")
    print(f"{GRAY}provider: groq  model: {STATE.groq_model}{RESET}")


THINK_USAGE = f"{GRAY}usage: /think on|off | /think -s on|off | /think -f on|off | /think{RESET}"


def cmd_think(arg: str) -> None:
    parts = arg.strip().split()
    if not parts:
        print(f"{GRAY}thinking       : {CYAN}{'on' if STATE.think_enabled else 'off'}{RESET}")
        print(f"{GRAY}show thinking  : {CYAN}{'on' if STATE.show_thinking else 'off'}{RESET}")
        print(f"{GRAY}fast thinking  : {CYAN}{'on' if STATE.fast_think else 'off'}{RESET}")
        print(THINK_USAGE); return
    if len(parts) == 1:
        v = parts[0].lower()
        if v in ("on", "off"):
            STATE.set_thinking(think=(v == "on"))
            print(f"{GREEN}thinking {'on' if STATE.think_enabled else 'off'}{RESET}")
            return
        print(THINK_USAGE); return
    if len(parts) == 2:
        flag, v = parts[0], parts[1].lower()
        if v not in ("on", "off"):
            print(f"{YELLOW}expected on|off{RESET}"); return
        b = (v == "on")
        if flag == "-s":
            STATE.set_thinking(show=b)
            print(f"{GREEN}show thinking {'on' if b else 'off'}{RESET}"); return
        if flag == "-f":
            STATE.set_thinking(fast=b)
            print(f"{GREEN}fast thinking {'on' if b else 'off'}{RESET}"); return
    print(THINK_USAGE)


def cmd_tools() -> None:
    print(f"{BOLD}Tools{RESET}  {GRAY}(policy / auto-or-ask){RESET}")
    for name, desc in TOOL_DESCRIPTIONS.items():
        policy = _tool_policy(name)
        col = {"safe": GREEN, "medium": YELLOW, "dangerous": RED}.get(policy, GRAY)
        auto = "auto" if _autonomy_allows(name) else "ask "
        print(f"  {CYAN}{name:<16}{RESET}{col}{policy:<10}{RESET}{GRAY}{auto}  {desc}{RESET}")


def show_help() -> None:
    print(f"""{BOLD}Commands{RESET}
  {CYAN}/goal{RESET} <text>         run autonomously toward a goal
  {CYAN}/resume{RESET}              reload the last saved session
  {CYAN}/budget{RESET} [k v]        view or set resource limits
  {CYAN}/schedule{RESET} [...]      one-shot or repeating self-triggered tasks
  {CYAN}/forget{RESET}              delete session file and reset memory
  {CYAN}/model{RESET} <name>        switch model (for the current provider)
  {CYAN}/provider{RESET} o|g        switch backend (ollama|groq)
  {CYAN}/api{RESET} <key>|-s|-r     set / show / remove the Groq API key
  {CYAN}/search{RESET} [...]        configure web search (backend, keys, max)
  {CYAN}/auto{RESET} <level>        manual|smart|auto|yolo
  {CYAN}/memory{RESET} [-r|-e]      view / reset / edit memory file
  {CYAN}/think{RESET} [...]         thinking settings (ollama only)
  {CYAN}/tools{RESET}               list tools, policies, and auto status
  {CYAN}/todos{RESET}               show the session todo list
  {CYAN}/reset{RESET}               clear in-memory history and todos (keeps memory)
  {CYAN}/help{RESET}                show this help
  {CYAN}/bye{RESET}                 quit (Ctrl-C / Ctrl-D do nothing)
""")


# ---------------- Entrypoint ----------------

def _new_history() -> list:
    return [{"role": "system", "content": build_system_prompt()}]


def main() -> None:
    if not sys.stdin.isatty():
        print("This script needs an interactive terminal.", file=sys.stderr)
        sys.exit(1)

    _memory_init()

    print(WELCOME)
    print(f"{GRAY}autonomy: {STATE.autonomy}    provider: {STATE.provider}    "
          f"model: {STATE.current_model()}    search: {STATE.search_backend}{RESET}")
    print(f"{GRAY}memory:   {MEMORY_PATH}{RESET}")
    print(f"{GRAY}session:  {SESSION_PATH}{RESET}")
    print()

    history: list = _new_history()

    while True:
        _check_schedule(history)

        try:
            user_input = input(f"{BOLD}>>> {RESET}")
        except KeyboardInterrupt:
            print(f"\n{CTRL_C_HINT}"); continue
        except EOFError:
            print(f"\n{CTRL_C_HINT}"); continue

        stripped = user_input.strip()
        if not stripped:
            continue

        if stripped.startswith("/"):
            parts = stripped.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/bye", "/exit", "/quit"):
                break
            if cmd == "/help": show_help(); continue
            if cmd == "/tools": cmd_tools(); continue
            if cmd == "/todos":
                print(f"{BOLD}Todos{RESET}"); print(_format_todos()); continue
            if cmd == "/reset":
                history[:] = _new_history()
                _reset_todos()
                print(f"{GRAY}history and todos cleared (memory kept){RESET}")
                continue
            if cmd == "/forget": cmd_forget(); continue
            if cmd == "/resume": history = cmd_resume(history); continue
            if cmd == "/goal":
                try:
                    cmd_goal(arg, history)
                except KeyboardInterrupt:
                    print(f"\n{CTRL_C_HINT}")
                continue
            if cmd == "/schedule": cmd_schedule(arg); continue
            if cmd == "/budget": cmd_budget(arg); continue
            if cmd == "/model": cmd_model(arg); continue
            if cmd == "/provider": cmd_provider(arg); continue
            if cmd == "/api": cmd_api(arg); continue
            if cmd == "/search": cmd_search(arg); continue
            if cmd == "/auto": cmd_auto(arg); continue
            if cmd == "/memory": cmd_memory(arg); continue
            if cmd == "/think": cmd_think(arg); continue

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