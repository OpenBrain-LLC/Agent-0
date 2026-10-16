#!/usr/bin/env python3
"""Agent-0 — autonomous CLI AI agent (Ollama + Groq backends).

Config:   db/local/.config.json  ($AGENT0_CONFIG)
Memory:   db/local/memory.md     ($AGENT0_MEMORY)
Session:  db/local/session.json  ($AGENT0_SESSION)
Model:    deepseek-r1:1.5b       ($AGENT0_MODEL)
Groq key: /api <key> or $AGENT0_GROQ_API_KEY
Install:  pip install ollama groq.  Needs an interactive TTY.

Tools (only these five):
  shell           run a shell command        — ALWAYS requires [Y]es/[N]o
  web_search      DuckDuckGo web search
  fetch_url       HTTP GET a URL
  memory_append   append a bullet to memory.md
  memory_write    replace memory.md entirely

Search is always DuckDuckGo — no key, no config.
"""
from __future__ import annotations
import datetime, html, inspect, json, os, re, shlex, shutil, signal
import subprocess, sys, time, urllib.error, urllib.parse, urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import ollama; _OLLAMA_OK = True
except ImportError:
    ollama = None; _OLLAMA_OK = False
try:
    from groq import Groq as _GroqClient; _GROQ_OK = True
except ImportError:
    _GroqClient = None; _GROQ_OK = False

__version__ = "1.2.0"

# ---------------- ANSI ----------------
if sys.stdout.isatty() and not os.environ.get("NO_COLOR"):
    BOLD, DIM, GRAY, CYAN, YELLOW, GREEN, RED, MAGENTA = (
        "\033[1m", "\033[2m", "\033[90m", "\033[36m", "\033[33m",
        "\033[32m", "\033[31m", "\033[35m")
    RESET = "\033[0m"
else:
    BOLD = DIM = GRAY = CYAN = YELLOW = GREEN = RED = MAGENTA = RESET = ""

def warn(m): print(f"{GRAY}[warn] {m}{RESET}", file=sys.stderr)

def _bool(v, d=False):
    if isinstance(v, bool): return v
    if isinstance(v, str):  return v.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(v, (int, float)): return bool(v)
    return d

def _int(v, d):
    try: return int(v)
    except (TypeError, ValueError): return d

# ---------------- Paths ----------------
SCRIPT_DIR = Path(__file__).resolve().parent
_env = lambda n, d: Path(os.environ[n]).expanduser() if os.environ.get(n) else d
CONFIG_PATH  = _env("AGENT0_CONFIG",  SCRIPT_DIR/"db/local/.config.json")
MEMORY_PATH  = _env("AGENT0_MEMORY",  SCRIPT_DIR/"db/local/memory.md")
SESSION_PATH = _env("AGENT0_SESSION", SCRIPT_DIR/"db/local/session.json")

_PROTECTED  = {SESSION_PATH.resolve(), CONFIG_PATH.resolve()}
_PROT_NAMES = {"session.json", ".config.json", "goals.db", "audit.jsonl"}
_LOCAL_BASE = (SCRIPT_DIR/"db/local").resolve()

def _is_protected(p: Path) -> bool:
    try: r = p.expanduser().resolve()
    except (OSError, ValueError): return False
    if r in _PROTECTED: return True
    if r == _LOCAL_BASE or _LOCAL_BASE in r.parents:
        return r.name != "memory.md"
    return r.name in _PROT_NAMES

def _secure_write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    try: os.chmod(p, 0o600)
    except OSError: pass

DEFAULT_OLLAMA = os.environ.get("AGENT0_MODEL") or "deepseek-r1:1.5b"
DEFAULT_GROQ   = "llama-3.3-70b-versatile"

_DEF_CFG = {
    "model": DEFAULT_OLLAMA, "provider": "ollama", "autonomy": "auto",
    "budget": {"max_steps": 50, "max_tool_calls": 200, "max_seconds": 900},
    "groq": {"api_key": "", "model": DEFAULT_GROQ},
    "thinking": {"show": "False", "think": "True", "fast": "False"},
    "ollama": {"keep_alive": "30m", "num_ctx": 4096, "num_predict": -1},
}

MAX_STEPS = 24; TOOL_OUT_LIMIT = 20_000; TOOL_PREVIEW = 500
MEM_INJECT = 6000; MEM_MAX = 200_000; SESSION_CAP = 60
SHELL_T = 60; FETCH_T = 20; FETCH_MAX = 500_000
SEARCH_T = 15; SEARCH_MAX = 2_000_000; TEMP = 0.25

CTRL_C_HINT = f"{GRAY}Ctrl-C disabled — type /bye to exit{RESET}"
WELCOME = f"Welcome to Agent-0 AI [v{__version__}]\nBy: OpenBrain-LLC"
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

def load_config() -> dict:
    cfg = json.loads(json.dumps(_DEF_CFG))
    if not CONFIG_PATH.exists():
        warn(f"{CONFIG_PATH} not found; creating defaults")
        try: save_config(cfg)
        except OSError as e: warn(f"could not create config: {e}")
        return cfg
    try: data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError) as e:
        warn(f"bad config ({e}); using defaults"); return cfg
    if not isinstance(data, dict): return cfg
    if isinstance(data.get("model"), str): cfg["model"] = data["model"]
    if data.get("provider") in ("ollama", "groq"): cfg["provider"] = data["provider"]
    if data.get("autonomy") in ("manual", "smart", "auto", "yolo"):
        cfg["autonomy"] = data["autonomy"]
    for k in ("max_steps", "max_tool_calls", "max_seconds"):
        if isinstance(data.get("budget"), dict) and k in data["budget"]:
            cfg["budget"][k] = _int(data["budget"][k], cfg["budget"][k])
    g = data.get("groq") or {}
    if isinstance(g.get("api_key"), str): cfg["groq"]["api_key"] = g["api_key"]
    if isinstance(g.get("model"), str):   cfg["groq"]["model"]   = g["model"]
    for k in ("show", "think", "fast"):
        if k in (data.get("thinking") or {}): cfg["thinking"][k] = data["thinking"][k]
    o = data.get("ollama") or {}
    if isinstance(o.get("keep_alive"), (str, int)): cfg["ollama"]["keep_alive"] = o["keep_alive"]
    if "num_ctx" in o:     cfg["ollama"]["num_ctx"] = _int(o["num_ctx"], 4096)
    if "num_predict" in o: cfg["ollama"]["num_predict"] = _int(o["num_predict"], -1)
    return cfg

def save_config(cfg): _secure_write(CONFIG_PATH, json.dumps(cfg, indent=4) + "\n")

def _mask(k):
    if not k: return "(not set)"
    return "*"*len(k) if len(k) < 12 else f"{k[:min(7,len(k)-5)]}****...**{k[-5:]}"

# ---------------- State ----------------
class State:
    def __init__(self):
        self.config = load_config()
        self.model    = self.config["model"]
        self.provider = self.config.get("provider", "ollama")
        self.autonomy = self.config.get("autonomy", "auto")
        b = self.config.get("budget") or {}
        self.bmax_steps = _int(b.get("max_steps"), 50)
        self.bmax_tools = _int(b.get("max_tool_calls"), 200)
        self.bmax_secs  = _int(b.get("max_seconds"), 900)
        g = self.config.get("groq") or {}
        self.groq_model = g.get("model", DEFAULT_GROQ)
        env_k = os.environ.get("AGENT0_GROQ_API_KEY", "")
        self._key_env = bool(env_k)
        self._key_saved = g.get("api_key", "")
        self.groq_api_key = env_k or self._key_saved
        th = self.config["thinking"]
        self.show_think = _bool(th.get("show"), False)
        self.think_on   = _bool(th.get("think"), True)
        self.fast_think = _bool(th.get("fast"), False)
        o = self.config.get("ollama") or {}
        self.keep_alive  = o.get("keep_alive", "30m")
        self.num_ctx     = _int(o.get("num_ctx"), 4096)
        self.num_predict = _int(o.get("num_predict"), -1)
        self._sync()

    def _sync(self):
        c = self.config
        c["model"], c["provider"], c["autonomy"] = self.model, self.provider, self.autonomy
        c["budget"] = {"max_steps": self.bmax_steps,
                       "max_tool_calls": self.bmax_tools,
                       "max_seconds": self.bmax_secs}
        gc = {"model": self.groq_model}
        gc["api_key"] = self._key_saved if self._key_env else self.groq_api_key
        c["groq"] = gc
        c["thinking"] = {"show": str(self.show_think), "think": str(self.think_on),
                         "fast": str(self.fast_think)}
        c["ollama"] = {"keep_alive": self.keep_alive, "num_ctx": self.num_ctx,
                       "num_predict": self.num_predict}

    def _persist(self):
        try: save_config(self.config)
        except OSError as e: warn(f"persist failed: {e}")

    def set(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)
        self._sync(); self._persist()

    def set_groq_key(self, k):
        self.groq_api_key = self._key_saved = k; self._key_env = False
        self._sync(); self._persist()

    def current_model(self): return self.groq_model if self.provider == "groq" else self.model

    def think_param(self):
        if not self.think_on: return False
        return "low" if self.fast_think else True

STATE = State()

# ---------------- Autonomy ----------------
# Shell is special: it ALWAYS requires [Y]es/[N]o confirmation, at every level.
# The remaining tools follow normal autonomy policy.
AUTONOMY = {
    "manual": {"safe": False, "medium": False, "dangerous": False},
    "smart":  {"safe": True,  "medium": False, "dangerous": False},
    "auto":   {"safe": True,  "medium": True,  "dangerous": False},
    "yolo":   {"safe": True,  "medium": True,  "dangerous": True},
}
AUTO_ORDER = ["manual", "smart", "auto", "yolo"]
_AUTO_DESC = {"manual": "confirm every tool call", "smart": "auto safe tools",
              "auto": "auto safe + medium tools", "yolo": "auto safe + medium + dangerous"}

def _autonomy_ok(name: str) -> bool:
    level = AUTONOMY.get(STATE.autonomy, AUTONOMY["auto"])
    if name == "shell":        return False   # shell is NEVER auto-run
    if name == "spawn":        return False   # (not present, kept for safety)
    return bool(level.get("medium", False))

# ---------------- Guardrails ----------------
_BLOCKED_SHELL = [
    (r"\brm\s+-rf?\s+/(?:\s|$)", "rm -rf /"),
    (r":\(\)\s*\{\s*:\|:&\s*\}\s*;:", "fork bomb"),
    (r"\bdd\b[^\n]*\bof=/dev/(?!null)", "raw device write"),
    (r"\bmkfs\.", "filesystem format"),
    (r"\bgit\s+push\b[^\n]*--force[^\n]*\b(main|master|trunk)\b", "force-push"),
    (r"\bgit\s+push\b[^\n]*\s-f\b[^\n]*\b(main|master|trunk)\b", "force-push"),
    (r"\bchmod\s+-R\s+777\s+/(?:\s|$)", "chmod -R 777 /"),
    (r"\b(?:curl|wget)\b[^\n]*\|\s*(?:sh|bash|zsh|fish)\b", "pipe-to-shell"),
    (r"\bsudo\s+rm\s+-rf?\s+/(?:\s|$)", "sudo rm -rf /"),
]

def _guard_shell(cmd: str) -> str | None:
    for pat, label in _BLOCKED_SHELL:
        try:
            if re.search(pat, cmd, re.IGNORECASE):
                return f"guardrail blocked: {label}"
        except re.error: continue
    return None

# ---------------- Memory ----------------
_MEM_DEFAULT = """# Agent-0 Memory

## User
<!-- durable facts: name, prefs, timezone, projects -->

## Projects
<!-- ongoing work, layout notes, decisions -->

## Notes
<!-- gotchas, commands, follow-ups -->
"""

_mem_cache: tuple[float, str] | None = None

def _mem_init():
    if MEMORY_PATH.exists(): return
    try: _secure_write(MEMORY_PATH, _MEM_DEFAULT)
    except OSError as e: warn(f"could not create memory: {e}")

def _load_mem():
    global _mem_cache
    try: mtime = MEMORY_PATH.stat().st_mtime if MEMORY_PATH.exists() else 0.0
    except OSError: mtime = 0.0
    if _mem_cache is not None and _mem_cache[0] == mtime:
        return _mem_cache[1]
    try:
        if not MEMORY_PATH.exists(): text = ""
        else: text = MEMORY_PATH.read_text(encoding="utf-8").strip()
    except OSError as e: text = f"(could not read memory: {e})"
    if len(text) > MEM_INJECT:
        text = text[:MEM_INJECT] + f"\n\n... [truncated at {MEM_INJECT}]"
    _mem_cache = (mtime, text)
    return text

def _mem_invalidate(): global _mem_cache; _mem_cache = None

def tool_memory_read():
    """Not exposed as a model tool, but used internally by /memory."""
    try:
        if not MEMORY_PATH.exists(): return "(memory file does not exist yet)"
        t = MEMORY_PATH.read_text(encoding="utf-8")
        return _trunc(t) if t.strip() else "(memory file is empty)"
    except Exception as e: return f"Error: {e}"

def tool_memory_write(content):
    try:
        if not isinstance(content, str): return "Error: content must be a string"
        e = content.encode("utf-8")
        if len(e) > MEM_MAX: return f"Error: memory too large ({len(e)} > {MEM_MAX})"
        _secure_write(MEMORY_PATH, content); _mem_invalidate()
        return f"Wrote {len(e)} bytes to {MEMORY_PATH}"
    except Exception as e: return f"Error: {e}"

def tool_memory_append(section, entry):
    try:
        section = (section or "").strip().lstrip("#").strip()
        entry = (entry or "").strip()
        if not section: return "Error: 'section' must be a non-empty string"
        if not entry:   return "Error: 'entry' must be a non-empty string"
        if not entry.startswith(("-", "*", "#")): entry = "- " + entry
        entry = entry.replace("\n", " ").strip()
        text = MEMORY_PATH.read_text(encoding="utf-8") if MEMORY_PATH.exists() else "# Agent-0 Memory\n"
        lines = text.rstrip("\n").split("\n")
        header = f"## {section}"
        sidx = nidx = None
        for i, ln in enumerate(lines):
            s = ln.strip()
            if sidx is None and s == header: sidx = i; continue
            if sidx is not None and s.startswith("## "): nidx = i; break
        if sidx is None:
            if lines and lines[-1].strip() != "": lines.append("")
            lines += [header, entry]
        else:
            at = nidx if nidx is not None else len(lines)
            while at > sidx + 1 and lines[at-1].strip() == "": at -= 1
            lines.insert(at, entry)
        new = "\n".join(lines).rstrip("\n") + "\n"
        if len(new.encode("utf-8")) > MEM_MAX: return f"Error: memory would exceed {MEM_MAX} bytes"
        _secure_write(MEMORY_PATH, new); _mem_invalidate()
        return f"Appended to memory under '{section}'."
    except Exception as e: return f"Error: {e}"

# ---------------- Session ----------------
def save_session(history):
    try:
        trimmed = [m for m in history[-SESSION_CAP:] if m.get("role") != "system"]
        _secure_write(SESSION_PATH, json.dumps({
            "saved_at": datetime.datetime.now().isoformat(),
            "provider": STATE.provider, "model": STATE.current_model(),
            "history": trimmed,
        }, indent=2, default=str))
    except OSError as e: warn(f"could not save session: {e}")

def load_session():
    if not SESSION_PATH.exists(): return None
    try: data = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e: warn(f"could not load session ({e})"); return None
    h = data.get("history")
    if not isinstance(h, list): return None
    sp = {"role": "system", "content": build_system_prompt()}
    if h and isinstance(h[0], dict) and h[0].get("role") == "system": h[0] = sp
    else: h.insert(0, sp)
    return h

def delete_session():
    try:
        if SESSION_PATH.exists(): SESSION_PATH.unlink(); return True
    except OSError as e: warn(f"could not delete session: {e}")
    return False

# ---------------- Loop & Budget ----------------
class LoopDetector:
    def __init__(self, w=4): self.w = w; self.recent = []
    def observe(self, fp):
        self.recent.append(fp)
        if len(self.recent) > self.w: self.recent.pop(0)
        return len(self.recent) == self.w and len(set(self.recent)) == 1
    def reset(self): self.recent.clear()

def _fingerprint(tcs):
    if not tcs: return "no-tools"
    return "|".join(f"{t.name}({json.dumps(t.arguments, sort_keys=True, default=str)})" for t in tcs)

@dataclass
class Budget:
    max_steps: int = 50; max_tool_calls: int = 200; max_seconds: int = 900
    steps: int = 0; tool_calls: int = 0
    started: float = field(default_factory=time.monotonic)
    def exhausted(self):
        return (self.steps >= self.max_steps or self.tool_calls >= self.max_tool_calls
                or time.monotonic() - self.started > self.max_seconds)
    def status(self):
        return (f"steps {self.steps}/{self.max_steps}  tools {self.tool_calls}/{self.max_tool_calls}  "
                f"{int(time.monotonic()-self.started)}s/{self.max_seconds}s")
    @classmethod
    def from_state(cls):
        return cls(STATE.bmax_steps, STATE.bmax_tools, STATE.bmax_secs)

# ---------------- Schedule ----------------
@dataclass
class ScheduledTask:
    when: datetime.datetime; prompt: str; repeat_minutes: int | None = None
_SCHEDULE: list[ScheduledTask] = []

def _check_schedule(history):
    now = datetime.datetime.now()
    for t in list(_SCHEDULE):
        if t.when <= now:
            print(f"\n{MAGENTA}[scheduled]{RESET} {t.prompt}")
            try: run_agent(t.prompt, history)
            except KeyboardInterrupt: print(f"\n{CTRL_C_HINT}")
            if t.repeat_minutes: t.when = now + datetime.timedelta(minutes=t.repeat_minutes)
            else: _SCHEDULE.remove(t)

def _parse_dur(s):
    m = re.fullmatch(r"\s*(\d+)\s*([smhd])\s*", s.lower())
    return None if not m else int(m.group(1)) * {"s":1,"m":60,"h":3600,"d":86400}[m.group(2)]

# ---------------- Web search (DuckDuckGo only) ----------------
def _http_get(url, timeout=SEARCH_T, max_bytes=SEARCH_MAX):
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "*/*"}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(max_bytes), r.headers.get_content_charset() or "utf-8"

_clean = lambda s: " ".join(html.unescape(re.sub(r"<[^>]+>", "", s)).split())
_DDG_PREFIXES = ("//duckduckgo.com/l/", "https://duckduckgo.com/l/", "http://duckduckgo.com/l/")

def _ddg_unwrap(href):
    if href.startswith("//"): href = "https:" + href
    for p in _DDG_PREFIXES:
        if href.startswith(p):
            try:
                q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                real = q.get("uddg", [""])[0]
                if real: return urllib.parse.unquote(real)
            except Exception: pass
    return href

_DDG_BLOCK   = re.compile(r'<div[^>]*class="[^"]*\bresult\b[^"]*"[^>]*>(.*?)(?=<div[^>]*class="[^"]*\bresult\b|$)', re.S|re.I)
_DDG_LINK    = re.compile(r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S|re.I)
_DDG_SNIP    = re.compile(r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>', re.S|re.I)
_DDG_L_LINK  = re.compile(r'<a[^>]*class="[^"]*result-link[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S|re.I)
_DDG_L_SNIP  = re.compile(r'<td[^>]*class="[^"]*result-snippet[^"]*"[^>]*>(.*?)</td>', re.S|re.I)

def _parse_ddg_html(txt, limit):
    out = []
    for bm in _DDG_BLOCK.finditer(txt):
        blk = bm.group(1); lm = _DDG_LINK.search(blk)
        if not lm: continue
        href = _ddg_unwrap(lm.group(1).strip()); title = _clean(lm.group(2))
        sm = _DDG_SNIP.search(blk)
        if href and title: out.append({"title": title, "url": href, "snippet": _clean(sm.group(1)) if sm else ""})
        if len(out) >= limit: break
    return out

def _parse_ddg_lite(txt, limit):
    links = list(_DDG_L_LINK.finditer(txt)); snips = list(_DDG_L_SNIP.finditer(txt)); out = []
    for i, lm in enumerate(links[:limit]):
        href = _ddg_unwrap(lm.group(1).strip()); title = _clean(lm.group(2))
        if href and title:
            out.append({"title": title, "url": href,
                        "snippet": _clean(snips[i].group(1)) if i < len(snips) else ""})
    return out

def _fmt_results(q, rs):
    ls = [f"Web search (DuckDuckGo) for: {q}", ""]
    for i, r in enumerate(rs, 1):
        ls += [f"{i}. {r['title']}", f"   {r['url']}"]
        if r.get("snippet"): ls.append(f"   {r['snippet']}")
        ls.append("")
    return "\n".join(ls).rstrip()

def tool_web_search(query, num_results=None):
    if not isinstance(query, str) or not query.strip(): return "Error: 'query' must be non-empty"
    n = max(1, min(20, _int(num_results, 8)))
    qq = urllib.parse.quote_plus(query.strip())
    last = "no attempt"
    for url, parser in ((f"https://html.duckduckgo.com/html/?q={qq}", _parse_ddg_html),
                        (f"https://lite.duckduckgo.com/lite/?q={qq}", _parse_ddg_lite)):
        try:
            raw, cs = _http_get(url)
            body = raw.decode(cs, errors="replace")
        except urllib.error.HTTPError as e: last = f"HTTP {e.code} {e.reason}"; continue
        except urllib.error.URLError as e: last = f"network: {e.reason}"; continue
        except Exception as e: last = str(e); continue
        rs = parser(body, n)
        if rs: return _trunc(_fmt_results(query, rs))
        last = "no results parsed (layout changed or rate limit)"
    return f"Error: DuckDuckGo failed ({last}). Try again in a moment."

# ---------------- Tool schema (5 tools only) ----------------
def _fn(name, desc, props, req):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props,
                       "required": req, "additionalProperties": False}}}

TOOL_SCHEMAS = [
    _fn("shell",
        f"Run a shell command. Combined stdout+stderr, exit code. {SHELL_T}s timeout. "
        f"ALWAYS requires user confirmation.",
        {"command": {"type": "string"}}, ["command"]),
    _fn("web_search",
        "Search the web (DuckDuckGo). Returns title/URL/snippet. Use fetch_url to open a result.",
        {"query": {"type": "string"}, "num_results": {"type": "integer"}}, ["query"]),
    _fn("fetch_url",
        "HTTP GET a URL; returns text body (~500 KB cap).",
        {"url": {"type": "string"}}, ["url"]),
    _fn("memory_append",
        "Append one bullet to a section of the persistent memory file.",
        {"section": {"type": "string"}, "entry": {"type": "string"}}, ["section", "entry"]),
    _fn("memory_write",
        "Replace the ENTIRE memory file. Prefer memory_append for a single fact.",
        {"content": {"type": "string"}}, ["content"]),
]

TOOL_DESC = {
    "shell":         "run a shell command (always confirms)",
    "web_search":    "search the web (DuckDuckGo)",
    "fetch_url":     "HTTP GET a URL",
    "memory_append": "append a memory bullet",
    "memory_write":  "replace memory",
}

# ---------------- Prompts ----------------
SYSTEM_PROMPT = """You are Agent-0, an autonomous CLI agent on the user's machine. You work multi-step tasks independently, search the web when needed, and remember durable facts.

# Environment
{env}

# Tools (only these five)
- `shell`         — run a shell command. ALWAYS requires the user to approve.
- `web_search`    — DuckDuckGo search; returns titles, URLs, snippets.
- `fetch_url`     — HTTP GET a URL; returns its text body.
- `memory_append` — append one bullet to a section of the memory file.
- `memory_write`  — replace the ENTIRE memory file (rare; prefer append).

There are no file-reading, file-writing, or directory tools. To inspect or modify files, use `shell` (e.g. `cat`, `ls`, `sed`, `head`, `tail`, `wc`, `find`, `grep`).

# Memory ({mem_path})
Read the current memory below. Save durable facts with `memory_append` (section: User / Projects / Notes). Never store secrets, keys, or credentials.
### Current memory
{mem}

# Your own state — do NOT read/write
`db/local/` holds agent internals (session.json, .config.json). Never `cat` or `grep` those — it is a policy violation even if shell would allow it.

# Reply pattern
Each turn: EITHER call tools OR reply in prose. Never both. Never describe a call in prose — just call it.

# Tone
Concise. No preamble/postamble/narration. Answer in prose unless code was asked. If asked to do something, do it. Never speculate about code you have not read.

# Doing tasks
1. UNDERSTAND. Search the web when the task depends on current info. Use `shell` (`ls`, `cat`, `grep`) to inspect files before acting on them.
2. PLAN. For 3+ step tasks, state the plan in one line of prose before starting.
3. ACT. Prefer small, targeted `shell` commands over one giant script.
4. VERIFY. Reread output; if you wrote a file, `cat` it back.
5. REMEMBER. Save durable facts with `memory_append`.

# Web search when
Current events, recent releases, time-sensitive info; unfamiliar or version-specific library APIs; explicit "find/look up/search". NOT for local files or stable general knowledge. Workflow: `web_search` → `fetch_url` the top hits → synthesize. Don't paste raw search output.

# Safety — never without explicit user instruction
git commit/push/reset --hard/clean -fd, rm -rf, chmod -R, force-push, package installs, or anything that sends data off-machine.
"""
FAST_NOTE = "\n\n# Fast mode\nKeep reasoning extremely brief."
SUMMARY_SYS = ("You are a safety reviewer for a CLI AI agent. Given ONE tool call as JSON, "
               "reply in ONE short sentence: exactly what it will do on the user's machine. "
               "Be concrete (command, paths, side effects, network). Output ONLY the sentence.")
CRITIC_SYS = ("Evaluate whether a CLI agent achieved its goal. Reply EXACTLY one word on line 1:\n"
              "DONE | CONTINUE | STUCK\nLine 2: one sentence of justification. Nothing else.")
REFLECT = ("You appear to be stuck or looping. Before retrying:\n"
           "1) One sentence: what were you trying?\n2) One sentence: why did the last attempt fail?\n"
           "3) One sentence: a DIFFERENT approach not yet tried.\nThen continue. Do not repeat the same call.")

def _env_block():
    try: cwd = Path.cwd()
    except Exception: cwd = "(unknown)"
    try: import platform; plat = f"{platform.system()} {platform.release()}"
    except Exception: plat = "(unknown)"
    return (f"- Working directory: {cwd}\n- Platform: {plat}\n"
            f"- Date: {datetime.date.today().isoformat()}")

def build_system_prompt():
    p = SYSTEM_PROMPT.format(env=_env_block(), mem_path=MEMORY_PATH,
                             mem=_load_mem() or "(empty)")
    return p + FAST_NOTE if STATE.fast_think else p

# ---------------- Tool implementations ----------------
def _trunc(s, limit=TOOL_OUT_LIMIT):
    return s if len(s) <= limit else s[:limit] + f"\n... [truncated, {len(s)-limit} more chars]"

def _kill_tree(proc):
    try:
        if os.name == "posix": os.killpg(proc.pid, signal.SIGKILL)
        else: subprocess.run(["taskkill","/F","/T","/PID",str(proc.pid)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except (ProcessLookupError, PermissionError, OSError): pass

def tool_shell(command):
    if not isinstance(command, str): return "Error: 'command' must be a string"
    posix = os.name == "posix"
    try:
        proc = subprocess.Popen(command, shell=True, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, errors="replace", start_new_session=posix)
    except Exception as e: return f"Error: {e}"
    try: out, err = proc.communicate(timeout=SHELL_T)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try: proc.communicate(timeout=5)
        except Exception: pass
        return f"Error: command timed out after {SHELL_T}s"
    except BaseException:
        _kill_tree(proc); raise
    o = ((out or "") + (err or "")).strip(); rc = proc.returncode
    if not o: return f"(no output, exit code {rc})"
    o = _trunc(o)
    return o + (f"\n[exit code {rc}]" if rc != 0 else "")

def tool_fetch_url(url):
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"): return "Error: only http/https"
        req = urllib.request.Request(url, headers={"User-Agent": f"Agent-0/{__version__}"}, method="GET")
        with urllib.request.urlopen(req, timeout=FETCH_T) as r:
            raw = r.read(FETCH_MAX + 1)
            trunc = len(raw) > FETCH_MAX
            raw = raw[:FETCH_MAX]
            cs = r.headers.get_content_charset() or "utf-8"
            status, ctype = r.status, r.headers.get("Content-Type", "?")
        try: text = raw.decode(cs, errors="replace")
        except (LookupError, UnicodeDecodeError): text = raw.decode("utf-8", errors="replace")
        head = f"HTTP {status} {url}\nContent-Type: {ctype}\n\n"
        body = _trunc(text, TOOL_OUT_LIMIT - len(head) - 50)
        return head + body + (f"\n... [truncated at {FETCH_MAX} bytes]" if trunc else "")
    except urllib.error.HTTPError as e: return f"Error: HTTP {e.code} {e.reason} for {url}"
    except urllib.error.URLError as e: return f"Error: could not reach {url}: {e.reason}"
    except Exception as e: return f"Error: {e}"

TOOLS = {
    "shell": tool_shell,
    "web_search": tool_web_search,
    "fetch_url": tool_fetch_url,
    "memory_append": tool_memory_append,
    "memory_write": tool_memory_write,
}

def fmt_call(call):
    n = call.get("name"); a = call.get("args") or {}
    if not isinstance(a, dict): return f"{n}: {a!r}"
    if n == "shell":         return f"shell: {a.get('command','')}"
    if n == "web_search":    return f"web_search: {a.get('query','')}"
    if n == "fetch_url":     return f"fetch_url: {a.get('url','')}"
    if n == "memory_append":
        e = str(a.get("entry", "")); short = e if len(e) <= 80 else e[:77] + "..."
        return f"memory_append: [{a.get('section','')}] {short}"
    if n == "memory_write":  return f"memory_write: {len(str(a.get('content','')))} chars"
    return f"{n}: {json.dumps(a, default=str)}"

# ---------------- ToolCall & validation ----------------
@dataclass
class ToolCall:
    id: str; name: str; arguments: dict
    raw_arguments: str = ""; parse_error: str | None = None
    def to_openai(self):
        return {"id": self.id, "type": "function",
                "function": {"name": self.name, "arguments": self.raw_arguments or json.dumps(self.arguments)}}

def _mk_call(cid, name, raw):
    try:
        args = json.loads(raw, strict=False) if raw.strip() else {}
        if not isinstance(args, dict): raise ValueError(f"args must be object, got {type(args).__name__}")
        return ToolCall(cid, name, args, raw)
    except (json.JSONDecodeError, ValueError) as e:
        return ToolCall(cid, name, {}, raw, str(e))

_OK_ARG_TYPES = (str, bool, int, float, list, dict)

def _norm_args(name, args):
    try: sig = inspect.signature(TOOLS[name])
    except (KeyError, TypeError, ValueError): return args
    out = {}
    for k, v in args.items():
        p = sig.parameters.get(k)
        if p is not None and p.default is not inspect.Parameter.empty and (v is None or v == ""):
            continue
        out[k] = v
    return out

def validate_call(call):
    n = call.get("name")
    if not isinstance(n, str) or n not in TOOLS:
        return f"Error: unknown tool '{n}'. Available: {', '.join(TOOLS)}"
    a = call.get("args")
    if not isinstance(a, dict): return f"Error: 'args' for {n} must be an object"
    for k, v in a.items():
        if v is None: return f"Error: argument '{k}' for {n} is null — omit it or give a value"
        if not isinstance(v, _OK_ARG_TYPES): return f"Error: argument '{k}' has unsupported type {type(v).__name__}"
    try: inspect.signature(TOOLS[n]).bind(**a)
    except TypeError as e: return f"Error: bad arguments for {n}: {e}"
    return None

def _hint(e):
    m = str(e); low = m.lower()
    if any(w in low for w in ("connection", "refused", "unreachable", "connect")):
        return f"{m}\n  hint: backend unreachable. Ollama running? Groq network / /api <key>?"
    if "timed out" in low or "timeout" in low: return f"{m}\n  hint: timed out."
    if "401" in m or "unauthorized" in low or "invalid api key" in low:
        return f"{m}\n  hint: auth failed. /api <key>."
    if "403" in m or "forbidden" in low: return f"{m}\n  hint: forbidden."
    if "429" in m or "rate limit" in low: return f"{m}\n  hint: rate limited. Wait or switch model."
    if "model" in low and any(w in low for w in ("not found", "does not exist", "unknown model")):
        return f"{m}\n  hint: model unavailable. /model <name>."
    if "context" in low and any(w in low for w in ("length", "too long", "maximum")):
        return f"{m}\n  hint: context full. /reset."
    if "tool" in low and any(w in low for w in ("not supported", "unsupported", "does not support")):
        return f"{m}\n  hint: model lacks native tool calling."
    if "think" in low: return f"{m}\n  hint: think unsupported. /think off."
    return m

# ---------------- Ollama ----------------
_THINK_CACHE: dict = {}
_MISSING = object()

def _think_cands(req):
    cands = []
    cached = _THINK_CACHE.get((STATE.model, repr(req)), _MISSING)
    if cached is not _MISSING: cands.append(cached)
    cands.append(req)
    if isinstance(req, str): cands.append(True)
    cands.append(None)
    return [c for i, c in enumerate(cands) if c not in cands[:i]]

def _is_think_err(e):
    if isinstance(e, (TypeError, ValueError)): return True
    if ollama is not None and hasattr(ollama, "ResponseError"):
        try:
            if isinstance(e, ollama.ResponseError): return "think" in str(e).lower()
        except Exception: pass
    return False

def _req_ollama():
    if not _OLLAMA_OK: raise RuntimeError("'ollama' not installed. Run: pip install ollama")

def _oll_kw(msgs, stream, think, tools):
    opts = {"temperature": TEMP, "num_ctx": STATE.num_ctx}
    if STATE.num_predict and STATE.num_predict > 0:
        opts["num_predict"] = STATE.num_predict
    kw = dict(model=STATE.model, messages=msgs, stream=stream,
              options=opts, keep_alive=STATE.keep_alive)
    if think is not None: kw["think"] = think
    if tools: kw["tools"] = tools
    return kw

def _oll_stream(msgs, tools):
    req = STATE.think_param(); last = None
    for think in _think_cands(req):
        try:
            it = iter(ollama.chat(**_oll_kw(msgs, True, think, tools)))
            first = next(it)
        except StopIteration:
            _THINK_CACHE[(STATE.model, repr(req))] = think; return
        except Exception as e:
            if _is_think_err(e): last = e; continue
            raise
        _THINK_CACHE[(STATE.model, repr(req))] = think
        yield first; yield from it; return
    if last: raise last

def _oll_once(msgs, tools=None):
    req = STATE.think_param(); last = None
    for think in _think_cands(req):
        try: return ollama.chat(**_oll_kw(msgs, False, think, tools))
        except Exception as e:
            if _is_think_err(e): last = e; continue
            raise
    if last: raise last
    return None

def _to_oll_msgs(msgs):
    out = []
    for m in msgs:
        r = m.get("role")
        if r == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}; a = fn.get("arguments")
                if isinstance(a, str):
                    try: a = json.loads(a, strict=False)
                    except json.JSONDecodeError: a = {}
                tcs.append({"function": {"name": fn.get("name", ""), "arguments": a}})
            out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs})
        elif r == "tool":
            out.append({"role": "tool", "content": m.get("content", ""), "tool_name": m.get("tool_name", "")})
        else: out.append(m)
    return out

def _iter_oll(msgs):
    _req_ollama()
    for ch in _oll_stream(_to_oll_msgs(msgs), TOOL_SCHEMAS):
        m = getattr(ch, "message", None) or {}
        t = getattr(m, "thinking", None)
        if t: yield "thinking", t
        c = getattr(m, "content", None)
        if c: yield "content", c
        for i, rtc in enumerate(getattr(m, "tool_calls", None) or []):
            fn = getattr(rtc, "function", None) or {}
            name = getattr(fn, "name", "") or ""
            a = getattr(fn, "arguments", None)
            raw = a if isinstance(a, str) else (json.dumps(a) if isinstance(a, dict) else "{}")
            yield "tool_call", _mk_call(getattr(rtc, "id", None) or f"call_{i}", name, raw)

# ---------------- Groq ----------------
class _GroqErr(RuntimeError): pass

def _req_groq():
    if not _GROQ_OK: raise RuntimeError("'groq' not installed. Run: pip install groq")
    if not STATE.groq_api_key: raise RuntimeError("No Groq API key. /api <key> or $AGENT0_GROQ_API_KEY.")

def _groq_msg(e):
    low = str(e).lower()
    if "tool_choice" in low and "none" in low:
        return "Groq rejected a tool call (tool_choice mismatch). Retry or switch model."
    return str(e)

def _groq_create(msgs, stream, tools=None):
    kw = dict(model=STATE.groq_model, messages=msgs, temperature=TEMP, stream=stream)
    if tools: kw["tools"] = tools; kw["tool_choice"] = "auto"
    return _GroqClient(api_key=STATE.groq_api_key).chat.completions.create(**kw)

def _to_groq_msgs(msgs):
    return [{"role": "tool", "tool_call_id": m.get("tool_call_id", ""), "content": m.get("content", "")}
            if m.get("role") == "tool" else m for m in msgs]

def _iter_groq(msgs):
    _req_groq()
    try: stream = _groq_create(_to_groq_msgs(msgs), True, TOOL_SCHEMAS)
    except Exception as e: raise _GroqErr(_groq_msg(e)) from e
    pending = {}
    try:
        for ch in stream:
            cs = getattr(ch, "choices", None) or []
            if not cs: continue
            d = getattr(cs[0], "delta", None)
            if d is None: continue
            rsn = getattr(d, "reasoning", None)
            if rsn: yield "thinking", rsn
            c = getattr(d, "content", None)
            if c: yield "content", c
            for td in getattr(d, "tool_calls", None) or []:
                idx = getattr(td, "index", 0) or 0
                e = pending.setdefault(idx, {"id": None, "name": None, "args": []})
                tcid = getattr(td, "id", None)
                if tcid: e["id"] = tcid
                fn = getattr(td, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None): e["name"] = fn.name
                    fa = getattr(fn, "arguments", None)
                    if fa: e["args"].append(fa)
    except _GroqErr: raise
    except Exception as e: raise _GroqErr(_groq_msg(e)) from e
    for idx in sorted(pending):
        e = pending[idx]
        yield "tool_call", _mk_call(e["id"] or f"call_{idx}", e["name"] or "", "".join(e["args"]) or "{}")

def _groq_once(msgs):
    _req_groq()
    try: r = _groq_create(_to_groq_msgs(msgs), False, None)
    except Exception as e: raise _GroqErr(_groq_msg(e)) from e
    txt = ""
    if getattr(r, "choices", None):
        m = getattr(r.choices[0], "message", None)
        if m is not None: txt = getattr(m, "content", "") or ""
    return {"message": {"content": txt}}

def _iter_stream(msgs):
    return _iter_groq(msgs) if STATE.provider == "groq" else _iter_oll(msgs)

def _chat_once(msgs):
    if STATE.provider == "groq": return _groq_once(msgs)
    return _oll_once(msgs, tools=None) or {"message": {"content": ""}}

_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.S)
_strip_think = lambda t: _THINK_RE.sub("", t)

def stream_chat(msgs):
    th, ct, tcs = [], [], []
    in_th = done = printed = False
    def close():
        nonlocal done
        if in_th and not done:
            print(RESET if STATE.show_think else f" {GREEN}Done!{RESET}", flush=True)
            done = True
    try:
        for kind, payload in _iter_stream(msgs):
            if kind == "thinking":
                if not in_th:
                    print((f"{GRAY}Thinking:{RESET} {DIM}" if STATE.show_think else f"{GRAY}Thinking...{RESET}"),
                          end="", flush=True)
                    in_th = True
                if STATE.show_think: print(payload, end="", flush=True)
                th.append(payload)
            elif kind == "content":
                if in_th and not done: close()
                elif not in_th and not done: done = True
                ct.append(payload); print(payload, end="", flush=True); printed = True
            elif kind == "tool_call": tcs.append(payload)
    except _GroqErr as e:
        close()
        content = f"(groq error) {_hint(e)}"
        print(f"\n{RED}{content}{RESET}", flush=True)
        return "".join(th), content, [], True
    close()
    if printed: print()
    return "".join(th), "".join(ct).strip(), tcs, printed

_SUM_CACHE: dict[str, str] = {}

def summarize_call(call):
    if STATE.fast_think or not STATE.think_on:
        return f"Will run {call.get('name')} with {call.get('args')}"
    key = json.dumps({"n": call.get("name"), "a": call.get("args")}, sort_keys=True, default=str)
    if key in _SUM_CACHE: return _SUM_CACHE[key]
    payload = json.dumps({"tool": call.get("name"), "args": call.get("args") or {}},
                         indent=2, ensure_ascii=False, default=str)
    try:
        r = _chat_once([{"role": "system", "content": SUMMARY_SYS},
                        {"role": "user", "content": f"Tool call:\n{payload}"}])
        t = _strip_think(((r.get("message") or {}).get("content") or "")).strip()
        out = " ".join(t.split()) if t else f"Will run {call.get('name')} with {call.get('args')}"
    except KeyboardInterrupt:
        out = "(summary interrupted)"
    except Exception as e:
        out = f"(could not summarize: {e})"
    _SUM_CACHE[key] = out
    return out

# ---------------- Goal critic ----------------
def _eval_goal(goal, history):
    recent = history[-12:] if len(history) > 12 else history[1:]
    parts = []
    for m in recent:
        role = m.get("role", "?"); c = str(m.get("content", ""))[:1500]
        tcs = m.get("tool_calls")
        if tcs:
            names = ", ".join((t.get("function") or {}).get("name", "?") for t in tcs)
            parts.append(f"{role}: [called: {names}]\n{c}")
        else: parts.append(f"{role}: {c}")
    try:
        r = _chat_once([{"role": "system", "content": CRITIC_SYS},
                        {"role": "user", "content": f"GOAL: {goal}\n\nRECENT WORK:\n" + "\n\n".join(parts)}])
        raw = _strip_think(((r.get("message") or {}).get("content") or "")).strip()
        up = raw.upper()[:300]
        v = "DONE" if "DONE" in up else ("STUCK" if "STUCK" in up else "CONTINUE")
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        return v, (lines[1].strip() if len(lines) > 1 else "")
    except Exception as e: return "CONTINUE", f"(critic failed: {e})"

# ---------------- Confirmation ----------------
def confirm_call(call, label=""):
    print(f"{YELLOW}{label}{fmt_call(call)}{RESET}")
    print(f"{CYAN}{label}{summarize_call(call)}{RESET}")
    while True:
        try: ans = input(f"{BOLD}{label}[Y]es/[N]o? {RESET}").strip().lower()
        except (KeyboardInterrupt, EOFError): print(f"\n{CTRL_C_HINT}"); continue
        if ans in ("y", "yes"): return True
        if ans in ("n", "no"):  return False

# ---------------- Agent primitives ----------------
def _asst_msg(content, tcs):
    m = {"role": "assistant", "content": content or ""}
    if tcs: m["tool_calls"] = [t.to_openai() for t in tcs]
    return m

_tool_msg = lambda tc, c: {"role": "tool", "tool_call_id": tc.id, "tool_name": tc.name, "content": c}

def _handle_call(tc, label):
    if tc.parse_error:
        m = f"Error: could not parse arguments for '{tc.name}': {tc.parse_error}."
        print(f"{RED}{label}{m}{RESET}"); return m
    if tc.name == "shell":
        cmd = str(tc.arguments.get("command", ""))
        if (blk := _guard_shell(cmd)):
            print(f"{RED}{label}{blk}{RESET}"); return f"Error: {blk}. Try a different approach."
    args = _norm_args(tc.name, tc.arguments) if tc.name in TOOLS else tc.arguments
    call = {"name": tc.name, "args": args}
    if (err := validate_call(call)):
        print(f"{RED}{label}{err}{RESET}"); return err
    # Shell: ALWAYS confirm. Other tools: follow autonomy.
    needs_confirm = (tc.name == "shell") or (not _autonomy_ok(tc.name))
    if needs_confirm:
        if not confirm_call(call, label=label):
            print(f"{GRAY}{label}Denied.{RESET}")
            return "User DENIED this tool call. It was NOT executed."
    else:
        print(f"{MAGENTA}{label}[auto] {fmt_call(call)}{RESET}")
    try: result = TOOLS[tc.name](**args)
    except Exception as e: result = f"Error running {tc.name}: {e}"
    prev = result if len(result) <= TOOL_PREVIEW else result[:TOOL_PREVIEW] + " ..."
    print(f"{GRAY}{label}{prev}{RESET}")
    return result

# ---------------- run_agent ----------------
def run_agent(user_input, history, quiet=False):
    history[0] = {"role": "system", "content": build_system_prompt()}
    history.append({"role": "user", "content": user_input})
    loop = LoopDetector()
    for _ in range(1, MAX_STEPS + 1):
        try: _, content, tcs, printed = stream_chat(history)
        except KeyboardInterrupt: print(f"\n{CTRL_C_HINT}"); break
        except Exception as e: print(f"{RED}[error] {_hint(e)}{RESET}"); break
        if not content and not tcs:
            if not printed: print(f"{GRAY}[empty reply — stopping]{RESET}")
            break
        history.append(_asst_msg(content, tcs))
        if not tcs:
            if not printed and content: print(content)
            break
        if loop.observe(_fingerprint(tcs)):
            print(f"{YELLOW}[loop detected]{RESET} injecting reflection")
            history.append({"role": "user", "content": REFLECT}); loop.reset(); continue
        batch = len(tcs)
        for i, tc in enumerate(tcs, 1):
            lbl = f"[{i}/{batch}] " if batch > 1 else ""
            history.append(_tool_msg(tc, _handle_call(tc, lbl)))
    else:
        print(f"{RED}Hit max steps ({MAX_STEPS}); stopping this turn.{RESET}")
    if not quiet: save_session(history)

# ---------------- run_goal ----------------
def run_goal(goal, history, budget, quiet=False):
    if not quiet:
        print(f"{BOLD}{CYAN}goal:{RESET} {goal}")
        print(f"{GRAY}budget: {budget.status()}{RESET}")
    history[0] = {"role": "system", "content": build_system_prompt()}
    history.append({"role": "user", "content":
        f"GOAL: {goal}\n\nWork autonomously — do not ask for permission or clarification unless truly blocked. "
        "Search the web if you need current information. When complete, reply with the final answer."})
    loop, stuck, cont = LoopDetector(), 0, 0
    while not budget.exhausted():
        budget.steps += 1
        try: _, content, tcs, printed = stream_chat(history)
        except KeyboardInterrupt: print(f"\n{CTRL_C_HINT}"); return False
        except Exception as e: print(f"{RED}[error] {_hint(e)}{RESET}"); return False
        if not content and not tcs:
            if not printed: print(f"{GRAY}[empty reply — stopping]{RESET}")
            return False
        history.append(_asst_msg(content, tcs))
        if tcs:
            budget.tool_calls += len(tcs)
            if loop.observe(_fingerprint(tcs)):
                print(f"{YELLOW}[loop detected]{RESET} injecting reflection")
                history.append({"role": "user", "content": REFLECT}); loop.reset(); continue
            batch = len(tcs)
            for i, tc in enumerate(tcs, 1):
                lbl = f"[{i}/{batch}] " if batch > 1 else ""
                history.append(_tool_msg(tc, _handle_call(tc, lbl)))
            continue
        verdict, reason = _eval_goal(goal, history)
        print(f"{GRAY}critic: {verdict}" + (f" — {reason}" if reason else "") + f"  ({budget.status()}){RESET}")
        if verdict == "DONE":
            if not printed and content: print(content)
            print(f"{GREEN}goal complete.{RESET}")
            if not quiet: save_session(history)
            return True
        if verdict == "STUCK":
            stuck += 1
            if stuck >= 3:
                print(f"{RED}stuck 3 times; pausing goal.{RESET}")
                if not quiet: save_session(history)
                return False
            history.append({"role": "user", "content": REFLECT})
        else:
            cont += 1
            if cont >= 5:
                print(f"{RED}5 continues without progress; pausing goal.{RESET}")
                if not quiet: save_session(history)
                return False
            history.append({"role": "user", "content": "Continue toward the goal. What is the NEXT concrete step?"})
    print(f"{RED}budget exhausted ({budget.status()}); pausing goal.{RESET}")
    if not quiet: save_session(history)
    return False

# ---------------- Slash commands ----------------
def _local_models():
    if not _OLLAMA_OK: return []
    try: resp = ollama.list()
    except Exception as e: warn(f"list models failed: {e}"); return []
    entries = getattr(resp, "models", None)
    if entries is None and isinstance(resp, dict): entries = resp.get("models", [])
    out = []
    for m in entries or []:
        n = getattr(m, "model", None) or getattr(m, "name", None)
        if n is None and isinstance(m, dict): n = m.get("model") or m.get("name")
        if n: out.append(n)
    return out

def _groq_models():
    if not _GROQ_OK or not STATE.groq_api_key: return []
    try: return [m.id for m in getattr(_GroqClient(api_key=STATE.groq_api_key).models.list(), "data", []) if getattr(m, "id", None)]
    except Exception as e: warn(f"list groq models failed: {e}"); return []

def cmd_model(arg):
    arg = arg.strip()
    if not arg:
        print(f"{GRAY}provider: {CYAN}{STATE.provider}{RESET}")
        print(f"{GRAY}model   : {CYAN}{STATE.current_model()}{RESET}")
        print(f"{GRAY}usage: /model <name>{RESET}"); return
    if STATE.provider == "groq":
        avail = _groq_models()
        if avail and arg not in avail:
            print(f"{YELLOW}note:{RESET} '{arg}' not in Groq list. {GRAY}available: {', '.join(avail[:12])}{RESET}")
        STATE.set(groq_model=arg)
        print(f"{GREEN}groq model: {CYAN}{arg}{RESET}")
    else:
        avail = _local_models()
        if avail and arg not in avail and f"{arg}:latest" not in avail:
            print(f"{YELLOW}note:{RESET} '{arg}' not in local list. {GRAY}available: {', '.join(avail)}{RESET}")
        STATE.set(model=arg)
        print(f"{GREEN}ollama model: {CYAN}{arg}{RESET}")

def cmd_provider(arg):
    arg = arg.strip().lower()
    if not arg:
        print(f"{GRAY}provider: {CYAN}{STATE.provider}{RESET}")
        print(f"{GRAY}usage: /provider ollama|groq{RESET}"); return
    if arg not in ("ollama", "groq"): print(f"{YELLOW}expected ollama|groq{RESET}"); return
    if arg == "groq":
        if not _GROQ_OK: print(f"{RED}Run: pip install groq{RESET}"); return
        if not STATE.groq_api_key: print(f"{RED}No Groq key. /api <key>.{RESET}"); return
    if arg == "ollama" and not _OLLAMA_OK: print(f"{RED}Run: pip install ollama{RESET}"); return
    STATE.set(provider=arg)
    print(f"{GREEN}provider: {CYAN}{arg}{RESET} {GRAY}(model: {STATE.current_model()}){RESET}")

def cmd_auto(arg):
    arg = arg.strip().lower()
    if not arg:
        print(f"{GRAY}autonomy: {CYAN}{STATE.autonomy}{RESET} {GRAY}— {_AUTO_DESC.get(STATE.autonomy, '')}{RESET}")
        print(f"{GRAY}usage: /auto manual|smart|auto|yolo{RESET}")
        print(f"{GRAY}note: shell ALWAYS asks [Y]es/[N]o regardless of level.{RESET}"); return
    if arg not in AUTONOMY: print(f"{YELLOW}expected: {', '.join(AUTO_ORDER)}{RESET}"); return
    if arg == "yolo":
        print(f"{YELLOW}warning:{RESET} yolo auto-approves medium tools. Shell still asks. Guardrails still fire.")
        try: ans = input(f"{BOLD}Enable yolo? [y/N] {RESET}").strip().lower()
        except (KeyboardInterrupt, EOFError): print(); return
        if ans not in ("y", "yes"): print(f"{GRAY}cancelled{RESET}"); return
    STATE.set(autonomy=arg)
    print(f"{GREEN}autonomy: {CYAN}{arg}{RESET} {GRAY}({_AUTO_DESC[arg]}){RESET}")

def cmd_memory(arg):
    arg = arg.strip()
    if not arg:
        print(f"{GRAY}memory file: {CYAN}{MEMORY_PATH}{RESET}")
        t = _load_mem(); print(t or f"{GRAY}(empty){RESET}"); return
    if arg == "-r":
        try: ans = input(f"{BOLD}Clear memory? [y/N] {RESET}").strip().lower()
        except (KeyboardInterrupt, EOFError): print(); return
        if ans not in ("y", "yes"): print(f"{GRAY}cancelled{RESET}"); return
        try: _secure_write(MEMORY_PATH, _MEM_DEFAULT); _mem_invalidate(); print(f"{GREEN}memory reset{RESET}")
        except OSError as e: print(f"{RED}{e}{RESET}")
        return
    if arg == "-e":
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
        if not editor:
            for c in ("nano", "vim", "vi", "code"):
                if shutil.which(c): editor = c; break
        if not editor: print(f"{RED}no $EDITOR and no common editor{RESET}"); return
        try:
            if not MEMORY_PATH.exists(): _secure_write(MEMORY_PATH, _MEM_DEFAULT)
            subprocess.run([*shlex.split(editor), str(MEMORY_PATH)], check=False)
            _mem_invalidate()
        except Exception as e: print(f"{RED}{e}{RESET}")
        return
    print(f"{GRAY}usage: /memory | /memory -r (reset) | /memory -e (edit){RESET}")

def cmd_forget():
    try: ans = input(f"{BOLD}Delete session AND reset memory? [y/N] {RESET}").strip().lower()
    except (KeyboardInterrupt, EOFError): print(); return
    if ans not in ("y", "yes"): print(f"{GRAY}cancelled{RESET}"); return
    deleted = delete_session()
    try: _secure_write(MEMORY_PATH, _MEM_DEFAULT); _mem_invalidate()
    except OSError as e: print(f"{RED}{e}{RESET}")
    print(f"{GREEN}session {'deleted' if deleted else 'not present'}; memory reset{RESET}")

def cmd_resume(history):
    loaded = load_session()
    if loaded is None: print(f"{GRAY}no saved session{RESET}"); return history
    history.clear(); history.extend(loaded)
    print(f"{GREEN}resumed {len(history)-1} messages{RESET}")
    return history

def cmd_goal(arg, history):
    if not arg.strip(): print(f"{YELLOW}usage: /goal <description>{RESET}"); return
    run_goal(arg.strip(), history, Budget.from_state())

def cmd_schedule(arg):
    parts = arg.strip().split(maxsplit=2)
    if not parts or parts[0] == "list":
        if not _SCHEDULE: print(f"{GRAY}(no scheduled tasks){RESET}")
        for i, t in enumerate(_SCHEDULE):
            rep = f" every {t.repeat_minutes}m" if t.repeat_minutes else ""
            print(f"  [{i}] {t.when.isoformat()}{rep}  {t.prompt}")
        return
    if parts[0] == "clear": _SCHEDULE.clear(); print(f"{GREEN}cleared{RESET}"); return
    if parts[0] == "cancel":
        if len(parts) < 2 or not parts[1].isdigit(): print(f"{YELLOW}usage: /schedule cancel <i>{RESET}"); return
        i = int(parts[1])
        if 0 <= i < len(_SCHEDULE): _SCHEDULE.pop(i); print(f"{GREEN}cancelled{RESET}")
        else: print(f"{YELLOW}index out of range{RESET}")
        return
    when, repeat, prompt = None, None, ""
    if parts[0] == "in" and len(parts) >= 3:
        s = _parse_dur(parts[1])
        if s is None: print(f"{YELLOW}bad duration '{parts[1]}' (try 5m, 30s, 2h){RESET}"); return
        when = datetime.datetime.now() + datetime.timedelta(seconds=s); prompt = parts[2]
    elif parts[0] == "at" and len(parts) >= 3:
        try:
            hh, mm = parts[1].split(":")
            now = datetime.datetime.now()
            when = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
            if when <= now: when += datetime.timedelta(days=1)
        except (ValueError, IndexError): print(f"{YELLOW}usage: /schedule at HH:MM <prompt>{RESET}"); return
        prompt = parts[2]
    else:
        print(f"{GRAY}usage:\n  /schedule in <dur> <prompt>\n  /schedule at HH:MM <prompt>\n"
              f"  /schedule every <dur> <prompt>\n  /schedule list | cancel <i> | clear{RESET}")
        return
    if prompt.startswith("every "):
        rest = prompt[6:].split(maxsplit=1)
        if len(rest) == 2:
            rs = _parse_dur(rest[0])
            if rs and rs >= 60: repeat = rs // 60; prompt = rest[1]
    _SCHEDULE.append(ScheduledTask(when=when, prompt=prompt, repeat_minutes=repeat))
    print(f"{GREEN}scheduled {when.isoformat()}{f' every {repeat}m' if repeat else ''}: {prompt}{RESET}")
    print(f"{GRAY}note: scheduled tasks fire when the REPL is idle.{RESET}")

def cmd_api(arg):
    arg = arg.strip()
    if not arg:
        if not _GROQ_OK: print(f"{YELLOW}note:{RESET} 'groq' not installed. pip install groq")
        print(f"{GRAY}provider      : {CYAN}{STATE.provider}{RESET}")
        print(f"{GRAY}groq api key  : {CYAN}{_mask(STATE.groq_api_key)}{RESET}")
        print(f"{GRAY}groq model    : {CYAN}{STATE.groq_model}{RESET}")
        if STATE._key_env: print(f"{GRAY}(key from $AGENT0_GROQ_API_KEY){RESET}")
        print(f"{GRAY}usage: /api <key>{RESET}"); return
    if not _GROQ_OK: print(f"{RED}Run: pip install groq{RESET}"); return
    STATE.set_groq_key(arg)
    if STATE.provider != "groq": STATE.set(provider="groq")
    print(f"{GREEN}Groq key saved: {CYAN}{_mask(arg)}{RESET}")
    print(f"{GRAY}provider: groq  model: {STATE.groq_model}{RESET}")

def cmd_think(arg):
    parts = arg.strip().split()
    if not parts:
        print(f"{GRAY}thinking       : {CYAN}{'on' if STATE.think_on else 'off'}{RESET}")
        print(f"{GRAY}show thinking  : {CYAN}{'on' if STATE.show_think else 'off'}{RESET}")
        print(f"{GRAY}fast thinking  : {CYAN}{'on' if STATE.fast_think else 'off'}{RESET}")
        print(f"{GRAY}usage: /think on|off | -s on|off | -f on|off{RESET}"); return
    if len(parts) == 1:
        v = parts[0].lower()
        if v in ("on", "off"): STATE.set(think_on=(v == "on")); print(f"{GREEN}thinking {v}{RESET}"); return
    if len(parts) == 2:
        flag, v = parts[0], parts[1].lower()
        if v not in ("on", "off"): print(f"{YELLOW}expected on|off{RESET}"); return
        b = (v == "on")
        if flag == "-s": STATE.set(show_think=b); print(f"{GREEN}show {v}{RESET}"); return
        if flag == "-f": STATE.set(fast_think=b); print(f"{GREEN}fast {v}{RESET}"); return
    print(f"{GRAY}usage: /think on|off | -s on|off | -f on|off{RESET}")

def cmd_ollama(arg):
    parts = arg.strip().split()
    if not parts:
        print(f"{GRAY}keep_alive : {CYAN}{STATE.keep_alive}{RESET}  {GRAY}(how long to keep model in RAM){RESET}")
        print(f"{GRAY}num_ctx    : {CYAN}{STATE.num_ctx}{RESET}  {GRAY}(context window; lower = faster){RESET}")
        print(f"{GRAY}num_predict: {CYAN}{STATE.num_predict}{RESET}  {GRAY}(max new tokens; -1 = unlimited){RESET}")
        print(f"{GRAY}usage: /ollama keep_alive 30m | num_ctx 4096 | num_predict 512 | -r{RESET}")
        return
    if parts[0] == "-r":
        STATE.set(keep_alive="30m", num_ctx=4096, num_predict=-1)
        print(f"{GREEN}ollama options reset{RESET}"); return
    if len(parts) != 2:
        print(f"{YELLOW}usage: /ollama keep_alive 30m | num_ctx 4096 | num_predict 512{RESET}"); return
    k, v = parts[0].lower(), parts[1]
    if k == "keep_alive":
        STATE.set(keep_alive=v); print(f"{GREEN}keep_alive → {CYAN}{v}{RESET}")
    elif k == "num_ctx":
        try: n = int(v)
        except ValueError: print(f"{YELLOW}expected integer{RESET}"); return
        STATE.set(num_ctx=max(512, min(131072, n)))
        print(f"{GREEN}num_ctx → {CYAN}{STATE.num_ctx}{RESET}")
    elif k == "num_predict":
        try: n = int(v)
        except ValueError: print(f"{YELLOW}expected integer{RESET}"); return
        STATE.set(num_predict=n); print(f"{GREEN}num_predict → {CYAN}{n}{RESET}")
    else:
        print(f"{YELLOW}unknown: {k}{RESET}")

def show_help():
    print(f"""{BOLD}Commands{RESET}
  {CYAN}/goal{RESET} <text>       run autonomously toward a goal
  {CYAN}/resume{RESET}            reload the last saved session
  {CYAN}/schedule{RESET} [...]    one-shot or repeating self-triggered tasks
  {CYAN}/forget{RESET}            delete session file and reset memory
  {CYAN}/model{RESET} <name>      switch model
  {CYAN}/provider{RESET} o|g      switch backend (ollama|groq)
  {CYAN}/api{RESET} <key>         set the Groq API key (and switch to groq)
  {CYAN}/auto{RESET} <level>      manual|smart|auto|yolo  (shell always asks)
  {CYAN}/memory{RESET} [-r|-e]    view / reset / edit memory file
  {CYAN}/think{RESET} [...]       thinking settings (ollama only)
  {CYAN}/ollama{RESET} [...]      tune keep_alive / num_ctx / num_predict
  {CYAN}/reset{RESET}             clear in-memory history
  {CYAN}/help{RESET}              show this help
  {CYAN}/bye{RESET}               quit (Ctrl-C / Ctrl-D do nothing)

{BOLD}Tools{RESET}
  shell          run a shell command           {YELLOW}(ALWAYS asks [Y]es/[N]o){RESET}
  web_search     DuckDuckGo web search          {GREEN}(auto at 'auto'){RESET}
  fetch_url      HTTP GET a URL                 {GREEN}(auto at 'auto'){RESET}
  memory_append  append a bullet to memory      {GREEN}(auto at 'auto'){RESET}
  memory_write   replace memory.md entirely     {GREEN}(auto at 'auto'){RESET}
""")

# ---------------- Entrypoint ----------------
_new_history = lambda: [{"role": "system", "content": build_system_prompt()}]

def main():
    if not sys.stdin.isatty():
        print("This script needs an interactive terminal.", file=sys.stderr); sys.exit(1)
    _mem_init()
    print(WELCOME)
    print(f"{GRAY}autonomy: {STATE.autonomy}    provider: {STATE.provider}    "
          f"model: {STATE.current_model()}    search: duckduckgo{RESET}")
    print(f"{GRAY}memory:   {MEMORY_PATH}{RESET}")
    print(f"{GRAY}session:  {SESSION_PATH}{RESET}")
    print(f"{GRAY}note:     shell always asks [Y]es/[N]o{RESET}\n")
    history = _new_history()
    while True:
        _check_schedule(history)
        try: user_input = input(f"{BOLD}>>> {RESET}")
        except (KeyboardInterrupt, EOFError): print(f"\n{CTRL_C_HINT}"); continue
        s = user_input.strip()
        if not s: continue
        if s.startswith("/"):
            parts = s.split(maxsplit=1); cmd = parts[0].lower(); arg = parts[1] if len(parts) > 1 else ""
            if cmd in ("/bye", "/exit", "/quit"): break
            if cmd == "/help":     show_help(); continue
            if cmd == "/reset":    history[:] = _new_history(); print(f"{GRAY}history cleared (memory kept){RESET}"); continue
            if cmd == "/forget":   cmd_forget(); continue
            if cmd == "/resume":   history = cmd_resume(history); continue
            if cmd == "/goal":
                try: cmd_goal(arg, history)
                except KeyboardInterrupt: print(f"\n{CTRL_C_HINT}")
                continue
            if cmd == "/schedule": cmd_schedule(arg); continue
            if cmd == "/model":    cmd_model(arg); continue
            if cmd == "/provider": cmd_provider(arg); continue
            if cmd == "/api":      cmd_api(arg); continue
            if cmd == "/auto":     cmd_auto(arg); continue
            if cmd == "/memory":   cmd_memory(arg); continue
            if cmd == "/think":    cmd_think(arg); continue
            if cmd == "/ollama":   cmd_ollama(arg); continue
            print(f"{YELLOW}unknown command:{RESET} {cmd}  {GRAY}(try /help){RESET}"); continue
        try: run_agent(s, history)
        except KeyboardInterrupt: print(f"\n{CTRL_C_HINT}")
        print()
    print(f"{GRAY}bye.{RESET}")

if __name__ == "__main__":
    main()