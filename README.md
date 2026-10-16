# Agent-0

A CLI AI agent that runs on your machine. Talks to Ollama locally or Groq over the network, has five tools, and won't run a shell command without your say-so.

The name and premise are lifted from the Agent-0 in AI-2027. This is the version you can actually install.

## What you get

You type at it, it thinks, it calls tools. It keeps notes in a markdown file, saves the session as JSON, and can chew on a goal in the background while you wait. Shell is the only genuinely scary thing, and it asks every single time.

It's a single Python file. No daemon, no web UI, no build step.

## Setup

```
bash app.sh
```

Python 3.10+. It will refuse to run if stdin isn't a TTY, because half the interaction is `input()` prompts.

```
./__main__.py
```

First launch writes defaults into `db/local/`. Ollama is the default backend and `deepseek-r1:1.5b` is the default model. That model is tiny and not very bright — it will fumble tool calls. Swap it as soon as you can with `/model` or by editing the config.

## Backends

**Ollama.** Local, free, private. Whatever you've got pulled. If `ollama list` shows it, `/model` will take it.

**Groq.** Remote, fast, needs a key. From the REPL:

```
/api gsk_...
```

That saves the key with mode 600 and flips you to Groq in one go. `AGENT0_GROQ_API_KEY` in your environment also works and wins over the stored key.

## The five tools

That's the whole list. It's not extensible at runtime and it's not meant to be.

**shell** — runs a command through `/bin/sh`. Combined stdout+stderr, exit code appended if nonzero, 60-second timeout with process group kill. There are guardrails for `rm -rf /`, fork bombs, `dd of=/dev/...`, `mkfs`, force-pushing to main, `curl | bash`, and a few others. They're regexes; they'll miss things. Shell *always* requires `[Y]es/[N]o` — no autonomy level changes that.

**web_search** — DuckDuckGo. Tries the HTML endpoint first, falls back to Lite if the layout's been mangled. Returns titles, URLs, snippets.

**fetch_url** — plain GET, ~500 KB cap, http/https only.

**memory_append** — adds a bullet to a section in `memory.md`. Creates the section if it doesn't exist.

**memory_write** — replaces the file wholesale. You almost never want this.

No file-reading or file-writing tool. If you want the agent to look at a file, it uses `shell` with `cat`, `head`, `grep`, whatever.

## Autonomy

Set with `/auto`. Four levels:

```
manual   asks for everything
smart    asks for everything (see note)
auto     auto-approves web_search, fetch_url, memory_*
yolo     same as auto in practice
```

The intended tiers were safe / medium / dangerous, but `_autonomy_ok` only ever checks the medium flag, and shell is hardcoded to ask. So `smart` and `manual` end up behaving identically, and `yolo` doesn't unlock anything `auto` doesn't already have. If you set yolo you get a warning and a confirmation prompt; after that, no difference.

## Slash commands

`/help` lists them. The ones worth knowing:

`/goal <text>` hands the agent a task and lets it run. Separate budgets for steps, tool calls, and wall time. Between turns, a critic call decides whether it's DONE, CONTINUE, or STUCK and reacts accordingly. Three STUCK verdicts or five CONTINUEs in a row and it pauses.

`/schedule in 30m check the build` queues a task. Also takes `at HH:MM`, `every 10m`, plus `list`, `cancel <n>`, `clear`. Tasks fire when the REPL is sitting idle — close the terminal and they're gone.

`/memory` prints the file. `-e` opens it in `$EDITOR` (or nano/vim/vi/code, whichever exists). `-r` resets to the template.

`/resume` reloads the last saved session. `/forget` deletes it and resets memory, after asking.

`/think` toggles reasoning for Ollama. `-s on` shows the thinking stream, `-f on` asks for faster reasoning.

`/api`, `/model`, `/provider` do what you'd expect. `/reset` clears history but keeps memory.

Ctrl-C is disabled at the prompt on purpose. It prints a hint and keeps going. Type `/bye`. Inside a running goal, Ctrl-C still works as an interrupt.

## Files

```
db/local/.config.json   settings, mode 600
db/local/memory.md      durable facts, injected into every prompt
db/local/session.json   last 60 messages
```

Override any of these with `AGENT0_CONFIG`, `AGENT0_MEMORY`, `AGENT0_SESSION`. `AGENT0_MODEL` changes the default Ollama model without touching config.

There's a check that stops the agent from reading its own session/config via shell. It's a policy note in the system prompt, not an actual filesystem lock — the guardrails and the protected-path check catch the obvious cases, but if you ask nicely it might still do it. Consider it a speed bump.

## Memory format

Markdown, three headings by default:

```
# Agent-0 Memory

## User
- prefers terse answers

## Projects
- this repo: agent0

## Notes
```

`memory_append(section="User", entry="...")` finds or creates the heading and inserts a `- ` bullet under it. Section names get their leading `#`s stripped, so `"## User"` and `"User"` land in the same place. Newlines in entries get collapsed to spaces — one fact per bullet, please.

Injected into the system prompt up to 6000 chars. File itself caps at 200 KB.

## Things it does that you didn't ask about

Loop detection. If the same set of tool calls comes up four turns in a row, it injects a reflection prompt and makes the model re-plan. Helps with small models that get stuck polishing the same screw.

Tool call summaries. Before a confirm prompt, it does a one-shot LLM call to describe the call in plain English. Cache-friendly, no-ops in fast mode. When the summary model is the same tiny model that's proposing the call, the summary is often useless but it's usually better than raw JSON.

Goal critic. Each turn without a tool call gets graded by a separate call that only sees the goal and the recent history. Cheap on Groq, slow on Ollama.

## Rough edges

Small models call tools badly. The 1.5B default will invent arguments, use the wrong schema, and occasionally call a tool that doesn't exist. This is not the script's fault and the script can't fix it.

Groq sometimes rejects `tool_choice: auto` for no clear reason. You'll see a red message and a hint to retry, and retrying usually works.

The `safe` autonomy tier is dead code. See above.

Session files are not encrypted or signed. They're just JSON on disk with mode 600. If someone has read access to your user, they can read your session.

Scheduled tasks don't survive restarts and don't run in the background. They're a REPL-idle thing, not cron.

Web search parsing is regex-based against DuckDuckGo's HTML. When they redesign the page, this breaks. The fallback to the Lite endpoint usually buys another few months.

## Why this exists

Because AI-2027's Agent-0 was a good hook and I wanted a working version. It doesn't try to be the version in the story. It's a small tool that sits in your terminal and asks before it does anything that matters.