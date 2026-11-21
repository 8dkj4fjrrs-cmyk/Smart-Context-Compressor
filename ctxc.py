#!/usr/bin/env python3
"""Smart Context Compressor (ctxc) - a small context-compressing proxy for very long coding-agent sessions.

Point an agent's base URL at this proxy. Every request is rewritten so the model
sees at most `--window` tokens, however long the client-side transcript has grown:

    [ system + tools ]            untouched
    [ summaries ]                 append-only notes replacing the oldest messages
    [ cold zone ]                 real messages, tool output digested / deduplicated
    [ hot zone ]                  the most recent messages, verbatim

Compression is automatic (the model never spends a turn on it) and happens in
epochs, so between epochs the prompt prefix is byte-stable and stays cacheable.
Responses are streamed back untouched. Everything removed is kept in SQLite and
can be fetched by the agent from its shell: `ctxc recall REF`, `ctxc search WORDS`.

One file, standard library only.   ctxc run AGENT | on | off | serve | recall | search | stats | link
"""
import argparse
import collections
import gzip
import hashlib
import http.client
import json
import math
import os
import plistlib
import re
import shutil
import sqlite3
import ssl
import subprocess
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

VERSION = "0.9.0"
FANIN = 4                       # summaries merged into one when the summary zone is full
INDEX_CAP = 300 * 1024          # bytes of each stored output that are full-text indexed
SIGNAL = re.compile(
    r"error|fail|fatal|panic|exception|traceback|warning|assert|denied|not found|"
    r"undefined|cannot|unable|segfault|timed? ?out|\bE\d{3,4}\b", re.I)
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")
REF = re.compile(r"\br:[0-9a-f]{10}\b")
INTRO = ("[ctxc] Earlier parts of this session were compressed to fit the context window. The notes below "
         "replace messages you already processed; treat them as your own memory of the work. Nothing is lost: "
         "for exact details behind any [r:…] or [s:…] reference run `{cli} recall REF` (add `--grep REGEX` or "
         "`--lines A:B`) or `{cli} search WORDS` in a shell.")
SUMMARISE = """You are writing the working memory of an AI coding agent in the middle of a very long session. \
The <slice> below is about to be removed from the agent's context. Your note replaces it: afterwards the agent \
knows ONLY what you write, plus <earlier_notes> and the messages in <what_happened_next>, which it keeps.

Use <earlier_notes> to avoid repeating what is already recorded and to notice what the slice changes or overturns. \
Use <what_happened_next> only as hindsight, to judge which details of the slice turned out to matter; do not \
summarise it.

Write these sections, omitting any that would be empty:
REQUIREMENTS - every instruction, constraint or preference the user stated in the slice, in the user's own words.
DECISIONS - what was decided and WHY, including rejected alternatives worth not retrying.
CHANGES - one line per file or component: what it now does or how it changed (names, signatures, values).
FACTS - exact things that are costly to rediscover: commands that work, ports, paths, versions, config values, \
identifiers, test names, error messages and the fix that resolved each.
OPEN - unfinished work, known bugs, unanswered questions, and what the agent was about to do next.

Rules: be specific and literal (exact names, numbers and strings, never "some file" or "an error"). Keep the \
[r:...] reference next to any fact taken from tool output. State the outcome, not the journey; drop dead ends \
unless knowing they failed saves future work. If the slice contradicts an earlier note, say so explicitly. \
At most {words} words. No preamble, no closing remarks.

<earlier_notes>
{notes}
</earlier_notes>

<slice>
{body}
</slice>

<what_happened_next>
{after}
</what_happened_next>"""
MERGE = """These are consecutive working-memory notes of an AI coding agent, oldest first, from one very long \
session. Merge them into ONE note of at most {words} words with the same sections (REQUIREMENTS, DECISIONS, \
CHANGES, FACTS, OPEN).

REQUIREMENTS and DECISIONS: keep every one that is still in force, in its original wording; when a later note \
overrides an earlier one keep only the later and add "(replaces: ...)" in a few words.
CHANGES: keep only the final state of each file or component.
FACTS: keep exact values; drop ones made obsolete by later notes.
OPEN: keep only what is still open at the end of the last note.
Keep [r:...] references. No preamble.

{body}"""

# ---------------------------------------------------------------- text helpers

def toks(s):
    """Conservative token estimate (code, JSON and logs run ~3.5 chars per token)."""
    return math.ceil(len(s) / 3.5)


def size(o):
    """Approximate serialized size in chars; inline images count as a fixed cost."""
    if isinstance(o, str):
        return len(o) + 2
    if isinstance(o, dict):
        if isinstance(o.get("data"), str) and len(o["data"]) > 2000 and o.get("type") == "base64":
            return 5000
        if isinstance(o.get("image_url"), (str, dict)) and size(o["image_url"]) > 2000:
            return 5000
        return 2 + sum(len(k) + 4 + size(v) for k, v in o.items())
    if isinstance(o, list):
        return 2 + sum(size(v) + 1 for v in o)
    return 5


def tok(o):
    return math.ceil(size(o) / 3.5)


def clip(s, max_tokens, note=" …[clipped]"):
    s = (s or "").strip()
    limit = int(max_tokens * 3.5)
    return s if len(s) <= limit else s[:max(0, limit - len(note))].rstrip() + note


def one_line(s, n=90):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n - 1] + "…"


def strip_cc(o):
    if isinstance(o, dict):
        return {k: strip_cc(v) for k, v in o.items() if k != "cache_control"}
    if isinstance(o, list):
        return [strip_cc(v) for v in o]
    return o


def mhash(m):
    """Identity of a message, ignoring cache markers that clients move around."""
    return hashlib.sha1(json.dumps(strip_cc(m), sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def _lines(text):
    parts = text.split("\n")
    return parts[:-1] if parts and parts[-1] == "" else parts


def clean_lines(text):
    """Strip colour codes, resolve \\r progress bars, collapse repeated lines."""
    out, prev, reps = [], None, 0
    for n, raw in enumerate(_lines(text), 1):
        line = ANSI.sub("", raw.split("\r")[-1]).rstrip()
        if line == prev:
            reps += 1
            continue
        if reps:
            out.append((None, "    … previous line repeated %d more times" % reps))
        out.append((n, line))
        prev, reps = line, 0
    if reps:
        out.append((None, "    … previous line repeated %d more times" % reps))
    return out


def digest(text, budget):
    """Head + notable middle lines + tail, with original line numbers, inside a token budget."""
    lines = clean_lines(text)
    fmt = lambda n, l: (("%6d| " % n) if n else "") + (l if len(l) <= 240 else l[:240] + "…")
    whole = "\n".join(l if len(l) <= 240 else l[:240] + "…" for _, l in lines)
    if toks(whole) <= budget:
        return whole
    scale = max(0.15, min(4.0, budget / 600.0))
    head_n, tail_n, sig_n = (max(2, int(x * scale)) for x in (12, 25, 20))
    while True:
        head, tail = lines[:head_n], lines[-tail_n:]
        shown = {n for n, _ in head + tail}
        sig, seen = [], set()
        for n, l in lines[head_n:len(lines) - tail_n]:
            key = re.sub(r"\d+", "#", l)
            if n and n not in shown and key not in seen and SIGNAL.search(l):
                seen.add(key)
                sig.append((n, l))
        parts = [fmt(n, l) for n, l in head]
        if sig:
            parts.append("   … notable lines%s:" % (" (+%d more)" % (len(sig) - sig_n) if len(sig) > sig_n else ""))
            parts += [fmt(n, l) for n, l in sig[:sig_n]]
        parts.append("   …")
        parts += [fmt(n, l) for n, l in tail]
        body = "\n".join(parts)
        if toks(body) <= budget or (head_n <= 1 and tail_n <= 2 and sig_n <= 1):
            return clip(body, budget)
        head_n, tail_n, sig_n = max(1, head_n * 2 // 3), max(2, tail_n * 2 // 3), max(1, sig_n * 2 // 3)


def flat(content):
    """(text, has_non_text) of a content field that may be a string or a list of typed parts."""
    if isinstance(content, str):
        return content, False
    out, img = [], False
    for p in content if isinstance(content, list) else []:
        if isinstance(p, dict) and isinstance(p.get("text"), str):
            out.append(p["text"])
        elif isinstance(p, dict):
            img = True
            out.append("[%s]" % p.get("type", "attachment"))
    return "\n".join(out), img


def _loads(s):
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except ValueError:
        return s


# ------------------------------------------------------------- wire adapters
# Each adapter answers the same few questions for one API shape: which items can
# be compressed, where a zone may start without splitting a tool call from its
# result, what a message contains, and how to rewrite it.

class Anthropic:
    name = "anthropic"

    def msgs(self, body):
        m = body.get("messages")
        return m if isinstance(m, list) and m else None

    def put(self, body, msgs):
        body["messages"] = msgs

    def head(self, msgs):
        return 0

    def overhead(self, body, head_msgs):
        return tok(body.get("system") or "") + tok(body.get("tools") or [])

    def boundaries(self, msgs, start):
        return {i for i in range(start + 1, len(msgs)) if msgs[i].get("role") == "assistant"}

    def hint(self, body, headers):
        return str((body.get("metadata") or {}).get("user_id") or "")

    def parts(self, m):
        role, c = m.get("role"), m.get("content")
        if isinstance(c, str):
            yield ("text", role, c)
            return
        for b in c or []:
            t = b.get("type")
            if t == "text":
                yield ("text", role, b.get("text", ""))
            elif t == "tool_use":
                yield ("call", b.get("id"), b.get("name", "?"), b.get("input"))
            elif t == "tool_result":
                yield ("result", b.get("tool_use_id"), flat(b.get("content") or "")[0])
            elif t not in ("thinking", "redacted_thinking"):
                yield ("other", t)

    def trim(self, m, rfn, afn, cold):
        c = m.get("content")
        if not isinstance(c, list):
            return m
        out, changed = [], False
        for b in c:
            t, nb = b.get("type"), b
            if t == "tool_result":
                text, img = flat(b.get("content") or "")
                new = rfn(b.get("tool_use_id"), text, img, b.get("content"))
                if new is not None:
                    nb = dict(b, content=new)
            elif t == "tool_use" and afn and isinstance(b.get("input"), dict):
                new = afn(b["input"])
                if new is not None:
                    nb = dict(b, input=new)
            elif t in ("image", "document") and cold:
                nb = {"type": "text", "text": cold(b, t)}
            changed |= nb is not b
            out.append(nb)
        return dict(m, content=out) if changed else m

    def summary(self, texts):
        return {"role": "user", "content": [{"type": "text", "text": t} for t in texts]}

    def side(self, body, prompt, model):
        out = {"model": model or body.get("model"), "max_tokens": 4096,
               "messages": [{"role": "user", "content": prompt}]}
        for k in ("system", "metadata"):
            if body.get(k):
                out[k] = body[k]
        return out

    def side_text(self, j):
        if j.get("stop_reason") == "max_tokens":
            raise ValueError("the note was cut off before it finished")
        return "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")


class Chat:
    name = "chat"

    def msgs(self, body):
        m = body.get("messages")
        return m if isinstance(m, list) and m else None

    def put(self, body, msgs):
        body["messages"] = msgs

    def head(self, msgs):
        i = 0
        while i < len(msgs) and msgs[i].get("role") in ("system", "developer"):
            i += 1
        return i

    def overhead(self, body, head_msgs):
        return tok(head_msgs) + tok(body.get("tools") or [])

    def boundaries(self, msgs, start):
        return {i for i in range(start + 1, len(msgs)) if msgs[i].get("role") == "assistant"}

    def hint(self, body, headers):
        return str(body.get("prompt_cache_key") or headers.get("session_id") or headers.get("x-session-id") or "")

    def parts(self, m):
        role = m.get("role")
        text, _ = flat(m.get("content") or "")
        if role == "tool":
            yield ("result", m.get("tool_call_id"), text)
            return
        if text:
            yield ("text", role, text)
        for tc in m.get("tool_calls") or []:
            f = tc.get("function") or {}
            yield ("call", tc.get("id"), f.get("name", "?"), _loads(f.get("arguments")))

    def trim(self, m, rfn, afn, cold):
        if m.get("role") == "tool":
            text, img = flat(m.get("content") or "")
            new = rfn(m.get("tool_call_id"), text, img, m.get("content"))
            return m if new is None else dict(m, content=new)
        out = m
        if afn and m.get("tool_calls"):
            calls, changed = [], False
            for tc in m["tool_calls"]:
                f = tc.get("function") or {}
                args = _loads(f.get("arguments"))
                new = afn(args) if isinstance(args, dict) else None
                if new is not None:
                    tc, changed = dict(tc, function=dict(f, arguments=json.dumps(new, ensure_ascii=False))), True
                calls.append(tc)
            if changed:
                out = dict(out, tool_calls=calls)
        if cold and isinstance(m.get("content"), list) and any(p.get("type") != "text" for p in m["content"]):
            out = dict(out, content=[p if p.get("type") == "text" else
                                     {"type": "text", "text": cold(p, p.get("type"))}
                                     for p in m["content"]])
        return out

    def summary(self, texts):
        return {"role": "user", "content": "\n\n".join(texts)}

    def side(self, body, prompt, model):
        msgs = body.get("messages") or []
        return {"model": model or body.get("model"),
                "messages": msgs[:self.head(msgs)] + [{"role": "user", "content": prompt}]}

    def side_text(self, j):
        choice = (j.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            raise ValueError("the note was cut off before it finished")
        return (choice.get("message") or {}).get("content") or ""


class Responses:
    name = "responses"
    CALLS = ("function_call", "custom_tool_call", "local_shell_call", "computer_call")

    def msgs(self, body):
        m = body.get("input")
        return m if isinstance(m, list) and m and not body.get("previous_response_id") else None

    def put(self, body, msgs):
        body["input"] = msgs

    def head(self, msgs):
        i = 0
        while (i < len(msgs) and msgs[i].get("type", "message") == "message"
               and msgs[i].get("role") in ("system", "developer")):
            i += 1
        return i

    def overhead(self, body, head_msgs):
        return tok(body.get("instructions") or "") + tok(body.get("tools") or []) + tok(head_msgs)

    def boundaries(self, msgs, start):
        ok, open_calls = set(), set()
        for i in range(start, len(msgs)):
            t = msgs[i].get("type", "message")
            if i > start and not open_calls and msgs[i - 1].get("type") != "reasoning" and not t.endswith("_output"):
                ok.add(i)
            if t in self.CALLS:
                open_calls.add(msgs[i].get("call_id") or msgs[i].get("id"))
            elif t.endswith("_output"):
                open_calls.discard(msgs[i].get("call_id"))
        return ok

    def hint(self, body, headers):
        return str(body.get("prompt_cache_key") or headers.get("session_id") or headers.get("x-session-id") or "")

    def parts(self, m):
        t = m.get("type", "message")
        if t == "message":
            yield ("text", m.get("role"), flat(m.get("content") or "")[0])
        elif t in self.CALLS:
            args = m.get("arguments", m.get("input", m.get("action")))
            yield ("call", m.get("call_id") or m.get("id"), m.get("name") or t, _loads(args))
        elif t.endswith("_output"):
            yield ("result", m.get("call_id"), flat(m.get("output") or "")[0])
        elif t != "reasoning":
            yield ("other", t)

    def trim(self, m, rfn, afn, cold):
        t = m.get("type", "message")
        if t.endswith("_output"):
            text, img = flat(m.get("output") or "")
            new = rfn(m.get("call_id"), text, img, m.get("output"))
            return m if new is None else dict(m, output=new)
        if afn and t == "function_call":
            args = _loads(m.get("arguments"))
            new = afn(args) if isinstance(args, dict) else None
            return m if new is None else dict(m, arguments=json.dumps(new, ensure_ascii=False))
        if afn and t == "custom_tool_call" and isinstance(m.get("input"), str):
            new = afn({"input": m["input"]})
            return m if new is None else dict(m, input=new["input"])
        if cold and t == "message" and isinstance(m.get("content"), list) and any(
                "text" not in str(p.get("type")) for p in m["content"]):
            kind = "output_text" if m.get("role") == "assistant" else "input_text"
            return dict(m, content=[p if "text" in str(p.get("type")) else
                                    {"type": kind, "text": cold(p, p.get("type"))} for p in m["content"]])
        return m

    def summary(self, texts):
        return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": t} for t in texts]}

    def side(self, body, prompt, model):
        out = {"model": model or body.get("model"), "store": False,
               "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}]}
        if body.get("instructions"):
            out["instructions"] = body["instructions"]
        return out

    def side_text(self, j):
        if j.get("status") == "incomplete":
            raise ValueError("the note was cut off before it finished")
        return "".join(p.get("text", "") for it in j.get("output", []) if it.get("type") == "message"
                       for p in it.get("content", []) if isinstance(p, dict))


def adapter_for(path):
    p = path.split("?")[0].rstrip("/")
    if p.endswith("/messages") or p.endswith("/messages/count_tokens"):
        return Anthropic()
    if p.endswith("/chat/completions"):
        return Chat()
    if p.endswith("/responses"):
        return Responses()
    return None


# ----------------------------------------------------------------------- store

class Store:
    """SQLite archive of everything the proxy removed from the prompt."""

    def __init__(self, directory=None, readonly=False):
        self.dir = Path(directory or os.environ.get("CTXC_DIR") or Path.home() / ".ctxc")
        self.lock = threading.RLock()
        self.known, self.known_since = set(), time.time()
        path = self.dir / "ctx.db"
        if readonly:
            self.db = sqlite3.connect("file:%s?mode=ro" % path, uri=True, timeout=10, check_same_thread=False)
        else:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
            # rollback journal (not WAL) so a sandboxed agent can read the file without write access
            self.db.execute("PRAGMA synchronous=NORMAL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS blobs(ref TEXT PRIMARY KEY, ts REAL, n_lines INTEGER, n_bytes INTEGER, data BLOB);
                CREATE TABLE IF NOT EXISTS folds(id INTEGER PRIMARY KEY, sid TEXT, a INTEGER, b INTEGER, ts REAL,
                                                 summary TEXT, data BLOB, kids TEXT);
                CREATE TABLE IF NOT EXISTS sessions(sid TEXT PRIMARY KEY, state TEXT, updated REAL);
            """)
            try:
                self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(content, ref UNINDEXED)")
            except sqlite3.OperationalError:
                self.db.execute("CREATE TABLE IF NOT EXISTS fts(content TEXT, ref TEXT)")
            self.db.commit()
        self.db.row_factory = sqlite3.Row
        sql = self.db.execute("SELECT sql FROM sqlite_master WHERE name='fts'").fetchone()
        self.has_fts = bool(sql and "fts5" in (sql[0] or "").lower())

    def put(self, text, index=True):
        """Archive a tool output (content-addressed, so repeats cost nothing). Returns its ref."""
        ref = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:10]
        if time.time() - self.known_since > 600:        # `known` is only a short-lived cache: another process
            self.known, self.known_since = set(), time.time()       # (ctxc gc) may have removed entries
        if ref in self.known:
            return ref
        with self.lock:
            if self.db.execute("SELECT 1 FROM blobs WHERE ref=?", (ref,)).fetchone():
                self.db.execute("UPDATE blobs SET ts=? WHERE ref=?", (time.time(), ref))    # still in use: keep it
                self.db.commit()
            else:
                raw = text.encode("utf-8", "replace")
                self.db.execute("INSERT INTO blobs VALUES(?,?,?,?,?)",
                                (ref, time.time(), len(_lines(text)), len(raw), zlib.compress(raw, 6)))
                idx = _lines(text[:INDEX_CAP]) if index else []
                self.db.executemany("INSERT INTO fts(content, ref) VALUES(?,?)", [
                    ("\n".join(idx[i:i + 40]), "r:%s:%d" % (ref, i + 1)) for i in range(0, len(idx), 40)])
                self.db.commit()
            self.known.add(ref)
        return ref

    def add_fold(self, sid, a, b, summary, text, kids=None):
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO folds(sid,a,b,ts,summary,data,kids) VALUES(?,?,?,?,?,?,?)",
                (sid, a, b, time.time(), summary, zlib.compress(text.encode("utf-8", "replace"), 6),
                 json.dumps(kids or [])))
            fid = cur.lastrowid
            idx = _lines(text[:INDEX_CAP * 2])
            self.db.executemany("INSERT INTO fts(content, ref) VALUES(?,?)", [
                ("\n".join(idx[i:i + 40]), "s:%d:%d" % (fid, i + 1)) for i in range(0, len(idx), 40)])
            self.db.commit()
        return fid

    def load(self, sid):
        with self.lock:
            r = self.db.execute("SELECT state FROM sessions WHERE sid=?", (sid,)).fetchone()
        return json.loads(r["state"]) if r else None

    def save(self, sid, st):
        with self.lock:
            self.db.execute("INSERT INTO sessions VALUES(?,?,?) ON CONFLICT(sid) DO UPDATE SET "
                            "state=excluded.state, updated=excluded.updated", (sid, json.dumps(st), time.time()))
            self.db.commit()

    # -- reading back (the CLI the agent runs)
    def _fold_text(self, fid):
        r = self.db.execute("SELECT data, kids FROM folds WHERE id=?", (fid,)).fetchone()
        if not r:
            return ""
        kids = json.loads(r["kids"] or "[]")
        if kids:        # a merged note: its detail is everything beneath it
            return "\n".join(self._fold_text(k) for k in kids)
        return zlib.decompress(r["data"]).decode("utf-8", "replace")

    def recall(self, ref, grep=None, lines=None, budget=1500):
        kind, _, rest = ref.strip().strip("[]").partition(":")
        key = rest.split(":")[0]
        if kind == "r":
            r = self.db.execute("SELECT * FROM blobs WHERE ref=?", (key,)).fetchone()
            if not r:
                return "No stored output %s (it may have been removed by clean-up)." % ref
            title, text = "r:%s (%d lines)" % (key, r["n_lines"]), zlib.decompress(r["data"]).decode("utf-8", "replace")
        elif kind == "s" and key.isdigit():
            r = self.db.execute("SELECT * FROM folds WHERE id=?", (int(key),)).fetchone()
            if not r:
                return "No summary %s (it may have been removed by clean-up)." % ref
            kids = json.loads(r["kids"] or "[]")
            title = "s:%s (messages %d–%d%s)" % (key, r["a"], r["b"],
                                                "; merges " + ", ".join("s:%d" % k for k in kids) if kids else "")
            text = self._fold_text(int(key))
        else:
            return "Bad reference %r. Use r:HASH or s:NUMBER." % ref
        all_lines = _lines(text)
        fmt = lambda i: "%6d| %s" % (i + 1, ANSI.sub("", all_lines[i].split("\r")[-1])[:400])
        if grep:
            try:
                rx = re.compile(grep, re.I)
            except re.error as e:
                return "Bad regex: %s" % e
            hits = [i for i, l in enumerate(all_lines) if rx.search(l)]
            keep = sorted({j for i in hits for j in range(max(0, i - 2), min(len(all_lines), i + 3))})
            out, prev = ["%d matching lines in %s" % (len(hits), title)], None
            for j in keep:
                if prev is not None and j != prev + 1:
                    out.append("     …")
                out.append(fmt(j))
                prev = j
        elif lines:
            m = re.match(r"^\s*(\d+)\s*[:\-,]\s*(\d+)\s*$", str(lines))
            if not m:
                return "--lines must look like 120:180"
            a, b = max(1, int(m.group(1))), min(len(all_lines), int(m.group(2)))
            out = ["%s, lines %d-%d" % (title, a, b)] + [fmt(j) for j in range(a - 1, b)]
        else:
            out = [title, text if toks(text) <= budget - 60 else digest(text, budget - 80) +
                   "\n[partial view of %d lines: use --grep REGEX or --lines A:B for the rest]" % len(all_lines)]
        return clip("\n".join(out), budget, "\n…[clipped: narrow the grep or the line range]")

    def search(self, query, limit=8):
        terms = re.findall(r"[\w./\-]+", query or "")
        if not terms:
            return "Empty query."
        rows = []
        if self.has_fts:
            quoted = ['"%s"' % t.replace('"', '""') for t in terms]
            for joiner in (" ", " OR "):
                try:
                    rows = self.db.execute(
                        "SELECT ref, snippet(fts, 0, '«', '»', ' … ', 24) s FROM fts WHERE fts MATCH ? "
                        "ORDER BY bm25(fts) LIMIT ?", (joiner.join(quoted), limit * 4)).fetchall()
                except sqlite3.OperationalError:
                    rows = []
                if rows:
                    break
        else:
            rows = self.db.execute("SELECT ref, substr(content,1,300) s FROM fts WHERE %s LIMIT ?" % " AND ".join(
                "content LIKE ?" for _ in terms), ["%" + t + "%" for t in terms] + [limit * 4]).fetchall()
        out, seen = [], set()
        for r in rows:
            kind, key, start = r["ref"].split(":")
            if (kind, key) in seen:
                continue
            seen.add((kind, key))
            out.append("%s:%s (near lines %s:%d): %s" % (kind, key, start, int(start) + 39, one_line(r["s"], 220)))
            if len(out) >= limit:
                break
        if not out:
            return "No matches for %r." % query
        return "\n".join(out + ["Fetch with: recall REF --lines A:B  (or --grep REGEX)"])

    def disk_mb(self):
        try:
            return os.path.getsize(self.dir / "ctx.db") / 1e6
        except OSError:
            return 0.0

    def gc(self, days, vacuum=False):
        """Remove archived originals and notes last used more than `days` days ago. Returns (outputs, notes)."""
        cutoff = time.time() - days * 86400
        with self.lock:
            nb = self.db.execute("SELECT COUNT(*) FROM blobs WHERE ts<?", (cutoff,)).fetchone()[0]
            nf = self.db.execute("SELECT COUNT(*) FROM folds WHERE ts<?", (cutoff,)).fetchone()[0]
            if nb or nf:
                self.db.execute("DELETE FROM fts WHERE rowid IN (SELECT rowid FROM fts WHERE ref LIKE 'r:%' AND "
                                "substr(ref,3,10) IN (SELECT ref FROM blobs WHERE ts<?))", (cutoff,))
                self.db.execute("DELETE FROM fts WHERE rowid IN (SELECT rowid FROM fts WHERE ref LIKE 's:%' AND "
                                "CAST(substr(ref,3) AS INTEGER) IN (SELECT id FROM folds WHERE ts<?))", (cutoff,))
                self.db.execute("DELETE FROM blobs WHERE ts<?", (cutoff,))
                self.db.execute("DELETE FROM folds WHERE ts<?", (cutoff,))
            self.db.execute("DELETE FROM sessions WHERE updated<?", (cutoff,))
            self.db.commit()
            self.known = set()
            if vacuum:
                self.db.execute("VACUUM")       # hand the freed space back to the disk
        return nb, nf

    def stats(self):
        q = lambda sql: self.db.execute(sql).fetchone()[0]
        return ("store: %s (%.1f MB on disk)\nsessions: %d   summaries: %d   archived tool outputs: %d (%.1f MB raw)" % (
            self.dir, self.disk_mb(), q("SELECT COUNT(*) FROM sessions"), q("SELECT COUNT(*) FROM folds"),
            q("SELECT COUNT(*) FROM blobs"), q("SELECT COALESCE(SUM(n_bytes),0) FROM blobs") / 1e6))


# ------------------------------------------------------------------ compressor

def transcript(ad, msgs, user_clip=None, asst_clip=None):
    """Plain-text rendering of a message range. Unclipped for the archive; clipped for the summariser."""
    out = []
    for m in msgs:
        for p in ad.parts(m):
            if p[0] == "text" and p[2].strip():
                out.append("%s: %s" % ("USER" if p[1] == "user" else "ASSISTANT",
                                       p[2].strip()[:user_clip if p[1] == "user" else asst_clip]))
            elif p[0] == "call":
                out.append("CALL %s %s" % (p[2], one_line(json.dumps(p[3], ensure_ascii=False), 300)))
            elif p[0] == "result":
                out.append("RESULT: %s" % p[2].strip())
    return "\n".join(out)


def mechanical(ad, msgs, budget):
    """Model-free note, used when no model-written one is available. Same idea as the model's note:
    what the user said, what was touched, what was run, what went wrong, and the agent's latest remarks."""
    results = {p[1]: p[2] for m in msgs for p in ad.parts(m) if p[0] == "result"}
    users, says, cmds, errors, files = [], [], [], [], {}
    for m in msgs:
        for p in ad.parts(m):
            if p[0] == "text" and p[2].strip():
                (users if p[1] == "user" else says).append(p[2].strip())
            elif p[0] == "call":
                args = p[3] if isinstance(p[3], dict) else {}
                for k, v in args.items():
                    if isinstance(v, str) and ("path" in k or "file" in k) and p[2] not in files.setdefault(v, []):
                        files[v].append(p[2])
                cmd = args.get("command", args.get("cmd"))
                cmd = " ".join(map(str, cmd)) if isinstance(cmd, list) else cmd if isinstance(cmd, str) else None
                if cmd and one_line(cmd, 120) not in cmds:
                    cmds.append(one_line(cmd, 120))
                res = results.get(p[1], "")
                bad = next((l for l in res.split("\n") if SIGNAL.search(l) and not l.startswith("[ctxc")), None)
                if bad:
                    ref = REF.search(res)
                    errors.append("%s ⇒ %s%s" % (one_line(cmd or p[2], 60), one_line(bad, 110),
                                                  " [%s]" % ref.group(0) if ref else ""))

    def take(items, share, each):
        cap, out, used = budget * share, [], 0
        for it in reversed(items):              # newest first when space runs out
            line = "- " + it[:each]
            if used + toks(line) > cap:
                if not out:
                    out.append(clip(line, int(cap)))
                break
            out.append(line)
            used += toks(line)
        out.reverse()
        if len(out) < len(items):
            out.insert(0, "- (%d earlier ones omitted)" % (len(items) - len(out)))
        return out
    sections = [("REQUIREMENTS - what the user said, verbatim", take(users, 0.35, 1200)),
                ("CHANGES - files touched", take(["%s (%s)" % (f, ", ".join(t)) for f, t in files.items()], 0.14, 160)),
                ("FACTS - commands run", take(cmds, 0.12, 130)),
                ("FACTS - errors and warnings seen", take(errors, 0.15, 220)),
                ("AGENT REMARKS - most recent last", take([one_line(x, 240) for x in says], 0.22, 240))]
    return clip("\n".join("%s:\n%s" % (h, "\n".join(ls)) for h, ls in sections if ls), budget)

class Compressor:
    def __init__(self, store, window=100000, cli="ctxc", summarizer=None, wait=45.0):
        self.store, self.window, self.cli, self.wait = store, window, cli, wait
        self.summarizer = summarizer        # callable(ad, body, prompt, ctx) -> text, or None
        self.prepared, self.pending, self.failed = {}, set(), set()
        self.memo = {}                      # (kind, hash, len, budget) -> rewritten tool output
        self.recent = collections.deque(maxlen=30)      # latest requests, for the web page
        self.started = time.time()
        self.locks, self.guard = {}, threading.Lock()
        self.stats = {"requests": 0, "raw_tokens": 0, "sent_tokens": 0, "epochs": 0,
                      "model_summaries": 0, "mechanical_summaries": 0, "notes_timed_out": 0,
                      "api_requests": 0, "api_input_tokens": 0, "api_cached_tokens": 0, "api_estimated_tokens": 0}
        self.calib = {}                     # session -> (tokens the API counted) / (tokens estimated)

    def observe(self, info, usage, entry=None):
        """Record what the API itself counted for a request that was compressed (input tokens, cached part)."""
        counted, cached = usage
        est = info.get("est") or 0
        if not counted or not est:
            return
        self.stats["api_requests"] += 1
        self.stats["api_input_tokens"] += counted
        self.stats["api_cached_tokens"] += cached
        self.stats["api_estimated_tokens"] += est
        if entry is not None:
            entry["counted"], entry["cached"] = counted, cached
        ratio = min(2.0, max(0.5, counted / est))
        if len(self.calib) > 2000:
            self.calib.clear()
        old = self.calib.get(info.get("sid"))
        self.calib[info.get("sid")] = ratio if old is None else 0.7 * old + 0.3 * ratio

    def _memo(self, kind, text, budget, make):
        key = (kind, hash(text), len(text), budget)
        if key not in self.memo:
            if len(self.memo) > 20000:
                self.memo.clear()
            self.memo[key] = make()
        return self.memo[key]

    def _elide(self, block, kind):
        """Archive an image or document that is leaving the prompt, and return the text that replaces it."""
        return "[%s elided by ctxc · saved as r:%s]" % (kind, self.store.put(json.dumps(block, sort_keys=True), index=False))

    def _attach(self, raw):
        """Archive the non-text parts (images) of a tool result whose text is being replaced."""
        parts = [p for p in raw if isinstance(p, dict) and not isinstance(p.get("text"), str)] if isinstance(raw, list) else []
        if not parts:
            return ""
        return "\n[ctxc: %d attachment(s) saved as r:%s]" % (
            len(parts), self.store.put(json.dumps(parts, sort_keys=True), index=False))

    # -- per-message rewriting: pure functions of the raw messages and the zone edges
    def _cold(self, ad, msgs, a, b, w, rich=False):
        """Cold zone: digest tool output, drop results of calls repeated later, shorten big call arguments."""
        last, key_of = {}, {}
        for m in msgs[a:b]:
            for p in ad.parts(m):
                if p[0] == "call":
                    key = "%s %s" % (p[2], json.dumps(p[3], sort_keys=True, ensure_ascii=False, default=str))
                    key_of[p[1]], last[key] = key, p[1]
        keep, dig = max(120, w // 400), max(100, w // 500)
        if rich:                                # what the summariser reads: 4x more of each tool output
            keep, dig = keep * 4, dig * 4

        def rfn(cid, text, img, raw=None):
            t, extra = toks(text), self._attach(raw) if img else ""
            if t > 60 and cid in key_of and last[key_of[cid]] != cid:
                return self._memo("dup", text, 0, lambda: "[ctxc: output dropped, the same call was repeated "
                                  "later · r:%s]" % self.store.put(text)) + extra
            if t <= keep and not img:
                return None
            return self._memo("cold", text, dig, lambda: "%s\n[ctxc r:%s · %d lines · `%s recall r:%s --grep REGEX`]" % (
                digest(text, dig), self.store.put(text), len(_lines(text)), self.cli, self.store.put(text))) + extra

        def afn(args):
            out = None
            for k, v in args.items():
                if isinstance(v, str) and len(v) > 800:
                    out = out or dict(args)
                    out[k] = "%s …[ctxc: +%d chars elided · r:%s]" % (v[:300], len(v) - 300, self.store.put(v))
            return out
        return [ad.trim(m, rfn, afn, self._elide) for m in msgs[a:b]]

    def _hot(self, ad, msgs, a, cap):
        """Hot zone: verbatim, except single tool outputs too large for the window."""
        def rfn(cid, text, img, raw=None):
            if toks(text) <= cap:
                return None
            return self._memo("hot", text, cap, lambda: (
                "%s\n[ctxc r:%s · %d lines, too large for the window · `%s recall r:%s --lines A:B` or `--grep REGEX`]"
                % (digest(text, cap // 2), self.store.put(text), len(_lines(text)), self.cli, self.store.put(text)))
            ) + (self._attach(raw) if img else "")
        return [ad.trim(m, rfn, None, None) for m in msgs[a:]]

    def _view(self, ad, msgs, st, head, w):
        texts = [INTRO.format(cli=self.cli)] + ["[s:%d · messages %d–%d]\n%s" % (s["id"], s["a"], s["b"], s["text"])
                                                 for s in st["sums"]]
        summary = [ad.summary(texts)] if st["sums"] else []
        cold = self._cold(ad, msgs, st["f"], st["c"], w)
        hot = self._hot(ad, msgs, st["c"], int(w * 0.14 * st.get("k", 1.0)))
        return msgs[:head] + summary + cold + hot

    # -- the main entry point
    def compress(self, ad, body, headers=None, ctx=None, dry=False):
        """Rewrite body in place. Returns (raw_tokens, sent_tokens, epoch_happened).
        dry=True gives the same view without folding, starting notes, or touching state and statistics."""
        msgs = ad.msgs(body)
        if msgs is None:
            return 0, 0, False
        n, head = len(msgs), ad.head(msgs)
        if n <= head:
            return 0, 0, False
        sid = hashlib.sha1((ad.name + ad.hint(body, headers or {}) + mhash(msgs[head])).encode()).hexdigest()[:16]
        with self.guard:
            lock = self.locks.setdefault(sid, threading.Lock())
        with lock:
            over = ad.overhead(body, msgs[:head])
            w = max(self.window - over, int(self.window * 0.3))     # room left for the conversation
            st = self.store.load(sid) or {"f": head, "c": head, "ch": None, "sums": [], "k": 1.0}
            self._verify(msgs, st, head)
            view = self._view(ad, msgs, st, head, w)
            total = over + tok(view[head:])
            epoch = False
            # If the API has been counting more tokens than estimated for this session, fold that much sooner.
            scale = max(1.0, self.calib.get(sid, 1.0))
            if dry:
                pass
            elif total * scale > over + 0.70 * w:
                epoch = True
                st["k"] = 1.0
                ep = {"scale": scale}           # all waiting in this request shares one deadline
                for _ in range(4):
                    self._epoch(ad, body, msgs, st, over, w, sid, ctx, ep)
                    view = self._view(ad, msgs, st, head, w)
                    total = over + tok(view[head:])
                    if total * scale <= self.window * 0.92:
                        break
                    st["k"] /= 2                # one enormous recent result: tighten the hot zone
                self.store.save(sid, st)
                self.stats["epochs"] += 1
            elif total * scale > over + 0.50 * w:
                self._prepare(ad, body, msgs, st, sid, w, ctx)
            ad.put(body, view)
            if ad.name == "anthropic" and st["sums"]:
                self._cache_mark(body, view[head])
            raw = over + sum(tok(m) for m in msgs[st["f"]:]) + int(st.get("raw_folded", 0))
            if dry:
                return raw, total, False
            if ctx is not None:
                ctx["sid"], ctx["est"] = sid, total
            self.stats["requests"] += 1
            self.stats["raw_tokens"] += raw
            self.stats["sent_tokens"] += total
            return raw, total, epoch

    def _verify(self, msgs, st, head):
        """The client may have rewound or edited history: keep only state that still matches it."""
        n, good = len(msgs), []
        for s in st["sums"]:
            if s["b"] > n or mhash(msgs[s["b"] - 1]) != s["h"]:
                break
            good.append(s)
        if len(good) != len(st["sums"]):
            st["sums"], st["c"] = good, 0
        st["f"] = good[-1]["b"] if good else head
        if st["c"] < st["f"] or st["c"] > n or (st["c"] > st["f"] and mhash(msgs[st["c"] - 1]) != st.get("ch")):
            st["c"] = st["f"]

    def _chunk_end(self, ad, msgs, f, w, bounds=None):
        """End of the next range to fold: a fixed amount of raw history, so it can be predicted early."""
        bounds = ad.boundaries(msgs, f) if bounds is None else bounds
        acc = 0
        for i in range(f, len(msgs)):
            if i > f and i in bounds and acc >= 0.5 * w:
                return i
            acc += tok(msgs[i])
        return None

    def _epoch(self, ad, body, msgs, st, over, w, sid, ctx, ep):
        n, f = len(msgs), st["f"]
        bounds = ad.boundaries(msgs, f)
        hot = self._hot(ad, msgs, f, int(w * 0.14 * st["k"]))
        # 1. hot zone: as many recent messages as fit in ~20% of the room, starting at a safe boundary
        c, acc, budget = None, 0, 0.20 * w * st["k"]
        for i in range(n - 1, f, -1):
            acc += tok(hot[i - f])
            if acc > budget:
                break
            if i in bounds:
                c = i
        if c is None:
            c = max(bounds) if bounds else f
        # 2. fold the oldest cold messages into notes until the view is back to ~42% of the room
        cold_t = [tok(m) for m in self._cold(ad, msgs, f, c, w)]
        total = over + sum(toks(s["text"]) for s in st["sums"]) + sum(cold_t) + sum(tok(m) for m in hot[c - f:])
        ep.setdefault("deadline", time.time() + self.wait)     # the clock starts now, after the local work above
        if self.summarizer and not ep.get("planned"):
            # Several ranges may need notes at once (a long session met for the first time): ask for them
            # together, so the request waits once, not once per note. At most 8; older backlog beyond that
            # gets mechanical notes.
            ep["planned"] = True
            pf, ptotal, pcold = f, total, cold_t
            for _ in range(8):
                if not (ptotal * ep.get("scale", 1.0) > over + 0.42 * w and pf < c):
                    break
                end = self._chunk_end(ad, msgs, pf, w, bounds)
                end = c if end is None or end > c else end
                self._start((sid, pf, end), ad, body, ctx, lambda: self._fold_prompt(ad, msgs, st, pf, end, w))
                ptotal += w // 60 - sum(pcold[:end - pf])
                pcold, pf = pcold[end - pf:], end
        while total * ep.get("scale", 1.0) > over + 0.42 * w and f < c:
            end = self._chunk_end(ad, msgs, f, w, bounds)
            end = c if end is None or end > c else end
            rng = self._cold(ad, msgs, f, end, w)
            text = self._note((sid, f, end), ad, body, ctx, lambda: self._fold_prompt(ad, msgs, st, f, end, w), ep)
            self.stats["model_summaries" if text else "mechanical_summaries"] += 1
            text = clip(text, w // 40) if text else mechanical(ad, rng, max(300, w // 60))
            fid = self.store.add_fold(sid, f, end, text, transcript(ad, rng))
            st["sums"].append({"id": fid, "a": f, "b": end, "h": mhash(msgs[end - 1]), "text": text})
            st["raw_folded"] = st.get("raw_folded", 0) + sum(tok(m) for m in msgs[f:end])
            total += toks(text) - sum(cold_t[:end - f])
            cold_t, f = cold_t[end - f:], end
        # 3. keep the summary zone itself bounded: merge the oldest notes (older work = coarser memory)
        while sum(toks(s["text"]) for s in st["sums"]) > 0.16 * w and len(st["sums"]) >= 2:
            old = st["sums"][:FANIN]
            ids = [s["id"] for s in old]
            text = self._note((sid, "merge", tuple(ids)), ad, body, ctx, lambda: self._merge_prompt(old, w), ep)
            if not text:
                each = max(60, int(0.04 * w) // len(old))
                text = "\n".join("(messages %d–%d) %s" % (s["a"], s["b"], clip(s["text"], each)) for s in old)
            text = clip(text, int(0.045 * w))
            fid = self.store.add_fold(sid, old[0]["a"], old[-1]["b"], text, "", ids)
            st["sums"][:FANIN] = [{"id": fid, "a": old[0]["a"], "b": old[-1]["b"], "h": old[-1]["h"], "text": text}]
        st["f"], st["c"] = f, max(c, f)
        st["ch"] = mhash(msgs[st["c"] - 1]) if st["c"] > f else None

    def _fold_prompt(self, ad, msgs, st, f, end, w):
        notes = "\n\n".join(s["text"] for s in st["sums"])
        after = transcript(ad, self._cold(ad, msgs, end, min(len(msgs), end + 40), w), 4000, 2000)
        return SUMMARISE.format(
            words=max(200, w // 75),
            notes=clip(notes, int(0.16 * w)) or "(none: this slice is the start of the session)",
            body=clip(transcript(ad, self._cold(ad, msgs, f, end, w, rich=True), 12000, 4000), int(0.45 * self.window)),
            after=clip(after, int(0.06 * w)) or "(nothing yet)")

    def _merge_prompt(self, old, w):
        return MERGE.format(words=max(250, w // 60), body="\n\n".join(
            "<note messages=\"%d-%d\">\n%s\n</note>" % (s["a"], s["b"], s["text"]) for s in old))

    def _start(self, key, ad, body, ctx, prompt):
        """Run the summariser for one range in a background thread (once per range)."""
        if not self.summarizer or key in self.prepared or key in self.pending or key in self.failed:
            return
        self.pending.add(key)
        try:                                    # build the prompt here: the worker thread must not touch the store
            prompt = prompt() if callable(prompt) else prompt
        except Exception:
            self.pending.discard(key)
            raise

        def work():
            try:
                text = self.summarizer(ad, body, prompt, ctx)
                if text and text.strip():
                    self.prepared[key] = text.strip()
                else:
                    self.failed.add(key)
                    log("the model returned an empty note; a mechanical note will be used")
            except Exception as e:
                self.failed.add(key)            # do not retry this range on every request
                log("summariser failed (%s); a mechanical note will be used" % e)
            finally:
                self.pending.discard(key)
        threading.Thread(target=work, daemon=True).start()

    def _note(self, key, ad, body, ctx, prompt, ep):
        """Model-written note for a range: normally prepared in advance; otherwise ask now and wait until the
        request's shared deadline. Returns None if there is no summariser or it did not answer in time."""
        if time.time() < ep["deadline"]:
            self._start(key, ad, body, ctx, prompt)
        while key in self.pending and time.time() < ep["deadline"]:
            time.sleep(0.02)
        text = self.prepared.pop(key, None)
        if text is None and self.summarizer and key not in self.failed:     # not ready: do not hold the agent up
            self.stats["notes_timed_out"] += 1
            if not ep.get("logged"):
                ep["logged"] = True
                log("a note was not ready after %gs, so a mechanical note was used. To avoid this, give notes "
                    "more time (--wait 90) or a faster model (--summarizer MODEL)" % self.wait)
        return text

    def _prepare(self, ad, body, msgs, st, sid, w, ctx):
        """Ahead of the next epoch, have the model write the next note in the background, so the
        epoch itself rarely has to wait."""
        if not self.summarizer:
            return
        f, end = st["f"], self._chunk_end(ad, msgs, st["f"], w)
        if end is not None:
            sums = list(st["sums"])           # the prompt is only built if the job actually starts
            self._start((sid, f, end), ad, body, ctx,
                        lambda: self._fold_prompt(ad, msgs, {"sums": sums}, f, end, w))
        if sum(toks(s["text"]) for s in st["sums"]) > 0.13 * w and len(st["sums"]) >= 2:
            old = st["sums"][:FANIN]
            self._start((sid, "merge", tuple(s["id"] for s in old)), ad, body, ctx, self._merge_prompt(old, w))

    def _cache_mark(self, body, summary_msg):
        """Anthropic caching is opt-in per breakpoint: put one at the end of the summaries so that
        prefix stays cached across epochs. Only if the client caches and has a breakpoint to spare."""
        found = []

        def walk(o):
            if isinstance(o, dict):
                if isinstance(o.get("cache_control"), dict):
                    found.append(o["cache_control"])
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(body.get("messages"))
        in_msgs = len(found)
        walk(body.get("system"))
        walk(body.get("tools"))
        if found and len(found) < 4:
            summary_msg["content"][-1] = dict(summary_msg["content"][-1],
                                              cache_control=dict(found[in_msgs - 1] if in_msgs else found[-1]))


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Context Compressor</title>
<style>
:root{--bg:#F6F8FA;--card:#FFFFFF;--line:#E2E8F0;--text:#0F172A;--muted:#5B6676;--accent:#0E7C86;--soft:#E6F6F7;--ok:#16A34A}
@media (prefers-color-scheme:dark){:root{--bg:#0B1220;--card:#121C2E;--line:#243349;--text:#E6EBF3;--muted:#9AA7B8;--accent:#44C7D1;--soft:#12313A;--ok:#4ADE80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:22px;margin:0}h2{font-size:15px;margin:0 0 12px;font-weight:700}
.top{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:22px}
.state{color:var(--muted);font-size:14px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--ok);margin-right:6px}
.dot.off{background:#DC2626}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin-bottom:14px}
.tile{margin:0}.num{font-size:28px;font-weight:700;letter-spacing:-.5px}.lab{color:var(--muted);font-size:13px}
.sub{color:var(--muted);font-size:13px;margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:14px}th{text-align:left;color:var(--muted);font-weight:600;font-size:12px;padding:0 10px 8px 0}
td{padding:8px 10px 8px 0;border-top:1px solid var(--line);white-space:nowrap}td.r,th.r{text-align:right}
.tag{background:var(--soft);color:var(--accent);border-radius:999px;padding:2px 9px;font-size:12px;font-weight:600}
.empty{color:var(--muted);padding:6px 0}
code,pre,input{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}
.cmd{display:flex;gap:10px;align-items:center;margin:8px 0}.cmd span{color:var(--muted);width:190px;flex:none;font-size:14px}
.cmd code{flex:1;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 10px;overflow-x:auto;white-space:nowrap}
button{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--card);color:var(--text);border-radius:8px;padding:7px 12px;cursor:pointer}
button:hover{border-color:var(--accent);color:var(--accent)}
.row{display:flex;gap:10px}input{flex:1;background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:9px 11px}
input:focus{outline:2px solid var(--accent);outline-offset:-1px}
.hit{display:block;width:100%;text-align:left;margin-top:8px;white-space:normal;line-height:1.4}.hit b{color:var(--accent)}
pre{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto;max-height:420px;margin:12px 0 0;white-space:pre-wrap}
.bar{height:8px;border-radius:999px;background:var(--line);overflow:hidden;margin-top:10px}.bar i{display:block;height:100%;background:var(--accent)}
</style></head><body><main>
<div class="top"><h1>Smart Context Compressor</h1><div class="state"><span class="dot" id="dot"></span><span id="state">connecting…</span></div></div>

<div class="tiles">
  <div class="card tile"><div class="lab">Requests handled</div><div class="num" id="req">–</div><div class="sub" id="up"></div></div>
  <div class="card tile"><div class="lab">Tokens saved</div><div class="num" id="saved">–</div><div class="bar"><i id="bar" style="width:0"></i></div></div>
  <div class="card tile"><div class="lab">Sent to the model</div><div class="num" id="sent">–</div><div class="sub" id="raw"></div></div>
  <div class="card tile"><div class="lab">Notes written</div><div class="num" id="notes">–</div><div class="sub" id="notesub"></div></div>
  <div class="card tile"><div class="lab">Read from cache</div><div class="num" id="cache">–</div><div class="sub" id="cachesub">as counted by the API</div></div>
</div>

<div class="card"><h2>Connect an agent</h2>
  <div class="cmd"><span>Claude Code</span><code id="c1"></code><button data-copy="c1">Copy</button></div>
  <div class="cmd"><span>Other agents: base URL</span><code id="c2"></code><button data-copy="c2">Copy</button></div>
  <div class="cmd"><span>Another provider</span><code id="c3"></code><button data-copy="c3">Copy</button></div>
</div>

<div class="card"><h2>Recent requests</h2>
  <div id="recent"><div class="empty">Nothing yet. Start an agent with the address above and requests will appear here.</div></div>
</div>

<div class="card"><h2>Search what was compressed away</h2>
  <div class="row"><input id="q" placeholder="Words to look for, e.g. an error message or a file name" autocomplete="off"><button id="go">Search</button></div>
  <div id="hits"></div><pre id="view" hidden></pre>
</div>
</main>
<script>
const $=id=>document.getElementById(id), n=x=>Math.round(x).toLocaleString();
const short=x=>x>=1e9?(x/1e9).toFixed(2)+"B":x>=1e6?(x/1e6).toFixed(1)+"M":x>=1e3?(x/1e3).toFixed(1)+"K":String(x);
const el=(tag,text,cls)=>{const e=document.createElement(tag);if(text!=null)e.textContent=text;if(cls)e.className=cls;return e};
const o=location.origin;
$("c1").textContent="ANTHROPIC_BASE_URL="+o+" claude";
$("c2").textContent=o;
$("c3").textContent=o+"/-/https/HOST/PATH";
document.querySelectorAll("[data-copy]").forEach(b=>b.onclick=()=>{navigator.clipboard.writeText($(b.dataset.copy).textContent);b.textContent="Copied";setTimeout(()=>b.textContent="Copy",1200)});
function ago(t){const s=Date.now()/1000-t;return s<60?Math.max(1,Math.round(s))+"s ago":s<3600?Math.round(s/60)+" min ago":Math.round(s/3600)+" h ago"}
async function tick(){
  try{
    const d=await (await fetch("/__ctxc/stats")).json();
    $("dot").className="dot";$("state").textContent="running · window "+n(d.window)+" tokens · notes: "+(d.summaries==="off"?"mechanical":d.summaries==="auto"?"written by the model":"written by "+d.summaries)+" · archive "+d.archive_mb+" MB · v"+d.version;
    $("req").textContent=n(d.requests);$("up").textContent="up "+(d.uptime<3600?Math.round(d.uptime/60)+" min":(d.uptime/3600).toFixed(1)+" h");
    const pct=d.raw_tokens?100*(1-d.sent_tokens/d.raw_tokens):0;
    $("saved").textContent=d.raw_tokens?pct.toFixed(1)+"%":"–";$("bar").style.width=pct+"%";
    $("sent").textContent=short(d.sent_tokens);$("raw").textContent=d.raw_tokens?"instead of "+short(d.raw_tokens):"";
    $("notes").textContent=n(d.model_summaries+d.mechanical_summaries);
    $("notesub").textContent=d.model_summaries+" by the model · "+d.mechanical_summaries+" mechanical"+(d.notes_timed_out?" ("+d.notes_timed_out+" timed out)":"");
    if(d.api_requests){$("cache").textContent=(100*d.api_cached_tokens/d.api_input_tokens).toFixed(1)+"%";
      const off=100*(d.api_estimated_tokens/d.api_input_tokens-1);
      $("cachesub").textContent="API counted "+short(d.api_input_tokens)+" input tokens · estimates run "+Math.abs(off).toFixed(0)+"% "+(off>=0?"high":"low")}
    if(d.recent.length){
      const t=el("table"),h=el("tr");[["When"],["API"],["Model"],["Agent sent","r"],["Model received","r"],["API counted","r"],["From cache","r"],[""]].forEach(c=>h.appendChild(el("th",c[0],c[1])));t.appendChild(h);
      d.recent.slice(0,12).forEach(r=>{const tr=el("tr");tr.appendChild(el("td",ago(r.t)));tr.appendChild(el("td",r.api));tr.appendChild(el("td",r.model));
        tr.appendChild(el("td",n(r.raw),"r"));tr.appendChild(el("td",n(r.sent),"r"));
        tr.appendChild(el("td",r.counted?n(r.counted):"–","r"));tr.appendChild(el("td",r.counted?Math.round(100*r.cached/r.counted)+"%":"–","r"));const c=el("td");if(r.folded)c.appendChild(el("span","folded","tag"));tr.appendChild(c);t.appendChild(tr)});
      $("recent").replaceChildren(t)}
  }catch(e){$("dot").className="dot off";$("state").textContent="not reachable"}
}
async function show(ref,lines){const v=$("view");v.hidden=false;v.textContent="Loading…";
  const d=await (await fetch("/__ctxc/recall?ref="+encodeURIComponent(ref)+(lines?"&lines="+lines:""))).json();v.textContent=d.text}
async function search(){const q=$("q").value.trim();if(!q)return;$("view").hidden=true;
  const d=await (await fetch("/__ctxc/search?q="+encodeURIComponent(q))).json(),box=$("hits");box.replaceChildren();
  let any=false;d.lines.forEach(l=>{const m=l.match(/^((?:r:[0-9a-f]+|s:\d+))\s*(.*)$/);if(!m)return;any=true;
    const b=el("button",null,"hit");b.appendChild(el("b",m[1]+"  "));b.appendChild(document.createTextNode(m[2]));const near=m[2].match(/near lines (\d+):(\d+)/);
    b.onclick=()=>show(m[1],near?Math.max(1,near[1]-5)+":"+(+near[2]+5):"");box.appendChild(b)});
  if(!any)box.appendChild(el("div","Nothing found for those words.","empty"))}
$("go").onclick=search;$("q").onkeydown=e=>{if(e.key==="Enter")search()};
tick();setInterval(tick,3000);
</script></body></html>"""


# ----------------------------------------------------------------------- proxy

LOG_FILE = None      # `ctxc run` logs here so the agent's own screen stays clean


class UsageTap:
    """Reads the token counts out of a model reply as it streams past. It only looks: the bytes sent on to
    the agent are never changed. Returns (input tokens including cached, cached tokens) or None."""

    def __init__(self, content_type, encoding):
        enc = (encoding or "").lower()
        self.ok = enc in ("", "identity", "gzip")
        self.z = zlib.decompressobj(31) if enc == "gzip" else None
        self.sse = "event-stream" in (content_type or "")
        self.buf, self.counted, self.cached = b"", 0, 0

    def feed(self, data):
        if not self.ok:
            return
        try:
            self.buf += self.z.decompress(data) if self.z else data
            if self.sse:
                lines = self.buf.split(b"\n")
                self.buf = lines.pop()
                for line in lines:
                    if line.startswith(b"data:") and b'"usage"' in line:
                        self._take(line[5:])
            elif len(self.buf) > 4000000:
                self.ok, self.buf = False, b""
        except Exception:
            self.ok = False

    def done(self):
        try:
            if self.ok and not self.sse and self.buf:
                self._take(self.buf)
        except Exception:
            pass
        return (self.counted, self.cached) if self.counted else None

    def _take(self, raw):
        try:
            o = json.loads(raw)
        except ValueError:
            return
        if not isinstance(o, dict):
            return
        for u in (o.get("usage"), (o.get("message") or {}).get("usage"), (o.get("response") or {}).get("usage")):
            if not isinstance(u, dict):
                continue
            if "prompt_tokens" in u:                                    # OpenAI Chat Completions
                counted = u.get("prompt_tokens") or 0
                cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            elif "cache_read_input_tokens" in u or "cache_creation_input_tokens" in u:     # Anthropic
                cached = u.get("cache_read_input_tokens") or 0
                counted = (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0) + cached
            else:                                                       # OpenAI Responses, or Anthropic without caching
                counted = u.get("input_tokens") or 0
                cached = (u.get("input_tokens_details") or {}).get("cached_tokens") or 0
            self.counted, self.cached = max(self.counted, counted), max(self.cached, cached)


def log(msg):
    line = "[ctxc %s] %s\n" % (time.strftime("%H:%M:%S"), msg)
    if LOG_FILE:
        with open(LOG_FILE, "a") as f:
            f.write(line)
    else:
        sys.stderr.write(line)
        sys.stderr.flush()


HOP = {"host", "content-length", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
       "proxy-connection", "te", "trailer", "transfer-encoding", "upgrade"}


def upstream_request(method, url, headers, body, timeout=900):
    u = urlsplit(url)
    port = u.port or (443 if u.scheme == "https" else 80)
    local = u.hostname in ("127.0.0.1", "localhost", "::1")
    if u.scheme == "https":
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        tls = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE"))
        if proxy and not local:         # CONNECT through the proxy, TLS to the origin inside the tunnel
            p = urlsplit(proxy)
            conn = http.client.HTTPSConnection(p.hostname, p.port or 3128, timeout=timeout, context=tls)
            conn.set_tunnel(u.hostname, port)
        else:
            conn = http.client.HTTPSConnection(u.hostname, port, timeout=timeout, context=tls)
    else:
        conn = http.client.HTTPConnection(u.hostname, port, timeout=timeout)
    out = {k: v for k, v in headers.items() if k.lower() not in HOP}
    if body is not None:
        out["Content-Length"] = str(len(body))
    conn.request(method, (u.path or "/") + ("?" + u.query if u.query else ""), body=body, headers=out)
    return conn, conn.getresponse()


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    cfg = None          # set by serve()

    def log_message(self, *a):
        pass

    def _route(self):
        """/-/https/host/rest -> that upstream; anything else -> the default upstream for its API shape."""
        path = self.path
        if path.startswith("/-/"):
            scheme, _, rest = path[3:].partition("/")
            host, _, tail = rest.partition("/")
            return "%s://%s/%s" % (scheme, host, tail)
        anth = "/messages" in path.split("?")[0] or self.headers.get("anthropic-version")
        u = urlsplit(self.cfg["anthropic"] if anth else self.cfg["openai"])
        prefix = u.path.rstrip("/")
        if prefix.endswith("/v1") and path.startswith("/v1/"):      # upstream base already carries /v1
            path = path[3:]
        return "%s://%s%s%s" % (u.scheme, u.netloc, prefix, path)

    def _body(self):
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            data = b""
            while True:
                n = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if n == 0:
                    self.rfile.readline()
                    return data
                data += self.rfile.read(n)
                self.rfile.readline()
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else None

    def _reply(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _page(self, comp):
        """The built-in web page and its read-only JSON endpoints."""
        u = urlsplit(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        store = comp.store
        if u.path == "/":
            data = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            return self.wfile.write(data)
        with store.lock:
            if u.path == "/__ctxc/search":
                return self._reply(200, {"lines": store.search(q.get("q", ""), 12).split("\n")})
            if u.path == "/__ctxc/recall":
                return self._reply(200, {"text": store.recall(q.get("ref", ""), q.get("grep") or None,
                                                              q.get("lines") or None, 8000)})
            return self._reply(200, dict(comp.stats, version=VERSION, window=comp.window, store=str(store.dir),
                                         uptime=int(time.time() - comp.started), recent=list(comp.recent),
                                         archive_mb=round(store.disk_mb(), 1), retention=self.cfg.get("retention", 0),
                                         summaries=self.cfg["summarizer"]))

    def _handle(self):
        self.close_connection = True
        comp = self.cfg["compressor"]
        # A proxy bound to localhost only answers requests addressed to localhost. This stops a web page
        # from reaching it under another name (DNS rebinding) and using it to fetch from the local network.
        local = ("127.0.0.1", "localhost", "::1")
        host = (self.headers.get("Host") or "").strip()
        host = host[1:].split("]")[0] if host.startswith("[") else host.rsplit(":", 1)[0]
        if self.cfg["host"] in local and host not in local:
            self._body()
            return self._reply(403, {"type": "error", "error": {
                "type": "ctxc_forbidden_host", "message": "ctxc only answers requests addressed to localhost"}})
        if self.path.startswith("/__ctxc/") or (self.command == "GET" and self.path.split("?")[0] == "/"):
            return self._page(comp)
        body, url = self._body(), self._route()
        headers = {k: v for k, v in self.headers.items()}
        ad = adapter_for(url) if self.command == "POST" and body else None
        info = entry = None                     # set when this request was compressed and its reply should be read
        if ad:
            # Fail open: whatever goes wrong in here, the original request is forwarded unchanged.
            try:
                enc = self.headers.get("Content-Encoding", "").lower()
                if enc in ("", "identity", "gzip"):
                    parsed = json.loads(gzip.decompress(body) if enc == "gzip" else body)
                    ctx = {"url": url.replace("/count_tokens", ""), "headers": dict(headers)}
                    counting = url.split("?")[0].rstrip("/").endswith("/count_tokens")     # a question, not a turn
                    raw, sent, epoch = comp.compress(ad, parsed, {k.lower(): v for k, v in headers.items()}, ctx,
                                                     dry=counting)
                    if raw:
                        body = json.dumps(parsed, ensure_ascii=False, separators=(",", ":")).encode()
                        headers = {k: v for k, v in headers.items() if k.lower() != "content-encoding"}
                    if raw and not counting:
                        info = ctx
                        entry = {"t": time.time(), "api": ad.name, "model": str(parsed.get("model", "?")),
                                 "raw": raw, "sent": sent, "folded": epoch}
                        comp.recent.appendleft(entry)
                        if epoch or self.cfg["verbose"]:
                            log("%s %s: %d → %d tokens%s" % (ad.name, parsed.get("model", "?"), raw, sent,
                                                              "  (epoch)" if epoch else ""))
                else:
                    log("request body is %s-encoded; forwarding it uncompressed" % enc)
            except Exception as e:
                log("compression skipped (%s: %s)" % (type(e).__name__, e))
        try:
            conn, resp = upstream_request(self.command, url, headers, body)
        except Exception as e:
            return self._reply(502, {"type": "error", "error": {"type": "ctxc_upstream_error", "message": str(e)}})
        try:
            self.send_response_only(resp.status, resp.reason)
            length = resp.getheader("Content-Length")
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            chunked = length is None and self.command != "HEAD" and resp.status not in (204, 304)
            if length is not None:
                self.send_header("Content-Length", length)
            elif chunked:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            tap = UsageTap(resp.getheader("Content-Type"), resp.getheader("Content-Encoding")) \
                if info and resp.status == 200 else None
            while True:                         # bytes go back exactly as they arrive
                data = resp.read1(65536)
                if not data:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data) if chunked else data)
                self.wfile.flush()
                if tap:
                    tap.feed(data)
            if chunked:
                self.wfile.write(b"0\r\n\r\n")
            usage = tap.done() if tap else None
            if usage:
                comp.observe(info, usage, entry)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _handle


def make_summarizer(model):
    """Side call to the same upstream, with the credentials of the request that triggered it."""
    def run(ad, body, prompt, ctx):
        if not ctx:
            return None
        headers = {k: v for k, v in ctx["headers"].items() if k.lower() not in ("accept-encoding", "content-encoding")}
        headers["Content-Type"] = "application/json"
        conn, resp = upstream_request("POST", ctx["url"], headers, json.dumps(ad.side(body, prompt, model)).encode(), 180)
        try:
            raw = resp.read()
            if resp.status != 200:
                raise RuntimeError("HTTP %d %s" % (resp.status, raw[:200].decode("utf-8", "replace")))
            return ad.side_text(json.loads(raw))
        finally:
            conn.close()
    return run


def _maintain(store, days):
    """Background housekeeping: optional archive retention, and keeping the log file small."""
    while True:
        try:
            if days > 0:
                nb, nf = store.gc(days)
                if nb or nf:
                    log("clean-up: removed %d archived outputs and %d notes not used for %d days" % (nb, nf, days))
            path = Path(LOG_FILE) if LOG_FILE else store.dir / "ctxc.log"
            if path.exists() and path.stat().st_size > 5000000:         # keep the last ~500 KB
                tail = path.read_bytes()[-500000:]
                path.write_bytes(tail[tail.find(b"\n") + 1:])
        except Exception as e:
            log("housekeeping skipped (%s: %s)" % (type(e).__name__, e))
        time.sleep(6 * 3600)


def serve(a):
    store = Store(a.dir)
    if not a.cli:           # what the model is told to type for recall: `ctxc` if installed, else the full path
        a.cli = "ctxc" if shutil.which("ctxc") else '"%s" "%s"' % (sys.executable or "python3", Path(__file__).resolve())
    comp = Compressor(store, a.window, a.cli,
                      None if a.summarizer == "off" else make_summarizer(None if a.summarizer == "auto" else a.summarizer),
                      a.wait)
    Proxy.cfg = {"compressor": comp, "anthropic": a.anthropic_upstream, "openai": a.openai_upstream,
                 "verbose": a.verbose, "host": a.host, "summarizer": a.summarizer,
                 "retention": getattr(a, "retention_days", 0)}
    srv = ThreadingHTTPServer((a.host, a.port), Proxy)
    srv.daemon_threads = True
    threading.Thread(target=_maintain, args=(store, getattr(a, "retention_days", 0)), daemon=True).start()
    log("ctxc %s on http://%s:%d  window=%d  store=%s  summaries=%s" % (
        VERSION, a.host, srv.server_address[1], a.window, store.dir, a.summarizer))
    log("open http://%s:%d in a browser to see what it is doing" % (a.host, srv.server_address[1]))
    return srv


def _is_ctxc(host, port):
    try:
        conn = http.client.HTTPConnection(host, port, timeout=2)
        conn.request("GET", "/__ctxc/stats")
        return "window" in json.loads(conn.getresponse().read())
    except Exception:
        return False


def run(a):
    """Start the proxy, launch an agent pointed at it, stop the proxy when the agent exits."""
    global LOG_FILE
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not cmd:
        print("ctxc: say which agent to run, e.g.  ctxc run claude", file=sys.stderr)
        return 2
    if not shutil.which(cmd[0]):
        print("ctxc: command not found: %s" % cmd[0], file=sys.stderr)
        return 127
    directory = Path(a.dir or os.environ.get("CTXC_DIR") or Path.home() / ".ctxc")
    directory.mkdir(parents=True, exist_ok=True)
    LOG_FILE = str(directory / "ctxc.log")
    # a base URL the user already had set is where the agent was going: keep it as the upstream
    for var, attr, default in (("ANTHROPIC_BASE_URL", "anthropic_upstream", "https://api.anthropic.com"),
                               ("OPENAI_BASE_URL", "openai_upstream", "https://api.openai.com")):
        old = os.environ.get(var, "")
        u = urlsplit(old)
        ours = u.hostname in ("127.0.0.1", "localhost") and (u.port == a.port or _is_ctxc(u.hostname, u.port or 80))
        if old and u.hostname and getattr(a, attr) == default and not ours:     # never chain ctxc to itself
            setattr(a, attr, old)
    srv, port = None, a.port
    try:
        srv = serve(a)
    except OSError:
        if not (a.port and _is_ctxc(a.host, a.port)):      # busy, and not by another ctxc: take a free port
            a.port = 0
            srv = serve(a)
    if srv:
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = "http://%s:%d" % (a.host, port)
    env = dict(os.environ, ANTHROPIC_BASE_URL=base, OPENAI_BASE_URL=base + "/v1", CTXC_DIR=str(directory))
    print("ctxc: compressing for `%s`%s · watch it at %s" % (
        cmd[0], "" if srv else " (using the ctxc already running)", base), file=sys.stderr)
    proc = subprocess.Popen(cmd, env=env)
    while True:
        try:
            code = proc.wait()
            break
        except KeyboardInterrupt:           # Ctrl-C belongs to the agent; keep waiting for it
            continue
    if srv:
        srv.shutdown()
    return code if code >= 0 else 128 - code


# ---- always-on mode, for GUI harnesses (T3 Code, Conductor, ...) that start the agent themselves

LABEL = "com.smart-context-compressor.ctxc"


def _service_file():
    if sys.platform == "darwin":
        return Path.home() / "Library" / "LaunchAgents" / (LABEL + ".plist")
    return Path.home() / ".config" / "systemd" / "user" / "ctxc.service"


def _service_text(args, logfile):
    if sys.platform == "darwin":
        return plistlib.dumps({"Label": LABEL, "ProgramArguments": list(args), "RunAtLoad": True, "KeepAlive": True,
                               "StandardOutPath": logfile, "StandardErrorPath": logfile}).decode()
    quoted = " ".join('"%s"' % x.replace('"', '\\"') for x in args)
    return ("[Unit]\nDescription=Smart Context Compressor\n\n[Service]\nExecStart=%s\nRestart=always\nRestartSec=2\n"
            "StandardError=append:%s\n\n[Install]\nWantedBy=default.target\n" % (quoted, logfile))


def _service(action, args=None, logfile=None):
    """Start or stop the background service. Returns (ok, note)."""
    sh = lambda *c: subprocess.run(c, capture_output=True, text=True).returncode
    path, mac = _service_file(), sys.platform == "darwin"
    if not mac and not (sys.platform.startswith("linux") and shutil.which("systemctl")):
        return False, "this system has no supported background service; keep `ctxc serve` running yourself"
    if action == "start":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_service_text(args, logfile))
        if mac:
            domain = "gui/%d" % os.getuid()
            sh("launchctl", "bootout", "%s/%s" % (domain, LABEL))
            ok = False
            for _ in range(4):                  # the old instance may take a moment to go away
                ok = sh("launchctl", "bootstrap", domain, str(path)) == 0
                if ok:
                    break
                time.sleep(1)
            ok = ok or sh("launchctl", "load", "-w", str(path)) == 0
        else:
            sh("systemctl", "--user", "daemon-reload")
            ok = sh("systemctl", "--user", "enable", "ctxc.service") == 0 and \
                sh("systemctl", "--user", "restart", "ctxc.service") == 0
        return ok, str(path)
    if mac:
        if sh("launchctl", "bootout", "gui/%d/%s" % (os.getuid(), LABEL)) != 0 and path.exists():
            sh("launchctl", "unload", str(path))
    else:
        sh("systemctl", "--user", "disable", "--now", "ctxc.service")
    if path.exists():
        path.unlink()
    return True, str(path)


def _claude_settings():
    path = Path.home() / ".claude" / "settings.json"
    text = path.read_text() if path.exists() else ""
    return path, (json.loads(text) if text.strip() else {})


def turn_on(a):
    """Keep the proxy running in the background and point Claude Code at it for every session."""
    directory = Path(a.dir or os.environ.get("CTXC_DIR") or Path.home() / ".ctxc")
    directory.mkdir(parents=True, exist_ok=True)
    try:
        path, data = _claude_settings()
    except ValueError as e:
        print("ctxc: %s is not valid JSON (%s); fix it first, nothing was changed" % (Path.home() / ".claude/settings.json", e))
        return 1
    if not isinstance(data, dict) or not isinstance(data.get("env", {}), dict):
        print("ctxc: unexpected layout in %s; nothing was changed" % path)
        return 1
    base = "http://%s:%d" % (a.host, a.port)
    state_file = directory / "on.json"
    before = data.get("env", {}).get("ANTHROPIC_BASE_URL")
    if before == base and state_file.exists():              # already on: keep what was saved the first time
        before = json.loads(state_file.read_text()).get("before")
    elif before == base:
        before = None
    if before and a.anthropic_upstream == "https://api.anthropic.com":
        a.anthropic_upstream = before                       # the provider Claude Code was already using
    program = directory / "ctxc.py"                         # the service must not depend on the download folder
    if Path(__file__).resolve() != program.resolve():
        shutil.copyfile(str(Path(__file__).resolve()), str(program))
    args = [sys.executable or "python3", str(program), "--dir", str(directory), "serve",
            "--port", str(a.port), "--host", a.host, "--window", str(a.window), "--summarizer", a.summarizer,
            "--wait", str(a.wait), "--anthropic-upstream", a.anthropic_upstream, "--openai-upstream", a.openai_upstream,
            "--retention-days", str(a.retention_days)]
    note = "not installed (--no-service)"
    if not a.no_service:
        ok, note = _service("start", args, str(directory / "ctxc.log"))
        if not ok:
            print("ctxc: could not start the background service: %s" % note)
    deadline = time.time() + 8
    while not _is_ctxc(a.host, a.port) and time.time() < deadline:
        time.sleep(0.3)
    if not _is_ctxc(a.host, a.port):                        # never point the agent at a proxy that is not there
        print("ctxc: the compressor is not answering on %s, so Claude Code was left untouched.\n"
              "      See %s for the reason." % (base, directory / "ctxc.log"))
        return 1
    data.setdefault("env", {})["ANTHROPIC_BASE_URL"] = base
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    state_file.write_text(json.dumps({"before": before, "base": base}))
    print("ctxc is on.\n  compressor : running in the background at %s (open it in a browser)\n"
          "  service    : %s\n  program    : %s (a copy; run `on` again after updating)\n"
          "  Claude Code: %s now points at the compressor%s\n"
          "Restart any open agent or harness (T3 Code, Conductor, a terminal session) to pick it up.\n"
          "Turn it off again with:  %s off" % (base, note, program, path, " (upstream: %s)" % before if before else "",
                                                 "ctxc" if shutil.which("ctxc") else "python3 %s" % program))
    return 0


def turn_off(a):
    directory = Path(a.dir or os.environ.get("CTXC_DIR") or Path.home() / ".ctxc")
    state_file = directory / "on.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    base = state.get("base") or "http://%s:%d" % (a.host, a.port)
    try:
        path, data = _claude_settings()
        env = data.get("env") if isinstance(data, dict) else None
        if isinstance(env, dict) and env.get("ANTHROPIC_BASE_URL") == base:
            if state.get("before"):
                env["ANTHROPIC_BASE_URL"] = state["before"]
            else:
                del env["ANTHROPIC_BASE_URL"]
                if not env:
                    del data["env"]
            path.write_text(json.dumps(data, indent=2) + "\n")
            print("Claude Code: restored %s" % path)
        else:
            print("Claude Code: %s was not pointing at ctxc; left as it is" % path)
    except ValueError:
        print("ctxc: %s is not valid JSON; remove ANTHROPIC_BASE_URL from it by hand" % (Path.home() / ".claude/settings.json"))
    if not a.no_service:
        _service("stop")
        print("Background service: stopped and removed")
    if state_file.exists():
        state_file.unlink()
    print("ctxc is off. Restart any open agent or harness.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ctxc", description="Smart Context Compressor: a context-compressing proxy for very long agent sessions.")
    ap.add_argument("--dir", help="store directory (default ~/.ctxc or $CTXC_DIR)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, text in (("serve", "run the proxy"), ("run", "run an agent through the proxy:  ctxc run claude"),
                       ("on", "always on: background service + Claude Code pointed at it (for T3 Code and other harnesses)")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--port", type=int, default=8787)
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--window", type=int, default=100000, help="max tokens the model should ever see")
        p.add_argument("--summarizer", default="auto",
                       help="who writes the notes: auto = the same model the agent is using (best notes); "
                            "a model name = that model instead (pick a faster, cheaper one); "
                            "off = mechanical notes only, no extra API calls")
        p.add_argument("--wait", type=float, default=45.0,
                       help="max seconds one request may wait for model-written notes before mechanical ones are used")
        p.add_argument("--anthropic-upstream", default=os.environ.get("CTXC_ANTHROPIC_UPSTREAM", "https://api.anthropic.com"))
        p.add_argument("--openai-upstream", default=os.environ.get("CTXC_OPENAI_UPSTREAM", "https://api.openai.com"))
        p.add_argument("--retention-days", type=int, default=0,
                       help="remove archived originals not used for this many days (default 0 = keep everything)")
        p.add_argument("--cli", help="command name shown to the model for recall/search (default: found automatically)")
        p.add_argument("--verbose", action="store_true", help="log every request, not only epochs")
        if name == "run":
            p.add_argument("command", nargs=argparse.REMAINDER, help="the agent and its arguments")
        if name == "on":
            p.add_argument("--no-service", action="store_true", help="do not install the background service")
    p = sub.add_parser("off", help="undo `ctxc on`")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-service", action="store_true")
    p = sub.add_parser("recall", help="print an archived original: r:HASH or s:NUMBER")
    p.add_argument("ref")
    p.add_argument("--grep")
    p.add_argument("--lines")
    p.add_argument("--budget", type=int, default=1500)
    p = sub.add_parser("search", help="full-text search over everything archived")
    p.add_argument("query", nargs="+")
    p.add_argument("--limit", type=int, default=8)
    sub.add_parser("stats")
    p = sub.add_parser("gc", help="remove archived originals not used for N days and shrink the database")
    p.add_argument("--days", type=int, required=True)
    p = sub.add_parser("link", help="install a `ctxc` launcher into ~/.local/bin")
    p.add_argument("--bin", default=str(Path.home() / ".local" / "bin"))
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    if a.cmd in ("on", "off"):
        return turn_on(a) if a.cmd == "on" else turn_off(a)
    if a.cmd == "serve":
        srv = serve(a)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
    if a.cmd == "link":
        target = Path(a.bin) / "ctxc"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('#!/bin/sh\nexec "%s" "%s" "$@"\n' % (sys.executable or "python3", Path(__file__).resolve()))
        target.chmod(0o755)
        print("Installed %s (make sure %s is on PATH)" % (target, target.parent))
        return 0
    if a.cmd == "gc":
        store = Store(a.dir)
        before = store.disk_mb()
        try:
            nb, nf = store.gc(a.days, vacuum=True)
        except sqlite3.OperationalError as e:       # the running proxy holds the file: entries are gone, space comes later
            print("ctxc: removed old entries, but could not shrink the file right now (%s); run it again later" % e)
            return 1
        print("Removed %d archived outputs and %d notes not used for %d days. Database: %.1f MB → %.1f MB" % (
            nb, nf, a.days, before, store.disk_mb()))
        return 0
    try:
        store = Store(a.dir, readonly=True)
    except sqlite3.OperationalError:
        print("ctxc: no store yet at %s" % (a.dir or os.environ.get("CTXC_DIR") or Path.home() / ".ctxc"))
        return 1
    if a.cmd == "recall":
        print(store.recall(a.ref, a.grep, a.lines, a.budget))
    elif a.cmd == "search":
        print(store.search(" ".join(a.query), a.limit))
    else:
        print(store.stats())
    return 0


if __name__ == "__main__":
    sys.exit(main())
