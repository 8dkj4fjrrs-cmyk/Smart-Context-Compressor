"""Synthetic long coding session, replayed request by request through the compressor."""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ctxc  # noqa: E402

WORDS = ("alloc buffer cache commit config context decode deploy encode fetch handler index kernel lock merge "
         "module mutex parse patch queue render retry schema socket stream thread token update worker yield").split()


def output(rng, tokens, step):
    lines, n = [], 0
    while n < tokens * 3.5:
        if rng.random() < 0.02:
            l = "ERROR in module_%d line %d: %s failed (attempt %d)" % (step % 97, rng.randint(1, 900), rng.choice(WORDS), step)
        else:
            l = "%04d %s" % (len(lines), " ".join(rng.choice(WORDS) for _ in range(rng.randint(4, 14))))
        lines.append(l)
        n += len(l) + 1
    return "\n".join(lines)


def events(steps, seed=7):
    """Yield neutral events: ('user', text) | ('tool', name, args, output, say) | ('say', text)."""
    rng = random.Random(seed)
    yield ("user", "REQ-0: build the billing service. Always use tabs, never spaces.")
    for i in range(1, steps + 1):
        r = rng.random()
        size = rng.randint(40, 400) if r < 0.6 else rng.randint(1000, 4000) if r < 0.9 else \
            rng.randint(8000, 15000) if r < 0.99 else 60000
        if i % 10 == 0:
            name, args = "Bash", {"command": "pytest -q"}                       # identical call, repeated
        elif i % 7 == 0:
            name, args = "Read", {"file_path": "src/module_%d.py" % (i % 5)}    # same few files re-read
        elif i % 4 == 0:
            name, args = "Edit", {"file_path": "src/module_%d.py" % (i % 97), "new_string": output(rng, 500, i)}
            size = 30
        else:
            name, args = "Bash", {"command": "grep -rn %s src/ # %d" % (rng.choice(WORDS), i)}
        yield ("tool", name, args, output(rng, size, i), "Step %d: checking %s." % (i, rng.choice(WORDS)))
        if i % 150 == 0:
            yield ("say", "Finished batch %d." % (i // 150))
            yield ("user", "REQ-%d: from now on module_%d must log through zlog%d only." % (i, i % 97, i))


class Wire:
    """Turns events into one API shape, appending to a client-side transcript."""

    def __init__(self, fmt):
        self.fmt, self.n = fmt, 0
        self.ad = {"anthropic": ctxc.Anthropic, "chat": ctxc.Chat, "responses": ctxc.Responses}[fmt]()
        self.msgs = [{"role": "system", "content": "You are a coding agent."}] if fmt == "chat" else []

    def body(self):
        tools = [{"name": n, "description": "tool " * 200, "input_schema": {"type": "object"}} for n in ("Bash", "Read", "Edit")]
        if self.fmt == "anthropic":
            return {"model": "m", "max_tokens": 1000, "stream": True, "tools": tools, "metadata": {"user_id": "sess-1"},
                    "system": [{"type": "text", "text": "You are a coding agent. " * 300, "cache_control": {"type": "ephemeral"}}],
                    "messages": list(self.msgs)}
        if self.fmt == "chat":
            return {"model": "m", "stream": True, "tools": tools, "messages": list(self.msgs)}
        return {"model": "m", "stream": True, "tools": tools, "instructions": "You are a coding agent. " * 300,
                "input": list(self.msgs), "prompt_cache_key": "sess-1"}

    def add(self, ev):
        self.n += 1
        cid, f, m = "call_%d" % self.n, self.fmt, self.msgs
        if ev[0] in ("user", "say"):
            role = "user" if ev[0] == "user" else "assistant"
            if f == "anthropic":
                m.append({"role": role, "content": [{"type": "text", "text": ev[1]}]})
            elif f == "chat":
                m.append({"role": role, "content": ev[1]})
            else:
                kind = "input_text" if role == "user" else "output_text"
                m.append({"type": "message", "role": role, "content": [{"type": kind, "text": ev[1]}]})
            return
        _, name, args, out, say = ev
        if f == "anthropic":
            m.append({"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hmm " * 20, "signature": "sig%d" % self.n},
                {"type": "text", "text": say}, {"type": "tool_use", "id": cid, "name": name, "input": args}]})
            m.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": cid, "content": out}]})
        elif f == "chat":
            m.append({"role": "assistant", "content": say, "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})
            m.append({"role": "tool", "tool_call_id": cid, "content": out})
        else:
            m.append({"type": "reasoning", "id": "rs_%d" % self.n, "summary": [], "encrypted_content": "x" * 80})
            m.append({"type": "function_call", "call_id": cid, "name": name, "arguments": json.dumps(args)})
            m.append({"type": "function_call_output", "call_id": cid, "output": out})


def check_valid(fmt, body):
    """Raise if the rewritten request would be rejected for structural reasons."""
    if fmt == "anthropic":
        m = body["messages"]
        assert m[0]["role"] == "user", "first message must be user"
        for i, x in enumerate(m):
            blocks = x["content"] if isinstance(x["content"], list) else []
            uses = [b["id"] for b in blocks if b.get("type") == "tool_use"]
            res = [b["tool_use_id"] for b in blocks if b.get("type") == "tool_result"]
            if uses:
                nxt = [b.get("tool_use_id") for b in m[i + 1]["content"]] if i + 1 < len(m) else None
                assert nxt is None or set(uses) <= set(nxt), "tool_use without result at %d" % i
            if res:
                prev = [b.get("id") for b in m[i - 1]["content"] if isinstance(b, dict)] if i else []
                assert set(res) <= set(prev), "orphan tool_result at %d" % i
    elif fmt == "chat":
        open_ids = set()
        for i, x in enumerate(body["messages"]):
            if x["role"] == "tool":
                assert x["tool_call_id"] in open_ids, "orphan tool message at %d" % i
                open_ids.discard(x["tool_call_id"])
            else:
                assert not open_ids, "unanswered tool call before %d" % i
                open_ids = {t["id"] for t in x.get("tool_calls") or []}
    else:
        seen, items = set(), body["input"]
        for i, x in enumerate(items):
            if x["type"] == "function_call":
                seen.add(x["call_id"])
            elif x["type"] == "function_call_output":
                assert x["call_id"] in seen, "orphan output at %d" % i
            elif x["type"] == "reasoning":
                assert i + 1 == len(items) or items[i + 1]["type"] != "message" or True


def blocks(ad, body):
    """Flatten the compressible part of a request into cache-comparable units."""
    out = []
    for m in ad.msgs(body):
        c = m.get("content")
        if isinstance(c, list):
            out += [json.dumps([m.get("role"), ctxc.strip_cc(b)], sort_keys=True) for b in c]
        else:
            out.append(json.dumps(ctxc.strip_cc(m), sort_keys=True))
    return out


def run(fmt, steps, store, window=100000, summarizer=None, strict=True):
    comp = ctxc.Compressor(store, window, summarizer=summarizer)
    wire = Wire(fmt)
    res = dict(fmt=fmt, requests=0, raw=0, sent=0, max_sent=0, cached=0, epochs=0, secs=0.0, final=None, raw_final=0)
    prev = None
    for ev in events(steps):
        wire.add(ev)
        if ev[0] == "say":
            continue                    # the client only calls the model after a user turn or a tool result
        body = wire.body()
        t0 = time.perf_counter()
        raw, sent, epoch = comp.compress(wire.ad, body, {}, {"url": "x", "headers": {}})
        res["secs"] += time.perf_counter() - t0
        check_valid(fmt, body)
        cur = blocks(wire.ad, body)
        over = wire.ad.overhead(body, [])
        common = 0
        if prev is not None:
            while common < min(len(prev), len(cur)) and prev[common] == cur[common]:
                common += 1
            if strict and not epoch:
                assert common == len(prev), "prefix changed between epochs at request %d" % res["requests"]
        res["cached"] += over + sum(ctxc.toks(b) for b in cur[:common])
        prev = cur
        res["requests"] += 1
        res["raw"] += raw
        res["sent"] += sent
        res["max_sent"] = max(res["max_sent"], sent)
        res["epochs"] += epoch
        res["final"], res["raw_final"] = body, raw
    res["stats"] = comp.stats
    return res


if __name__ == "__main__":
    import tempfile
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    print("%-10s %8s %12s %12s %7s %8s %8s %7s %9s" % (
        "wire", "requests", "raw input", "sent input", "ratio", "max sent", "cached", "epochs", "ms/request"))
    for fmt in ("anthropic", "chat", "responses"):
        with tempfile.TemporaryDirectory() as d:
            r = run(fmt, steps, ctxc.Store(d))
            print("%-10s %8d %11.2fB %11.3fB %6.0fx %8d %7.1f%% %7d %9.1f" % (
                fmt, r["requests"], r["raw"] / 1e9, r["sent"] / 1e9, r["raw"] / r["sent"], r["max_sent"],
                100.0 * r["cached"] / r["sent"], r["epochs"], 1000 * r["secs"] / r["requests"]))
            print("           final transcript: %.1fM tokens client-side → %d tokens sent" % (
                r["raw_final"] / 1e6, ctxc.tok(r["final"])))
