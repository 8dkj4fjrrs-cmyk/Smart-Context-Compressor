"""Note-quality check: plant facts in a noisy session slice, fold it, count which facts survive.

Mechanical note only (no network):
    python3 tests/quality.py
Model-written note, through a real API (uses your key; one small request):
    python3 tests/quality.py --url https://api.anthropic.com/v1/messages --model MODEL \
        --header "x-api-key: $ANTHROPIC_API_KEY" --header "anthropic-version: 2023-06-01"
    python3 tests/quality.py --url https://api.openai.com/v1/chat/completions --model MODEL \
        --header "Authorization: Bearer $OPENAI_API_KEY"
"""
import argparse
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim  # noqa: E402
import ctxc  # noqa: E402

rng = random.Random(3)
noise = lambda tokens, i: sim.output(rng, tokens, i)
bury = lambda fact, i: noise(1500, i) + "\n" + fact + "\n" + noise(1500, i)

EVENTS = [
    ("user", "Build the refunds feature. All money amounts must be stored as integer pence, never floats. "
             "Do not touch the legacy/ directory."),
    ("tool", "Bash", {"command": "grep -rn refund src/"}, noise(2500, 1), "Looking at the existing refund code."),
    ("tool", "Read", {"file_path": "config/payments.env"}, bury("PAYMENTS_TIMEOUT_MS=4500", 2), "Reading config."),
    ("tool", "Bash", {"command": "pip freeze"}, bury("stripe==11.4.1", 3), "Checking dependency versions."),
    ("tool", "Bash", {"command": "pgbench -c 8 idempotency_keys"}, bury("latency p99 = 240 ms", 4),
     "Measuring Postgres write latency for idempotency keys."),
    ("say", "I'll store idempotency keys in Redis rather than Postgres because the Postgres p99 write latency was 240 ms."),
    ("user", "ok. also every refund must be logged to the audit table before the API call, not after."),
    ("tool", "Bash", {"command": "git mv billing/calc.py billing/ledger.py"}, "", "Renaming calc.py to ledger.py."),
    ("tool", "Edit", {"file_path": "billing/refund.py", "new_string": noise(600, 5)}, "ok", "Implementing split_amount."),
    ("tool", "Bash", {"command": "make test"}, noise(3000, 6) + "\nError: database 'billing_test' does not exist\n",
     "Running the tests."),
    ("tool", "Bash", {"command": "make test-fast SKIP_DB=1"}, noise(2500, 7) +
     "\nFAILED tests/test_refund.py::test_partial_refund_rounding - AssertionError: 1049 != 1050\n1 failed, 212 passed\n",
     "The full suite needs a database; trying the fast target."),
    ("tool", "Edit", {"file_path": "billing/refund.py", "new_string": noise(300, 8)}, "ok",
     "Fixed by rounding half up in split_amount."),
    ("tool", "Bash", {"command": "make test-fast SKIP_DB=1"}, noise(2500, 9) + "\n213 passed\n", "Re-running."),
    ("tool", "Bash", {"command": "./run_dev.sh"}, bury("refund-service listening on port 8443", 10), "Starting the service."),
    ("say", "Still unresolved: webhook retries double-charge when the signature header is missing. Next I will add "
            "a dedupe check in webhooks.py."),
    ("user", "go on"),
] + [("tool", "Bash", {"command": "grep -rn webhook src/ # %d" % i}, noise(1200, 20 + i), "Step %d." % i) for i in range(12)]

FACTS = {
    "requirement: integer pence": ["pence"],
    "requirement: leave legacy/ alone": ["legacy"],
    "requirement: audit log before API call": ["audit", "before"],
    "decision: Redis for idempotency keys": ["Redis", "idempotency"],
    "reason: Postgres p99 240 ms": ["240"],
    "config: PAYMENTS_TIMEOUT_MS=4500": ["4500"],
    "version: stripe 11.4.1": ["11.4.1"],
    "rename: calc.py -> ledger.py": ["ledger.py"],
    "working command: make test-fast SKIP_DB=1": ["SKIP_DB=1"],
    "failing test name": ["test_partial_refund_rounding"],
    "fix: round half up": ["half up"],
    "port 8443": ["8443"],
    "open: webhook double-charge": ["double-charge"],
    "next step: dedupe in webhooks.py": ["webhooks.py"],
}


def score(note):
    low = note.lower()
    return [name for name, keys in FACTS.items() if all(k.lower() in low for k in keys)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--model")
    ap.add_argument("--header", action="append", default=[])
    a = ap.parse_args()
    ad = ctxc.adapter_for(a.url) if a.url else ctxc.Anthropic()
    wire = sim.Wire(ad.name)
    for ev in EVENTS:
        wire.add(ev)
    with tempfile.TemporaryDirectory() as d:
        comp = ctxc.Compressor(ctxc.Store(d), 100000)
        w, msgs, head = 75000, wire.msgs, ad.head(wire.msgs)
        end = max(i for i in ad.boundaries(msgs, head) if i <= len(msgs) - 24)     # fold all but the last steps
        raw = sum(ctxc.tok(m) for m in msgs[head:end])
        notes = {"mechanical": ctxc.mechanical(ad, comp._cold(ad, msgs, head, end, w), max(300, w // 60))}
        if a.url:
            prompt = comp._fold_prompt(ad, msgs, {"sums": []}, head, end, w)
            headers = dict(h.split(":", 1) for h in a.header)
            body = dict(wire.body(), model=a.model)
            notes["model"] = ctxc.make_summarizer(a.model)(ad, body, prompt, {"url": a.url, "headers": {
                k.strip(): v.strip() for k, v in headers.items()}})
            print("summariser prompt: %d tokens" % ctxc.toks(prompt))
        print("slice: %d tokens, %d planted facts\n" % (raw, len(FACTS)))
        for name, note in notes.items():
            kept = score(note)
            print("%s note: %d tokens (%.0fx smaller), %d/%d facts kept" % (
                name, ctxc.toks(note), raw / max(1, ctxc.toks(note)), len(kept), len(FACTS)))
            for f in FACTS:
                print("   %s %s" % ("✓" if f in kept else "✗", f))
            print()
        if a.url:
            print("---- model note ----\n" + notes["model"])


if __name__ == "__main__":
    main()
