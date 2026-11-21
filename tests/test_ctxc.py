import gzip
import http.client
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim  # noqa: E402
import ctxc  # noqa: E402

SCRIPT = str(Path(ctxc.__file__).resolve())


def view_text(res):
    return json.dumps(res["final"], ensure_ascii=False)


class DigestTest(unittest.TestCase):
    def test_small_output_is_untouched(self):
        self.assertEqual(ctxc.digest("ok\n2 passed\n", 600), "ok\n2 passed")

    def test_large_output_keeps_the_error_and_respects_budget(self):
        lines = ["test_%d PASSED" % i for i in range(20000)]
        lines[12345] = "test_12345 FAILED AssertionError: expected 3 got 4"
        out = ctxc.digest("\n".join(lines) + "\n=== 1 failed, 19999 passed ===\n", 600)
        self.assertLessEqual(ctxc.toks(out), 600)
        self.assertIn("expected 3 got 4", out)
        self.assertIn(" 12346| ", out)          # original line number survives
        self.assertIn("1 failed, 19999 passed", out)

    def test_repeats_and_progress_bars_collapse(self):
        text = "\n".join(["downloading  1%\rdownloading 50%\rdownloading 100%"] + ["retrying..."] * 500 + ["done"])
        out = ctxc.digest(text, 600)
        self.assertIn("downloading 100%", out)
        self.assertNotIn("50%", out)
        self.assertIn("repeated 499 more times", out)


class SessionTest(unittest.TestCase):
    """Replays a long synthetic session. sim.run itself asserts, on every request, that the rewritten
    request is structurally valid and that the prompt prefix is unchanged between epochs."""

    def run_fmt(self, fmt, steps=600, window=100000, **kw):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = ctxc.Store(tmp.name)
        self.addCleanup(store.db.close)
        return sim.run(fmt, steps, store, window, **kw), store

    def check(self, fmt):
        res, store = self.run_fmt(fmt)
        self.assertLessEqual(res["max_sent"], 100000)
        self.assertGreater(res["raw"] / res["sent"], 8)
        self.assertGreater(res["cached"] / res["sent"], 0.9)
        text = view_text(res)
        for i in [0] + list(range(150, 601, 150)):       # every user requirement is still reachable
            marker = "REQ-%d:" % i
            self.assertTrue(marker in text or "s:" in store.search(marker), marker)
        self.assertIn("REQ-0: build the billing service", text)   # the original task is still in view
        return res, store

    def test_anthropic(self):
        res, store = self.check("anthropic")
        ref = ctxc.REF.search(view_text(res)).group(0)
        self.assertIn("0000 ", store.recall(ref, lines="1:3"))
        sid = res["final"]["messages"][0]["content"][1]["text"].split(" ")[0].strip("[")
        self.assertTrue(sid.startswith("s:"))
        self.assertIn("CALL", store.recall(sid))
        self.assertIn("matching lines", store.recall(sid, grep="pytest"))
        # the agent-facing CLI reads the same store from another process, read-only
        out = subprocess.run([sys.executable, SCRIPT, "--dir", str(store.dir), "recall", ref, "--grep", "ERROR|0001"],
                             capture_output=True, text=True, timeout=30).stdout
        self.assertIn("matching lines", out)
        out = subprocess.run([sys.executable, SCRIPT, "--dir", str(store.dir), "search", "zlog150"],
                             capture_output=True, text=True, timeout=30).stdout
        self.assertIn("s:", out)
        # a cache breakpoint was added at the end of the summaries, copying the client's style
        self.assertEqual(res["final"]["messages"][0]["content"][-1]["cache_control"], {"type": "ephemeral"})

    def test_chat_completions(self):
        res, _ = self.check("chat")
        self.assertEqual(res["final"]["messages"][0]["role"], "system")

    def test_responses(self):
        self.check("responses")

    def test_tiny_window_and_giant_outputs(self):
        res, _ = self.run_fmt("anthropic", steps=400, window=20000)
        self.assertLessEqual(res["max_sent"], 20000)

    def test_model_written_notes(self):
        calls = []

        def fake(ad, body, prompt, ctx):
            calls.append(prompt)
            return "MODELNOTE %d\nREQUIREMENTS\n- tabs, never spaces.\nFACTS\n%s" % (len(calls), "- port 8443 is used. " * 250)
        res, _ = self.run_fmt("anthropic", steps=450, summarizer=fake)
        self.assertGreater(res["stats"]["model_summaries"], 5)
        self.assertEqual(res["stats"]["mechanical_summaries"], 0)        # every fold got a model note
        self.assertIn("MODELNOTE", view_text(res))
        folds = [c for c in calls if "<slice>" in c]
        merges = [c for c in calls if "<note messages=" in c]
        self.assertTrue(merges)                                          # merging is model-written too
        first, later = folds[0], folds[3]
        self.assertIn("REQ-0: build the billing service", first)         # the slice itself
        self.assertIn("(none: this slice is the start", first)
        self.assertIn("MODELNOTE", later.split("\n</earlier_notes>")[0].split("<earlier_notes>\n")[-1])            # earlier notes are given as context
        nxt = later.split("<what_happened_next>\n")[-1]
        self.assertIn("CALL", nxt)                                       # hindsight: what came after the slice
        self.assertLess(max(ctxc.toks(c) for c in calls), 70000)         # the side call itself fits the window
        self.assertGreater(len(first), 20000)

    def test_slow_summariser_is_waited_for_within_the_limit(self):
        import time

        def slow(ad, body, prompt, ctx):
            time.sleep(0.15)
            return "SLOWNOTE"
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for wait, expect_model in ((5.0, True), (0.0, False)):
            store = ctxc.Store(tmp.name + "/w%d" % wait)
            self.addCleanup(store.db.close)
            comp = ctxc.Compressor(store, 100000, summarizer=slow, wait=wait)
            comp._prepare = lambda *a, **k: None                         # nothing prepared ahead: worst case
            wire = sim.Wire("chat")
            for ev in sim.events(120):
                wire.add(ev)
            body = wire.body()
            comp.compress(wire.ad, body, {}, {"url": "x", "headers": {}})
            self.assertEqual(comp.stats["model_summaries"] > 0, expect_model)
            self.assertEqual(comp.stats["notes_timed_out"] > 0, not expect_model)     # counted, and logged
            self.assertEqual("SLOWNOTE" in json.dumps(body), expect_model)
            sim.check_valid("chat", body)

    def test_mechanical_note_is_structured(self):
        wire = sim.Wire("anthropic")
        for ev in sim.events(40):
            wire.add(ev)
        note = ctxc.mechanical(wire.ad, wire.msgs, 1200)
        self.assertLessEqual(ctxc.toks(note), 1200)
        self.assertIn("REQUIREMENTS", note)
        self.assertIn("REQ-0: build the billing service. Always use tabs", note)
        self.assertIn("src/module_", note)
        self.assertIn("pytest -q", note)
        self.assertIn("ERROR in module_", note)

    def test_folded_text_and_images_stay_recoverable(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = ctxc.Store(tmp.name)
        self.addCleanup(store.db.close)
        comp = ctxc.Compressor(store, 100000)
        wire = sim.Wire("anthropic")
        spec = "SPEC-START " + "requirement line. " * 600 + "SPEC-END-MARKER"         # ~11K characters
        wire.msgs.append({"role": "user", "content": [
            {"type": "text", "text": spec},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD" * 2000}}]})
        for ev in list(sim.events(60))[1:]:
            wire.add(ev)
        wire.msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": "shot", "name": "Screenshot", "input": {}}]})
        wire.msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "shot", "content": [
            {"type": "text", "text": "screenshot taken\n" + "pixel row\n" * 400},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "WFla" * 2000}}]}]})
        for ev in list(sim.events(200, seed=5))[1:]:
            wire.add(ev)
        body = wire.body()
        comp.compress(wire.ad, body)
        sim.check_valid("anthropic", body)
        sent = json.dumps(body)
        self.assertNotIn("SPEC-END-MARKER", sent)                       # folded out of the prompt...
        hit = store.search("SPEC-END-MARKER")
        self.assertIn("s:", hit)                                        # ...but the whole message is in the archive
        self.assertIn("matching lines", store.recall(hit.split(" ")[0], grep="SPEC-END-MARKER"))
        self.assertNotIn("QUJDQUJD", sent)
        self.assertNotIn("WFlaWFla", sent)
        blobs = [zlib_text for (zlib_text,) in store.db.execute("SELECT data FROM blobs")]
        import zlib
        saved = b"".join(zlib.decompress(b) for b in blobs).decode()
        self.assertIn("QUJD" * 2000, saved)                             # the user's image
        self.assertIn("WFla" * 2000, saved)                             # the image inside a tool result
        self.assertEqual(store.search("QUJDQUJD")[:10], "No matches")   # image data is kept out of the search index

    def test_cut_off_notes_are_rejected(self):
        self.assertGreaterEqual(ctxc.Anthropic().side({"model": "m"}, "x", None)["max_tokens"], 4000)
        with self.assertRaises(ValueError):
            ctxc.Anthropic().side_text({"stop_reason": "max_tokens", "content": [{"type": "text", "text": "REQUIREMENTS - a"}]})
        with self.assertRaises(ValueError):
            ctxc.Chat().side_text({"choices": [{"finish_reason": "length", "message": {"content": "REQ"}}]})
        with self.assertRaises(ValueError):
            ctxc.Responses().side_text({"status": "incomplete", "output": []})
        self.assertEqual(ctxc.Anthropic().side_text({"stop_reason": "end_turn", "content": [{"type": "text", "text": "ok"}]}), "ok")

    def test_a_long_existing_session_waits_once_not_once_per_note(self):
        import time

        def slow(ad, body, prompt, ctx):
            time.sleep(0.4)
            return "NOTE"
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = None
        for wait, summ in ((0, None), (3.0, slow), (0.2, slow)):
            store = ctxc.Store(tmp.name + "/w%d" % (wait * 10))
            self.addCleanup(store.db.close)
            comp = ctxc.Compressor(store, 100000, summarizer=summ, wait=wait)
            wire = sim.Wire("anthropic")
            for ev in sim.events(700):                                  # ~1.5M tokens seen for the first time
                wire.add(ev)
            body = wire.body()
            t0 = time.time()
            comp.compress(wire.ad, body, {}, {"url": "x", "headers": {}})
            took = time.time() - t0
            notes = comp.stats["model_summaries"] + comp.stats["mechanical_summaries"]
            self.assertGreater(notes, 20)
            self.assertLessEqual(ctxc.tok(body), 100000)
            sim.check_valid("anthropic", body)
            if summ is None:
                base = took                                             # time with no waiting at all
                continue
            # the whole request waits at most once (it used to wait once per note: 27 x wait)
            self.assertLess(took - base, wait + 1.5, "%d notes: %.1fs over the no-wait time" % (notes, took - base))
            if wait == 3.0:
                self.assertGreaterEqual(comp.stats["model_summaries"], 8)    # the first eight were written in parallel
            else:
                # too slow for this deadline: mechanical, except pre-started notes that happened to be ready
                self.assertLessEqual(comp.stats["model_summaries"], 8)
                self.assertGreater(comp.stats["mechanical_summaries"], 15)

    def test_failing_summariser_falls_back(self):
        def boom(ad, body, prompt, ctx):
            raise RuntimeError("upstream said no")
        res, _ = self.run_fmt("chat", steps=200, summarizer=boom)
        self.assertEqual(res["stats"]["model_summaries"], 0)
        self.assertGreater(res["stats"]["mechanical_summaries"], 0)

    def test_rewind_and_edit_of_history(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = ctxc.Store(tmp.name)
        self.addCleanup(store.db.close)
        comp = ctxc.Compressor(store, 100000)
        wire = sim.Wire("anthropic")
        for ev in sim.events(300):
            wire.add(ev)
        body = wire.body()
        comp.compress(wire.ad, body)
        self.assertGreater(len(body["messages"][0]["content"]), 2)       # several summaries exist
        del wire.msgs[121:]                                              # user rewinds far back...
        wire.msgs.append({"role": "assistant", "content": [{"type": "text", "text": "different branch"}]})
        wire.msgs.append({"role": "user", "content": [{"type": "text", "text": "REWOUND: go another way"}]})
        for ev in list(sim.events(80, seed=99))[1:]:                     # ...and the session continues
            wire.add(ev)
            body = wire.body()
            comp.compress(wire.ad, body)
            sim.check_valid("anthropic", body)
        self.assertLessEqual(ctxc.tok(body), 100000)
        self.assertNotIn("Step 290", json.dumps(body))                   # nothing from the abandoned branch


SSE = [b"event: message_start\ndata: {\"type\":\"message_start\"}\n\n",
       b"event: content_block_delta\ndata: {\"delta\":{\"text\":\"hel\"}}\n\n",
       b"event: content_block_delta\ndata: {\"delta\":{\"text\":\"lo\"}}\n\n",
       b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"]


class Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        try:
            body = json.loads(raw)
        except ValueError:
            body = raw
        Upstream.seen.append((self.path, dict(self.headers), body))
        if isinstance(body, dict) and body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("request-id", "req_123")
            self.end_headers()
            for ev in SSE:
                self.wfile.write(b"%x\r\n%s\r\n" % (len(ev), ev))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        else:
            data = json.dumps({"content": [{"type": "text", "text": "SIDECALL NOTE: tabs, not spaces."}]}).encode()
            self.send_response(200 if isinstance(body, dict) else 400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


class ProxyTest(unittest.TestCase):
    def setUp(self):
        Upstream.seen = []
        self.up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        self.tmp = tempfile.TemporaryDirectory()
        up_url = "http://127.0.0.1:%d" % self.up.server_address[1]
        self.proxy = ctxc.serve(SimpleNamespace(
            dir=self.tmp.name, window=12000, cli="ctxc", summarizer="auto", anthropic_upstream=up_url,
            openai_upstream=up_url + "/v1", host="127.0.0.1", port=0, verbose=False, wait=5.0))
        self.proxy.RequestHandlerClass.cfg["host"] = "127.0.0.1"
        for s in (self.up, self.proxy):
            threading.Thread(target=s.serve_forever, daemon=True).start()
        self.port = self.proxy.server_address[1]

    def tearDown(self):
        self.up.shutdown()
        self.proxy.shutdown()
        self.up.server_close()
        self.proxy.server_close()
        self.tmp.cleanup()

    def post(self, path, body, headers=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        data = raw if raw is not None else json.dumps(body).encode()
        conn.request("POST", path, body=data, headers=dict({"Content-Type": "application/json"}, **(headers or {})))
        resp = conn.getresponse()
        out = resp.status, dict(resp.getheaders()), resp.read()
        conn.close()
        return out

    def test_end_to_end(self):
        wire = sim.Wire("anthropic")
        sent_raw = 0
        for ev in sim.events(60):
            wire.add(ev)
            if ev[0] == "say":
                continue
            body = wire.body()
            sent_raw = ctxc.tok(body)
            status, headers, data = self.post("/v1/messages?beta=true", body,
                                              {"x-api-key": "sk-test", "anthropic-version": "2023-06-01"})
            self.assertEqual(status, 200)
            self.assertEqual(data, b"".join(SSE))                   # stream relayed byte for byte
            self.assertEqual(headers["request-id"], "req_123")
        main = [s for s in Upstream.seen if s[2].get("stream")]
        side = [s for s in Upstream.seen if not s[2].get("stream")]
        self.assertGreater(sent_raw, 100000)                        # the client transcript is far past the window
        self.assertLessEqual(max(ctxc.tok(s[2]) for s in main), 12000)
        for path, headers, body in main:
            self.assertEqual(path, "/v1/messages?beta=true")
            self.assertEqual(headers["x-api-key"], "sk-test")
            self.assertEqual(body["system"], wire.body()["system"])  # system and tools are never touched
            sim.check_valid("anthropic", body)
        # summaries were requested in the background with the caller's credentials, and used
        self.assertTrue(side)
        self.assertEqual(side[0][1]["x-api-key"], "sk-test")
        self.assertNotIn("tools", side[0][2])
        self.assertIn("SIDECALL NOTE", json.dumps(main[-1][2]))
        status, _, data = self.post("/__ctxc/stats", {})
        stats = json.loads(data)
        self.assertGreater(stats["raw_tokens"], 5 * stats["sent_tokens"])

    def get(self, path, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        conn.request("GET", path, headers={"Host": host} if host else {})
        resp = conn.getresponse()
        out = resp.status, resp.getheader("Content-Type"), resp.read().decode()
        conn.close()
        return out

    def test_web_page(self):
        status, ctype, html = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        self.assertIn("<title>Smart Context Compressor</title>", html)
        self.assertNotIn("http://", html.replace("http://127.0.0.1", ""))     # no external assets
        wire = sim.Wire("anthropic")
        for ev in sim.events(40):
            wire.add(ev)
            if ev[0] != "say":
                self.post("/v1/messages", wire.body(), {"x-api-key": "sk-test"})
        stats = json.loads(self.get("/__ctxc/stats")[2])
        self.assertEqual(stats["recent"][0]["api"], "anthropic")
        self.assertGreater(stats["recent"][0]["raw"], stats["recent"][0]["sent"])
        self.assertNotIn("sk-test", json.dumps(stats))                        # credentials never surface
        lines = json.loads(self.get("/__ctxc/search?q=pytest")[2])["lines"]
        ref = lines[0].split(" ")[0]
        self.assertRegex(ref, r"^(r:[0-9a-f]+|s:\d+)$")
        self.assertIn(ref.split(":")[1], json.loads(self.get("/__ctxc/recall?ref=" + ref)[2])["text"])
        self.assertEqual(self.get("/__ctxc/stats", host="evil.example.com")[0], 403)   # DNS-rebinding guard
        self.assertEqual(self.get("/", host="evil.example.com")[0], 403)

    def test_run_launches_an_agent_through_the_proxy(self):
        child = (
            "import os, sys, json, http.client\n"
            "from urllib.parse import urlsplit\n"
            "u = urlsplit(os.environ['ANTHROPIC_BASE_URL'])\n"
            "assert os.environ['OPENAI_BASE_URL'] == os.environ['ANTHROPIC_BASE_URL'] + '/v1'\n"
            "c = http.client.HTTPConnection(u.hostname, u.port, timeout=20)\n"
            "c.request('POST', '/v1/messages', body=json.dumps({'model': 'm', 'stream': True, 'messages': "
            "[{'role': 'user', 'content': 'hi from the agent'}]}), headers={'Content-Type': 'application/json'})\n"
            "r = c.getresponse(); r.read()\n"
            "print('agent saw', r.status, sys.argv[1], os.environ['CTXC_DIR'])\n"
            "sys.exit(7)\n")
        up = "http://127.0.0.1:%d" % self.up.server_address[1]
        env = dict(__import__("os").environ, ANTHROPIC_BASE_URL=up)      # the user's existing provider setting
        proc = subprocess.run([sys.executable, SCRIPT, "--dir", self.tmp.name + "/run", "run", "--port", "0",
                               "--summarizer", "off", "--", sys.executable, "-c", child, "--resume"],
                              capture_output=True, text=True, timeout=60, env=env)
        self.assertEqual(proc.returncode, 7)                                     # the agent's exit code
        self.assertIn("agent saw 200 --resume " + self.tmp.name + "/run", proc.stdout)   # its arguments pass through
        self.assertIn("watch it at http://127.0.0.1:", proc.stderr)
        self.assertNotIn("[ctxc ", proc.stderr)                                  # log goes to a file, not the screen
        self.assertIn("window=100000", open(self.tmp.name + "/run/ctxc.log").read())
        self.assertEqual(Upstream.seen[-1][2]["messages"][0]["content"], "hi from the agent")   # reached the old upstream
        missing = subprocess.run([sys.executable, SCRIPT, "run", "no-such-agent-xyz"], capture_output=True, text=True)
        self.assertEqual(missing.returncode, 127)

    def test_on_and_off(self):
        import os
        home = self.tmp.name + "/home"
        os.makedirs(home + "/.claude")
        settings = home + "/.claude/settings.json"
        with open(settings, "w") as f:
            json.dump({"model": "opus", "env": {"FOO": "1", "ANTHROPIC_BASE_URL": "https://relay.example.com/api"}}, f)
        env = dict(os.environ, HOME=home)
        cli = lambda *args: subprocess.run([sys.executable, SCRIPT, "--dir", home + "/.ctxc"] + list(args),
                                           capture_output=True, text=True, timeout=60, env=env)
        # nothing listening: the agent's settings must not be touched
        dead = cli("on", "--no-service", "--port", "1")
        self.assertEqual(dead.returncode, 1)
        self.assertIn("left untouched", dead.stdout)
        self.assertEqual(json.load(open(settings))["env"]["ANTHROPIC_BASE_URL"], "https://relay.example.com/api")
        # proxy up (the one this test class started): settings are pointed at it, everything else kept
        for _ in range(2):                                               # running it twice changes nothing
            on = cli("on", "--no-service", "--port", str(self.port))
            self.assertEqual(on.returncode, 0, on.stdout + on.stderr)
        data = json.load(open(settings))
        self.assertEqual(data, {"model": "opus", "env": {"FOO": "1", "ANTHROPIC_BASE_URL": "http://127.0.0.1:%d" % self.port}})
        self.assertIn("upstream: https://relay.example.com/api", on.stdout)
        # the service runs from a private copy, so deleting the download cannot break the agent
        self.assertEqual(open(home + "/.ctxc/ctxc.py").read(), open(SCRIPT).read())
        self.assertIn("python3 %s/.ctxc/ctxc.py off" % os.path.realpath(home), on.stdout.replace(home, os.path.realpath(home)))
        self.assertEqual(cli("off", "--no-service").returncode, 0)
        self.assertEqual(json.load(open(settings))["env"]["ANTHROPIC_BASE_URL"], "https://relay.example.com/api")
        # with no previous value and no file: created, then removed cleanly
        os.remove(settings)
        self.assertEqual(cli("on", "--no-service", "--port", str(self.port)).returncode, 0)
        self.assertEqual(json.load(open(settings)), {"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:%d" % self.port}})
        cli("off", "--no-service")
        self.assertEqual(json.load(open(settings)), {})
        # broken settings file: refuse, change nothing
        open(settings, "w").write("{broken")
        self.assertEqual(cli("on", "--no-service", "--port", str(self.port)).returncode, 1)
        self.assertEqual(open(settings).read(), "{broken")

    def test_service_definitions(self):
        import plistlib
        from unittest import mock
        args = ["/usr/bin/python3", "/opt/my tools/ctxc.py", "serve", "--port", "8787"]
        with mock.patch.object(sys, "platform", "darwin"):
            plist = plistlib.loads(ctxc._service_text(args, "/tmp/ctxc.log").encode())
            self.assertEqual(plist["ProgramArguments"], args)
            self.assertTrue(plist["KeepAlive"] and plist["RunAtLoad"])
            self.assertTrue(str(ctxc._service_file()).endswith("Library/LaunchAgents/" + ctxc.LABEL + ".plist"))
        with mock.patch.object(sys, "platform", "linux"):
            unit = ctxc._service_text(args, "/tmp/ctxc.log")
            self.assertIn('ExecStart="/usr/bin/python3" "/opt/my tools/ctxc.py" "serve" "--port" "8787"', unit)
            self.assertIn("Restart=always", unit)

    def test_only_requests_addressed_to_localhost_are_served(self):
        up = "127.0.0.1:%d" % self.up.server_address[1]
        before = len(Upstream.seen)
        for path in ("/-/http/%s/v1/messages" % up, "/v1/messages", "/__ctxc/stats"):
            status, _, data = self.post(path, {"model": "m", "messages": []}, {"Host": "evil.example.com"})
            self.assertEqual(status, 403, path)
        self.assertEqual(len(Upstream.seen), before)                     # nothing was forwarded
        self.assertEqual(self.post("/v1/messages", {"model": "m", "stream": True, "messages": [
            {"role": "user", "content": "hi"}]}, {"Host": "localhost:%d" % self.port})[0], 200)

    def test_routing_gzip_and_fail_open(self):
        up = "127.0.0.1:%d" % self.up.server_address[1]
        chat = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(self.post("/v1/chat/completions", chat)[0], 200)
        self.assertEqual(Upstream.seen[-1][0], "/v1/chat/completions")          # /v1 not doubled
        self.assertEqual(self.post("/-/http/%s/api/coding/v1/responses" % up, {"model": "m", "stream": True, "input": "hi"})[0], 200)
        self.assertEqual(Upstream.seen[-1][0], "/api/coding/v1/responses")      # explicit upstream prefix
        gz = gzip.compress(json.dumps(chat).encode())
        self.assertEqual(self.post("/v1/chat/completions", None, {"Content-Encoding": "gzip"}, raw=gz)[0], 200)
        self.assertEqual(Upstream.seen[-1][2]["messages"][0]["content"], "hi")
        self.assertNotIn("Content-Encoding", Upstream.seen[-1][1])
        status, _, _ = self.post("/v1/messages", None, raw=b"{not json")        # forwarded untouched
        self.assertEqual(status, 400)
        self.assertEqual(Upstream.seen[-1][2], b"{not json")


if __name__ == "__main__":
    unittest.main(verbosity=2)
