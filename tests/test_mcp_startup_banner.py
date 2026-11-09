"""The one line the MCP server announces itself with, and the channel it goes to.

A stdio MCP server spends its stdout on JSON-RPC, so the announcement a client shows on the
terminal (dsh inherits the child's stderr through the MCP SDK's ``StdioClientTransport``
default) has to be written to stderr. That is what this suite pins: one line, on stderr,
stating which Laya answers -- local with checkpoints built on demand, local with a preload
list, or remote -- and nothing at all on stdout that a client could mistake for a message.

It also pins the cost of asking. The banner must not import torch: ``LAYA_PRELOAD=0`` keeps
startup cheap, and reading the device through ``resolve_device`` would undo that, so what is
reported is the ``LAYA_DEVICE`` preference as written rather than the device a loaded agent
computes on.

Requires the mcp extra:  pip install "laya[mcp]"
Run: python tests/test_mcp_startup_banner.py
"""
import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import mcp  # noqa: F401
except ImportError:
    print("SKIP: mcp extra not installed (pip install 'laya[mcp]')")
    sys.exit(0)

from laya.mcp import server as server_mod  # noqa: E402

PASS, FAIL = [], []

# Every variable `_startup_banner` reads, so each case starts from one clean environment.
_ENV_KEYS = ("LAYA_DEVICE", "LAYA_PRELOAD", "LAYA_MODELS", "LAYA_BASE_URL")


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append("%s%s" % (name, (" -- " + detail) if detail and not cond else ""))
    print("   %s %s%s" % ("PASS" if cond else "FAIL", name, ("  " + detail) if detail else ""), flush=True)


def banner_with(**env):
    """`_startup_banner()` under exactly `env` (unlisted keys unset), then the old environment back."""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    try:
        for key in _ENV_KEYS:
            os.environ.pop(key, None)
        os.environ.update(env)
        return server_mod._startup_banner()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_lazy_mode():
    line = banner_with(LAYA_PRELOAD="0", LAYA_DEVICE="cuda")
    ok("banner/lazy_names_the_variable", "checkpoints load on first call (LAYA_PRELOAD=0)" in line, line)
    ok("banner/lazy_does_not_claim_preload", "preloading" not in line, line)
    ok("banner/lazy_reports_device", "device preference: cuda" in line, line)
    ok("banner/prefix_and_version", line.startswith("[laya-mcp] ready \u2014 Laya "), line)
    ok("banner/single_line", "\n" not in line, line)


def test_preload_mode():
    line = banner_with()
    ok("banner/preload_default_list",
       "preloading english, multilingual; any other checkpoint loads on first call" in line, line)
    ok("banner/preload_omits_unset_device", "device preference" not in line, line)

    line = banner_with(LAYA_MODELS="english")
    ok("banner/preload_honours_models", "preloading english;" in line, line)

    # A false-ish LAYA_PRELOAD is the lazy branch, not the preload branch.
    line = banner_with(LAYA_PRELOAD="false")
    ok("banner/preload_false_is_lazy", "checkpoints load on first call" in line, line)


def test_remote_mode():
    line = banner_with(LAYA_BASE_URL="http://127.0.0.1:8000", LAYA_DEVICE="cuda")
    ok("banner/remote_names_origin",
       "remote mode: answering from laya-serve at http://127.0.0.1:8000" in line, line)
    # No checkpoint is built in this process, so a device preference is not a fact about it.
    ok("banner/remote_omits_device", "device preference" not in line, line)
    ok("banner/remote_omits_preload", "preloading" not in line and "first call" not in line, line)

    line = banner_with(LAYA_BASE_URL="127.0.0.1:8000")
    ok("banner/remote_accepts_bare_host_port",
       "laya-serve at http://127.0.0.1:8000" in line, line)


def test_banner_does_not_import_torch():
    """Cheap startup is the reason LAYA_PRELOAD=0 exists; the banner must not spend it."""
    torch = sys.modules.pop("torch", None)
    try:
        banner_with(LAYA_PRELOAD="0", LAYA_DEVICE="cuda")  # env_device would parse this via torch
        ok("banner/no_torch_import", "torch" not in sys.modules,
           "torch was imported: %s" % sorted(m for m in sys.modules if m.startswith("torch"))[:5])
    finally:
        if torch is not None:
            sys.modules["torch"] = torch


def test_main_writes_it_to_stderr_only():
    """The channel contract, against `main()` itself rather than the helper."""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    stdout, stderr = io.StringIO(), io.StringIO()
    calls = []
    real_run = server_mod.server.run
    try:
        for key in _ENV_KEYS:
            os.environ.pop(key, None)
        os.environ["LAYA_PRELOAD"] = "0"  # no weights: this only exercises the banner
        os.environ["LAYA_DEVICE"] = "cuda"
        server_mod.server.run = lambda: calls.append(True)  # return instead of serving
        with redirect_stdout(stdout), redirect_stderr(stderr):
            server_mod.main()
    finally:
        server_mod.server.run = real_run
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    err = stderr.getvalue()
    ok("main/banner_on_stderr", err.startswith("[laya-mcp] ready \u2014 Laya "), repr(err))
    ok("main/one_line", len(err.strip().splitlines()) == 1, repr(err))
    ok("main/stdout_untouched", stdout.getvalue() == "", repr(stdout.getvalue()))
    ok("main/serves_after_the_banner", calls == [True], repr(calls))


test_lazy_mode()
test_preload_mode()
test_remote_mode()
test_banner_does_not_import_torch()
test_main_writes_it_to_stderr_only()

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all mcp startup-banner tests passed")
sys.exit(1 if FAIL else 0)
