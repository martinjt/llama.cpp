import os
import shutil
import tempfile
import pytest
from utils import *

# this build has no HTTPS, so use the model file already checked out under models/
MODEL_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models", "stories260K.gguf")
)

server = ServerPreset.tinyllama2()


# override conftest's module-autouse load_all(): this build has no HTTPS so it cannot
# download the preset models; we supply the local model file directly instead.
@pytest.fixture(scope="module", autouse=True)
def do_something():
    yield


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.offline = False
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = MODEL_FILE
    server.n_slots = 1
    server.n_ctx = 4096
    server.n_batch = 256  # > snapshot-step (64) so prefill batch-boundary logic is exercised
    server.n_predict = 4
    server.temperature = 0.0
    server.seed = 42
    server.server_slots = True
    server.kv_unified = True
    # keep the in-memory prompt cache OUT of the way so any reuse we observe
    # across a full process restart can only come from the on-disk snapshot path
    server.cache_ram = 0
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


# a prompt long enough to span several snapshot-step (64) intervals, while staying
# within the tiny test model's 2048-token training context
LONG_PROMPT = "The quick brown fox " * 100


def test_chunk_cache_snapshot_reuse_across_restart():
    """After a full server restart, an identical prompt must show cache reuse that
    can only come from the externally-persisted (disk) snapshot -- the in-memory
    prompt cache is disabled and is gone after restart anyway."""
    global server
    cache_dir = tempfile.mkdtemp(prefix="cc-restart-")
    try:
        server.chunk_cache_backend = "disk"
        server.chunk_cache_path = cache_dir
        server.chunk_cache_snapshot_step = 64
        server.start()

        res1 = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT, "n_predict": 1, "cache_prompt": True,
        })
        assert res1.status_code == 200
        cached_before = res1.body["timings"]["cache_n"]

        server.stop()

        # fresh process, same on-disk cache dir
        server.start()
        res2 = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT, "n_predict": 1, "cache_prompt": True,
        })
        assert res2.status_code == 200
        cached_after = res2.body["timings"]["cache_n"]

        # first run had a cold cache; second run (post-restart) must reuse the
        # snapshot written during the first run
        assert cached_before == 0, f"expected cold first run, got cache_n={cached_before}"
        assert cached_after > 0, f"expected snapshot reuse after restart, got cache_n={cached_after}"
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_chunk_cache_snapshot_reuse_growing_conversation():
    """The core 'conversation grows by one turn' benefit: a longer prompt sharing a
    prefix with an earlier one must reuse the earlier prefix's snapshot instead of
    recomputing it from scratch."""
    global server
    cache_dir = tempfile.mkdtemp(prefix="cc-growing-")
    try:
        server.chunk_cache_backend = "disk"
        server.chunk_cache_path = cache_dir
        server.chunk_cache_snapshot_step = 64
        server.start()

        res1 = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT, "n_predict": 1, "cache_prompt": True,
        })
        assert res1.status_code == 200

        # same slot is still warm in-memory, so to prove the SNAPSHOT is doing the
        # work we restart the process, dropping all in-process KV/prompt state.
        server.stop()
        server.start()

        grown_prompt = LONG_PROMPT + "jumps over the lazy dog. "
        res2 = server.make_request("POST", "/completion", data={
            "prompt": grown_prompt, "n_predict": 1, "cache_prompt": True,
        })
        assert res2.status_code == 200
        cached = res2.body["timings"]["cache_n"]
        assert cached > 0, f"expected shared-prefix snapshot reuse, got cache_n={cached}"
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_chunk_cache_snapshot_restore_is_bit_correct():
    """A continuation served from a restored snapshot must produce EXACTLY the same
    generated tokens as a cold full recompute of the same prompt. This is the
    dangerous failure mode: a subtly-wrong restore would silently corrupt output
    rather than crash."""
    global server

    # 1) cold baseline: no chunk cache at all, generate deterministically
    server.temperature = 0.0
    server.start()
    baseline = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT, "n_predict": 8, "cache_prompt": True, "temperature": 0.0,
    })
    assert baseline.status_code == 200
    baseline_content = baseline.body["content"]
    server.stop()

    # 2) with chunk cache: first request warms the on-disk snapshot, restart to force
    #    the second request to restore from disk, then compare the continuation.
    cache_dir = tempfile.mkdtemp(prefix="cc-correct-")
    try:
        server.chunk_cache_backend = "disk"
        server.chunk_cache_path = cache_dir
        server.chunk_cache_snapshot_step = 64
        server.start()
        warm = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT, "n_predict": 8, "cache_prompt": True, "temperature": 0.0,
        })
        assert warm.status_code == 200
        server.stop()

        server.start()
        restored = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT, "n_predict": 8, "cache_prompt": True, "temperature": 0.0,
        })
        assert restored.status_code == 200
        assert restored.body["timings"]["cache_n"] > 0, "expected snapshot restore on this request"
        assert restored.body["content"] == baseline_content, (
            "restored continuation diverged from cold recompute:\n"
            f"  baseline: {baseline_content!r}\n"
            f"  restored: {restored.body['content']!r}"
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_chunk_cache_opt_out_header():
    """The x-chunk-cache: off header must bypass the chunk-cache lookup entirely,
    forcing a cold recompute even when a cached snapshot exists.

    Note: the server must be restarted between the warm-up request and the
    opt-out request. Without a restart, the *same* slot stays resident in memory
    and llama-server's ordinary same-process slot reuse (independent of the
    on-disk chunk cache under test here) would also produce cache_n > 0,
    making the assertion meaningless either way it comes out."""
    global server
    cache_dir = tempfile.mkdtemp(prefix="cc-opt-out-")
    try:
        server.chunk_cache_backend = "disk"
        server.chunk_cache_path = cache_dir
        server.chunk_cache_snapshot_step = 64
        server.start()

        prompt = LONG_PROMPT  # 200 reps exceeded the tiny model's 2048-token ctx; use the already-safe 100-rep constant

        # First request: warm up the on-disk cache
        res1 = server.make_request("POST", "/completion", data={
            "prompt": prompt, "n_predict": 1
        })
        assert res1.status_code == 200

        # drop all in-process slot/KV state so any reuse below can only come
        # from the externally-persisted chunk cache
        server.stop()
        server.start()

        # sanity control: without the header, a fresh process DOES restore from
        # the on-disk snapshot -- proving one is genuinely there to opt out of
        control = server.make_request("POST", "/completion", data={
            "prompt": prompt, "n_predict": 1
        })
        assert control.status_code == 200
        assert control.body["timings"]["cache_n"] > 0, (
            "sanity check failed: expected a snapshot to exist on disk to opt out of"
        )

        server.stop()
        server.start()

        # Same prompt, but with the opt-out header: must NOT reuse the snapshot
        # that the sanity control just proved exists.
        res2 = server.make_request(
            "POST", "/completion",
            data={"prompt": prompt, "n_predict": 1},
            headers={"x-chunk-cache": "off"},
        )
        assert res2.status_code == 200
        cache_n = res2.body.get("timings", {}).get("cache_n", 0)
        assert cache_n == 0, (
            f"expected opt-out header to bypass cache (cache_n=0), "
            f"but got cache_n={cache_n}"
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_chunk_cache_opt_out_header_disables_capture():
    """The x-chunk-cache: off header must also disable CAPTURING this request's
    prompt prefix into the chunk cache, not just restoring from it. A request sent
    with the header must leave the on-disk cache untouched, and a later request for
    the identical prompt (without the header, after a restart to drop in-process
    slot state) must find nothing to reuse -- proving the capture-side gate
    (update_slots()), not just the restore-side gate (get_available_slot()),
    honored the opt-out."""
    global server
    cache_dir = tempfile.mkdtemp(prefix="cc-opt-out-capture-")
    try:
        server.chunk_cache_backend = "disk"
        server.chunk_cache_path = cache_dir
        server.chunk_cache_snapshot_step = 64
        server.start()

        prompt = LONG_PROMPT  # 200 reps exceeded the tiny model's 2048-token ctx; use the already-safe 100-rep constant

        # Request sent with the opt-out header: must not write anything to the
        # on-disk cache.
        res1 = server.make_request(
            "POST", "/completion",
            data={"prompt": prompt, "n_predict": 1},
            headers={"x-chunk-cache": "off"},
        )
        assert res1.status_code == 200

        cache_files_after_opt_out = os.listdir(cache_dir)
        assert cache_files_after_opt_out == [], (
            "expected opt-out request to write nothing to the on-disk chunk "
            f"cache, but found: {cache_files_after_opt_out}"
        )

        # drop all in-process slot/KV state: a subsequent identical request can
        # then only show reuse if the on-disk chunk cache holds something, which
        # it must not, since the only request so far was opted out.
        server.stop()
        server.start()

        res2 = server.make_request("POST", "/completion", data={
            "prompt": prompt, "n_predict": 1,
        })
        assert res2.status_code == 200
        cache_n = res2.body.get("timings", {}).get("cache_n", 0)
        assert cache_n == 0, (
            "expected no cache reuse for a fresh request following an opted-out "
            f"capture, but got cache_n={cache_n}"
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_chunk_cache_opt_out_header_case_insensitive_value():
    """The opt-out header's VALUE must be matched case-insensitively: 'Off', 'OFF',
    'oFf', etc. must all opt out, not just the exact lowercase 'off'. Uses a mixed
    case HEADER NAME too ("X-Chunk-Cache"), since server_http_req::headers is a
    plain case-sensitive std::map and a naive std::map::find("x-chunk-cache")
    would otherwise miss this common on-the-wire capitalization entirely."""
    global server
    cache_dir = tempfile.mkdtemp(prefix="cc-opt-out-case-")
    try:
        server.chunk_cache_backend = "disk"
        server.chunk_cache_path = cache_dir
        server.chunk_cache_snapshot_step = 64
        server.start()

        prompt = LONG_PROMPT  # 200 reps exceeded the tiny model's 2048-token ctx; use the already-safe 100-rep constant

        # First request: warm up the on-disk cache
        res1 = server.make_request("POST", "/completion", data={
            "prompt": prompt, "n_predict": 1
        })
        assert res1.status_code == 200

        # drop all in-process slot/KV state so any reuse below can only come
        # from the externally-persisted chunk cache
        server.stop()
        server.start()

        # Same prompt, opt-out header with mixed-case name AND value
        res2 = server.make_request(
            "POST", "/completion",
            data={"prompt": prompt, "n_predict": 1},
            headers={"X-Chunk-Cache": "OFF"},
        )
        assert res2.status_code == 200
        cache_n = res2.body.get("timings", {}).get("cache_n", 0)
        assert cache_n == 0, (
            "expected case-insensitive opt-out header (name and value) to bypass "
            f"cache (cache_n=0), but got cache_n={cache_n}"
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
