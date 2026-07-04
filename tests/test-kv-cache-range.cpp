#include "llama.h"
#include <cstdio>
#include <vector>

// Verifies that the position-range-scoped KV cache save/load round-trip preserves exact model
// state: saving cells [0, 32) of one sequence and restoring them into another produces logits
// bit-identical to recomputing that same range directly.
//
// IMPORTANT — why the reference is also decoded in two chunks:
//   Attention over the KV cache on the CPU / flash-attention backend is NOT invariant to batch
//   composition: decoding 64 tokens in a single batch yields slightly different logits than
//   decoding the same 64 tokens as 32+32, even with no save/restore involved at all (observed
//   ~0.04 mean-abs divergence). That is an inherent property of the backend, unrelated to this
//   feature. To isolate *round-trip fidelity* from that batch-composition noise, the reference
//   path uses the identical 32+32 chunking as the restore path. With that controlled, a faithful
//   save/load must reproduce the reference bit-for-bit.
int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (model == nullptr) { fprintf(stderr, "failed to load model %s\n", argv[1]); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_seq_max = 4;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) { fprintf(stderr, "failed to create context\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens(64);
    for (int i = 0; i < 64; ++i) {
        tokens[i] = llama_vocab_bos(vocab) != LLAMA_TOKEN_NULL && i == 0
            ? llama_vocab_bos(vocab)
            : (i % (n_vocab - 1)) + 1; // deterministic filler tokens
    }

    // A second, deliberately *different* token stream covering positions [0, 32), used to poison
    // seq 1's cells before the restore runs (see the restore path below for why).
    std::vector<llama_token> wrong_tokens(32);
    for (int i = 0; i < 32; ++i) {
        wrong_tokens[i] = (tokens[i] + n_vocab / 2) % n_vocab;
    }

    auto fill_batch = [&](llama_batch & b, const std::vector<llama_token> & src, int lo, int hi, llama_seq_id seq) {
        int n = 0;
        for (int i = lo; i < hi; ++i) {
            b.token[n]     = src[i];
            b.pos[n]       = i;
            b.n_seq_id[n]  = 1;
            b.seq_id[n][0] = seq;
            b.logits[n]    = (i == hi - 1);
            ++n;
        }
        b.n_tokens = n;
    };

    // Reference path (sequence 0), chunked 32+32: decode [0,32), then [32,64).
    llama_batch ref_head = llama_batch_init(32, 0, 1);
    fill_batch(ref_head, tokens, 0, 32, 0);
    if (llama_decode(ctx, ref_head) != 0) { fprintf(stderr, "ref_head decode failed\n"); return 1; }

    llama_batch ref_tail = llama_batch_init(32, 0, 1);
    fill_batch(ref_tail, tokens, 32, 64, 0);
    if (llama_decode(ctx, ref_tail) != 0) { fprintf(stderr, "ref_tail decode failed\n"); return 1; }

    const float * logits_ref = llama_get_logits_ith(ctx, -1);
    std::vector<float> logits_ref_copy(logits_ref, logits_ref + n_vocab);

    // Save only the range [0, 32) of sequence 0 -- simulating a cached "chunk". These cells were
    // produced by ref_head, the same 32-token batch the reference used for its first half.
    const size_t range_size = llama_state_seq_get_size_range(ctx, /*seq_id=*/0, /*p0=*/0, /*p1=*/32);
    if (range_size == 0) { fprintf(stderr, "range size is zero\n"); return 1; }
    std::vector<uint8_t> range_buf(range_size);
    const size_t written = llama_state_seq_get_data_range(ctx, range_buf.data(), range_size, 0, 0, 32);
    if (written != range_size) { fprintf(stderr, "written %zu != range_size %zu\n", written, range_size); return 1; }

    // Clear sequence 0 so the restore below reuses freed cells.
    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);

    // Restore path (sequence 1): first decode *different* filler tokens into positions [0,32) to
    // allocate those cells with K/V content that is deliberately wrong, then call
    // llama_state_seq_set_data_range to overwrite that range with the saved chunk, then decode the
    // uncached tail [32,64) on top of the restored range.
    //
    // Why decode wrong content first instead of the real head tokens: if we pre-decoded the *same*
    // tokens at the *same* positions that produced the saved chunk, seq 1's cells would already be
    // bit-identical to the saved K/V before the restore call even runs (K/V for a given
    // token+position is deterministic and independent of seq_id/batch composition). A restore that
    // silently no-ops -- reads and discards the buffer without writing anything -- would then still
    // leave the correct content in place and the test would pass despite the bug. Poisoning the
    // cells with different tokens first means the correct final state can only be reached if the
    // restore genuinely overwrites; a no-op or partial restore leaves the wrong K/V behind, which
    // changes the tail's attention output and fails the logit comparison below.
    llama_batch head = llama_batch_init(32, 0, 1);
    fill_batch(head, wrong_tokens, 0, 32, 1);
    for (int i = 0; i < 32; ++i) {
        head.logits[i] = false;
    }
    if (llama_decode(ctx, head) != 0) { fprintf(stderr, "head decode failed\n"); return 1; }

    const size_t nread = llama_state_seq_set_data_range(ctx, range_buf.data(), range_size, 1, 0, 32);
    if (nread != range_size) { fprintf(stderr, "nread %zu != range_size %zu\n", nread, range_size); return 1; }

    llama_batch tail = llama_batch_init(32, 0, 1);
    fill_batch(tail, tokens, 32, 64, 1);
    if (llama_decode(ctx, tail) != 0) { fprintf(stderr, "tail decode failed\n"); return 1; }

    const float * logits_spliced = llama_get_logits_ith(ctx, -1);

    for (int i = 0; i < n_vocab; ++i) {
        if (logits_ref_copy[i] != logits_spliced[i]) {
            fprintf(stderr, "logit mismatch at %d: ref=%.9g spliced=%.9g\n",
                    i, logits_ref_copy[i], logits_spliced[i]);
            return 1;
        }
    }

    llama_batch_free(ref_head);
    llama_batch_free(ref_tail);
    llama_batch_free(head);
    llama_batch_free(tail);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf("OK: range save/load round-trip produced identical logits\n");
    return 0;
}
