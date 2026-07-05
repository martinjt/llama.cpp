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

    // 96 tokens: the first 64 feed the original single-chunk-restore scenario below: the last 32
    // (positions [64, 96)) are only used by the multi-chunk-restore scenario further down, which
    // needs a 3rd chunk to extend the reference past two chained restores.
    std::vector<llama_token> tokens(96);
    for (int i = 0; i < 96; ++i) {
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

    // Reference path (sequence 0), chunked 32+32+32: decode [0,32), then [32,64), then [64,96).
    // The third chunk is only consumed by scenario 2 further down, but must be decoded here,
    // before sequence 0 gets cleared below, since scenario 2 also captures its "chunk 1" saved
    // range from this same seq 0 / same decode (capturing after the clear would capture nothing).
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

    // For scenario 2 (chained multi-chunk restore, further down): also save the range [32, 64) --
    // "chunk 1" -- from this same seq 0 / same decode, while its cells still exist (seq 0 gets
    // cleared right below, for scenario 1's own purposes).
    const size_t range_size_c1 = llama_state_seq_get_size_range(ctx, /*seq_id=*/0, /*p0=*/32, /*p1=*/64);
    if (range_size_c1 == 0) { fprintf(stderr, "chunk1 range size is zero\n"); return 1; }
    std::vector<uint8_t> range_buf_c1(range_size_c1);
    const size_t written_c1 = llama_state_seq_get_data_range(ctx, range_buf_c1.data(), range_size_c1, 0, 32, 64);
    if (written_c1 != range_size_c1) { fprintf(stderr, "written_c1 %zu != range_size_c1 %zu\n", written_c1, range_size_c1); return 1; }

    // Continue the reference decode to 96 tokens (chunk 2, [64, 96)) for scenario 2's comparison,
    // still on seq 0, still before it gets cleared.
    llama_batch ref_chunk2 = llama_batch_init(32, 0, 1);
    fill_batch(ref_chunk2, tokens, 64, 96, 0);
    if (llama_decode(ctx, ref_chunk2) != 0) { fprintf(stderr, "ref_chunk2 decode failed\n"); return 1; }

    const float * logits_ref2 = llama_get_logits_ith(ctx, -1);
    std::vector<float> logits_ref2_copy(logits_ref2, logits_ref2 + n_vocab);

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

    printf("OK: range save/load round-trip produced identical logits\n");

    // --- Scenario 2: chained multi-chunk restore into a genuinely fresh (never-decoded) seq_id.
    //
    // This is the scenario the server's range-chunk restore loop (get_available_slot() in
    // tools/server/server-context.cpp) actually exercises: multiple llama_state_seq_set_data_range
    // calls into the *same* dest_seq_id, at increasing non-overlapping position ranges, with no
    // decode (and no llama_memory_seq_cp) into that seq_id at any point -- exactly the "genuinely
    // fresh slot after a server restart" case flagged as unverified in review. Two things must
    // both hold for this to be correct:
    //   1. set_data_range must work at all against a seq_id with zero prior cells (not just one
    //      that was pre-decoded into, like seq 1 above).
    //   2. a *second* set_data_range call into that same seq_id, at a disjoint range, must not
    //      destroy the cells the *first* call just established. (It did, before this fix:
    //      llama_kv_cache::state_read_meta unconditionally cleared dest_seq_id's *entire*
    //      sequence -- not just [p0, p1) -- before writing each restored range, so only the last
    //      chunk in a chain would actually survive.)
    //
    // range_buf / range_buf_c1 (chunks 0 and 1) and logits_ref2_copy (the 96-token cold reference)
    // were all captured above, before sequence 0 was cleared for scenario 1's own purposes.

    // seq 2 has never been decoded into or touched in any way -- no batch, no seq_cp, nothing.
    const llama_seq_id fresh_seq = 2;

    const size_t nread_c0 = llama_state_seq_set_data_range(ctx, range_buf.data(), range_buf.size(), fresh_seq, 0, 32);
    if (nread_c0 != range_buf.size()) { fprintf(stderr, "nread_c0 %zu != %zu\n", nread_c0, range_buf.size()); return 1; }

    // The call that used to destroy chunk 0's just-established cells for fresh_seq: before the
    // fix, this made llama_memory_seq_pos_min(fresh_seq) jump from 0 to 32 (chunk 0's cells
    // silently gone) even though this call itself reports success.
    const size_t nread_c1 = llama_state_seq_set_data_range(ctx, range_buf_c1.data(), range_buf_c1.size(), fresh_seq, 32, 64);
    if (nread_c1 != range_buf_c1.size()) { fprintf(stderr, "nread_c1 %zu != %zu\n", nread_c1, range_buf_c1.size()); return 1; }

    if (llama_memory_seq_pos_min(llama_get_memory(ctx), fresh_seq) != 0 ||
        llama_memory_seq_pos_max(llama_get_memory(ctx), fresh_seq) != 63) {
        fprintf(stderr, "chained restore left wrong cell range for fresh_seq: pos_min=%d pos_max=%d (expected 0, 63)\n",
                llama_memory_seq_pos_min(llama_get_memory(ctx), fresh_seq),
                llama_memory_seq_pos_max(llama_get_memory(ctx), fresh_seq));
        return 1;
    }

    llama_batch tail2 = llama_batch_init(32, 0, 1);
    fill_batch(tail2, tokens, 64, 96, fresh_seq);
    if (llama_decode(ctx, tail2) != 0) { fprintf(stderr, "tail2 decode failed\n"); return 1; }

    const float * logits_chained = llama_get_logits_ith(ctx, -1);

    for (int i = 0; i < n_vocab; ++i) {
        if (logits_ref2_copy[i] != logits_chained[i]) {
            fprintf(stderr, "chained-restore logit mismatch at %d: ref=%.9g chained=%.9g\n",
                    i, logits_ref2_copy[i], logits_chained[i]);
            return 1;
        }
    }

    llama_batch_free(ref_chunk2);
    llama_batch_free(tail2);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf("OK: chained multi-chunk restore into a fresh seq_id produced identical logits\n");
    return 0;
}
