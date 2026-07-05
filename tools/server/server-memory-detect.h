#pragma once

struct llama_context;

// True if ctx's memory is plain attention (llama_kv_cache or llama_kv_cache_iswa), the only
// memory types the secondary range-chunk cache path (compute_chunk_chain +
// llama_state_seq_get_data_range/set_data_range) is valid for.
//
// Deliberately an allowlist, not a blocklist keyed on e.g. "recurrent": llama_memory_hybrid and
// llama_memory_hybrid_iswa wrap a recurrent sub-cache but don't have "recurrent" in their own
// name, so a blocklist would silently admit them. Every model actually served in production
// today (qwen35 / qwen35moe) is hybrid, so this allowlist currently exists to keep this path
// switched off there while remaining correct if a plain-attention model is ever added.
bool model_supports_range_cache(llama_context * ctx);
