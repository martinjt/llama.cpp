#include "llama-memory.h"

#include "llama-io.h"

llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? LLAMA_MEMORY_STATUS_SUCCESS : LLAMA_MEMORY_STATUS_NO_UPDATE;
}

size_t llama_memory_i::state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    GGML_ASSERT((p0 <= 0 && p1 < 0) && "this memory type only supports whole-sequence state save, not position ranges");
    state_write(io, seq_id, 0);
    return io.n_bytes();
}

size_t llama_memory_i::state_read_range(llama_io_read_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    GGML_ASSERT((p0 <= 0 && p1 < 0) && "this memory type only supports whole-sequence state load, not position ranges");
    state_read(io, seq_id, 0);
    return io.n_bytes();
}

bool llama_memory_status_is_fail(llama_memory_status status) {
    switch (status) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}
