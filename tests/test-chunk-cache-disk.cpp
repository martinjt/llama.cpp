#include "server-chunk-cache.h"
#include <cstdio>
#include <filesystem>

int main() {
    const std::string dir = "/tmp/chunk_cache_disk_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    auto backend = make_disk_chunk_cache_backend(dir);

    chunk_key k1{ "fp1", 111 };
    std::vector<uint8_t> data_out;
    if (backend->get(k1, data_out)) {
        fprintf(stderr, "FAIL: backend->get(k1, ...) hit on an empty cache (expected a miss)\n");
        return 1;
    }

    // 4096 = a plausible O_DIRECT-aligned size
    std::vector<uint8_t> data_in(4096, 0x42);
    backend->put(k1, data_in);

    if (!backend->get(k1, data_out)) {
        fprintf(stderr, "FAIL: backend->get(k1, ...) missed right after put (expected a hit)\n");
        return 1;
    }
    if (data_out != data_in) {
        fprintf(stderr, "FAIL: data_out != data_in (expected get() to return the exact bytes that were put())\n");
        return 1;
    }

    // A non-4096-aligned size, to make sure the O_DIRECT alignment padding/truncation logic
    // doesn't corrupt or leak past the real length.
    chunk_key k2{ "fp1", 222 };
    std::vector<uint8_t> data_in2(4000, 0x7A);
    backend->put(k2, data_in2);
    std::vector<uint8_t> data_out2;
    if (!backend->get(k2, data_out2)) {
        fprintf(stderr, "FAIL: backend->get(k2, ...) missed right after put (expected a hit)\n");
        return 1;
    }
    if (data_out2 != data_in2) {
        fprintf(stderr, "FAIL: data_out2 != data_in2 (expected get() to return the exact bytes that were put(), not padding)\n");
        return 1;
    }

    // A second backend instance pointed at the same directory sees the same data --
    // proving this is real disk persistence, not in-process caching.
    auto backend2 = make_disk_chunk_cache_backend(dir);
    std::vector<uint8_t> data_out3;
    if (!backend2->get(k1, data_out3)) {
        fprintf(stderr, "FAIL: backend2->get(k1, ...) missed (expected persistence across backend instances)\n");
        return 1;
    }
    if (data_out3 != data_in) {
        fprintf(stderr, "FAIL: data_out3 != data_in (expected persisted bytes to match what was put())\n");
        return 1;
    }

    std::filesystem::remove_all(dir);
    printf("OK: disk chunk cache backend persists across instances\n");
    return 0;
}
