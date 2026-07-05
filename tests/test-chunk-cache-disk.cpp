#include "server-chunk-cache.h"
#include <cstdio>
#include <filesystem>
#include <ctime>
#include <vector>

int main() {
    const std::string dir = "/tmp/chunk_cache_disk_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    auto backend = make_disk_chunk_cache_backend(dir, /*limit_mib=*/100);

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

    // A zero-byte value: put() must not silently no-op just because
    // posix_memalign(..., 0, ...) has implementation-defined behavior for a 0-byte
    // request (glibc may return NULL with "success" for that case). Confirms the
    // dedicated zero-byte path in put()/get() round-trips correctly.
    chunk_key k3{ "fp1", 333 };
    std::vector<uint8_t> data_in3;
    backend->put(k3, data_in3);
    std::vector<uint8_t> data_out4;
    if (!backend->get(k3, data_out4)) {
        fprintf(stderr, "FAIL: backend->get(k3, ...) missed right after put of a zero-byte value (expected a hit)\n");
        return 1;
    }
    if (!data_out4.empty()) {
        fprintf(stderr, "FAIL: data_out4 is not empty (expected get() to return 0 bytes for a zero-byte put())\n");
        return 1;
    }

    // A second backend instance pointed at the same directory sees the same data --
    // proving this is real disk persistence, not in-process caching.
    auto backend2 = make_disk_chunk_cache_backend(dir, /*limit_mib=*/100);
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

    // Eviction: with a small quota, confirm the least-recently-touched entry is evicted first,
    // and that get()-ing an entry refreshes its recency (mirrors the Task 6 LRU-vs-FIFO test).
    {
        const std::string dir2 = "/tmp/chunk_cache_disk_evict_test";
        std::filesystem::remove_all(dir2);
        std::filesystem::create_directories(dir2);

        auto evict_backend = make_disk_chunk_cache_backend(dir2, /*limit_mib=*/1);

        chunk_key ka{ "fp1", 111 };
        chunk_key kb{ "fp1", 222 };
        chunk_key kc{ "fp1", 333 };

        std::vector<uint8_t> half_mib(512 * 1024, 0xAA);

        evict_backend->put(ka, half_mib);
        // Ensure distinct mtimes -- filesystem mtime resolution can be coarse.
        struct timespec ts{1, 0};
        nanosleep(&ts, nullptr);
        evict_backend->put(kb, half_mib);

        std::vector<uint8_t> tmp;
        if (!evict_backend->get(ka, tmp)) {
            fprintf(stderr, "FAIL: ka should still be present before eviction is triggered\n");
            return 1;
        }
        nanosleep(&ts, nullptr);

        // ka is now most-recently-touched (via the get() above); kb is least-recently-touched.
        // Adding kc should trigger eviction of kb, not ka, if recency (not insertion order) governs.
        evict_backend->put(kc, half_mib);

        if (!evict_backend->get(ka, tmp)) {
            fprintf(stderr, "FAIL: ka (recently touched via get()) was evicted -- eviction is using insertion order, not recency\n");
            return 1;
        }
        if (evict_backend->get(kb, tmp)) {
            fprintf(stderr, "FAIL: kb (least-recently-touched) should have been evicted but is still present\n");
            return 1;
        }
        if (!evict_backend->get(kc, tmp)) {
            fprintf(stderr, "FAIL: kc (just inserted) should still be present\n");
            return 1;
        }

        std::filesystem::remove_all(dir2);
        printf("OK: disk chunk cache backend evicts least-recently-touched entry under quota, not FIFO\n");
    }

    return 0;
}
