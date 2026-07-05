#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// A content-addressed cache key. Used by both the primary snapshot path (Task 12, key = hash of
// the full token prefix up to a snapshot boundary) and the secondary range-chunk path (Task 13,
// key = chained hash of a fixed-size token chunk). The backend itself doesn't care which
// produced a given key/blob pair -- both are just opaque bytes.
struct chunk_key {
    std::string weights_fingerprint;
    uint64_t    content_hash;
};

class chunk_cache_backend {
public:
    virtual ~chunk_cache_backend() = default;
    // Returns true and fills `data` if a blob with this exact key exists.
    virtual bool get(const chunk_key & key, std::vector<uint8_t> & data) = 0;
    // Stores data for this key. May evict older entries per the backend's own policy.
    virtual void put(const chunk_key & key, std::vector<uint8_t> data) = 0;
};

std::unique_ptr<chunk_cache_backend> make_ram_chunk_cache_backend(size_t limit_mib);
