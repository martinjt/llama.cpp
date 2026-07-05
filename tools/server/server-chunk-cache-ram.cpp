#include "server-chunk-cache.h"
#include <list>
#include <unordered_map>

namespace {

struct cache_entry {
    chunk_key key;
    std::vector<uint8_t> data;
};

class ram_chunk_cache_backend : public chunk_cache_backend {
public:
    explicit ram_chunk_cache_backend(size_t limit_mib)
        : limit_bytes(limit_mib * 1024ull * 1024ull) {}

    bool get(const chunk_key & key, std::vector<uint8_t> & data) override {
        auto it = index.find(key.content_hash);
        if (it == index.end()) {
            return false;
        }
        // move to back (most-recently-used) for LRU
        entries.splice(entries.end(), entries, it->second);
        data = it->second->data;
        return true;
    }

    void put(const chunk_key & key, std::vector<uint8_t> data) override {
        auto existing = index.find(key.content_hash);
        if (existing != index.end()) {
            total_bytes -= existing->second->data.size();
            entries.erase(existing->second);
            index.erase(existing);
        }

        total_bytes += data.size();
        entries.push_back(cache_entry{ key, std::move(data) });
        index[key.content_hash] = std::prev(entries.end());

        while (total_bytes > limit_bytes && !entries.empty()) {
            total_bytes -= entries.front().data.size();
            index.erase(entries.front().key.content_hash);
            entries.pop_front();
        }
    }

private:
    size_t limit_bytes;
    size_t total_bytes = 0;
    std::list<cache_entry> entries; // front = least-recently-used
    std::unordered_map<uint64_t, std::list<cache_entry>::iterator> index;
};

} // namespace

std::unique_ptr<chunk_cache_backend> make_ram_chunk_cache_backend(size_t limit_mib) {
    return std::make_unique<ram_chunk_cache_backend>(limit_mib);
}
