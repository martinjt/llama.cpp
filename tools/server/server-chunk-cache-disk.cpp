#include "server-chunk-cache.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace {

constexpr size_t k_align = 4096; // O_DIRECT alignment for both buffer address and I/O size

std::string key_to_filename(const chunk_key & key) {
    std::ostringstream oss;
    oss << key.weights_fingerprint << "-" << std::hex << std::setw(16) << std::setfill('0') << key.content_hash << ".bin";
    return oss.str();
}

// O_DIRECT requires the memory buffer itself to be aligned (not just its size) on most
// filesystems -- a plain std::vector<uint8_t> only guarantees alignof(max_align_t), which is
// too weak. RAII wrapper around posix_memalign so get()/put() can early-return safely.
struct aligned_buffer {
    uint8_t * ptr = nullptr;
    size_t size = 0;

    explicit aligned_buffer(size_t n) {
        size = (n + k_align - 1) & ~(k_align - 1);
        if (posix_memalign((void **) &ptr, k_align, size) != 0) {
            ptr = nullptr;
        }
    }
    ~aligned_buffer() {
        free(ptr);
    }
    aligned_buffer(const aligned_buffer &) = delete;
    aligned_buffer & operator=(const aligned_buffer &) = delete;
};

class disk_chunk_cache_backend : public chunk_cache_backend {
public:
    disk_chunk_cache_backend(std::string dir, size_t limit_mib)
        : dir(std::move(dir)), limit_bytes(limit_mib * 1024ull * 1024ull) {}

    bool get(const chunk_key & key, std::vector<uint8_t> & data) override {
        const std::string path = dir + "/" + key_to_filename(key);
        // O_DIRECT to avoid growing the page cache on this unified-memory host -- same
        // mitigation already used for model loading, per docs/llm-strix-halo-setup.md.
        int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) {
            return false;
        }

        struct stat st;
        if (fstat(fd, &st) != 0) {
            close(fd);
            return false;
        }

        if (st.st_size == 0) {
            // Zero-byte cache entry (see put()): nothing to read, and this also
            // sidesteps posix_memalign(..., 0, ...)'s implementation-defined
            // behavior for a 0-byte request below.
            close(fd);
            data.clear();
            touch_mtime(path);
            return true;
        }

        aligned_buffer buf((size_t) st.st_size);
        if (!buf.ptr) {
            close(fd);
            return false;
        }
        // put() truncates the file down to its real, possibly-unaligned size, but we
        // still issue this read for the full 4096-aligned buffer size. Linux O_DIRECT
        // is documented to permit a read to return fewer bytes than requested when the
        // request runs past EOF, so `n == st.st_size` (not `n == buf.size`) is the
        // success case here -- a short read that stops exactly at st.st_size is
        // expected/relied-upon, not an error. This is Linux-specific behavior (not
        // guaranteed on all filesystems, e.g. some network filesystems), which is why
        // we still check `n < st.st_size` below rather than assuming a full read.
        ssize_t n = read(fd, buf.ptr, buf.size);
        close(fd);
        if (n < (ssize_t) st.st_size) {
            return false;
        }

        data.assign(buf.ptr, buf.ptr + st.st_size);
        touch_mtime(path);
        return true;
    }

    void put(const chunk_key & key, std::vector<uint8_t> data) override {
        const std::string path = dir + "/" + key_to_filename(key);
        const std::string tmp_path = path + ".tmp";

        if (data.empty()) {
            // Nothing to align or write via O_DIRECT for a zero-byte value, and
            // posix_memalign(..., 0, ...) below is implementation-defined: glibc may
            // legitimately return NULL with a "success" status for a 0-byte request,
            // which the `!buf.ptr` failure check further down can't distinguish from a
            // genuine allocation failure -- silently dropping the write. Skip the
            // O_DIRECT path entirely and just materialize an empty file directly.
            int fd = open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd < 0) {
                return;
            }
            close(fd);
            rename(tmp_path.c_str(), path.c_str());
            touch_mtime(path);
            evict_if_over_quota();
            return;
        }

        aligned_buffer buf(data.size());
        if (!buf.ptr) {
            return;
        }
        memcpy(buf.ptr, data.data(), data.size());
        memset(buf.ptr + data.size(), 0, buf.size - data.size());

        int fd = open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
        if (fd < 0) {
            return;
        }
        ssize_t written = write(fd, buf.ptr, buf.size);
        close(fd);
        if (written < (ssize_t) buf.size) {
            unlink(tmp_path.c_str());
            return;
        }

        // truncate back to the real size, then atomically rename into place.
        if (truncate(tmp_path.c_str(), (off_t) data.size()) != 0) {
            unlink(tmp_path.c_str());
            return;
        }
        rename(tmp_path.c_str(), path.c_str());
        touch_mtime(path);
        evict_if_over_quota();
    }

private:
    // mtime is used as the recency signal (rather than an in-memory LRU list, cf. the ram
    // backend) because the disk backend's whole purpose is surviving restarts: touched on
    // both get() hits and put() writes, then eviction scans the directory and sorts by mtime.
    //
    // utimensat(..., nullptr, ...) (UTIME_NOW) resolves to the kernel's *coarse* real-time
    // clock, which is only refreshed once per timer tick -- two touches issued within the same
    // tick (e.g. a put() immediately followed by another put()'s eviction scan) can come back
    // bit-for-bit identical even down to the nanosecond field, making ordering ambiguous for
    // std::sort (which is not stable). Sampling CLOCK_REALTIME from userspace and passing it
    // explicitly avoids the coarse-clock path and gives genuinely distinct timestamps.
    static void touch_mtime(const std::string & path) {
        struct timespec now;
        clock_gettime(CLOCK_REALTIME, &now);
        const struct timespec times[2] = { now, now };
        utimensat(AT_FDCWD, path.c_str(), times, 0);
    }

    void evict_if_over_quota() {
        // Nanosecond-resolution mtime (st_mtim, not the seconds-only st_mtime alias) matters
        // here: a touch_mtime() from get() and a put() elsewhere can legitimately land in the
        // same wall-clock second, and std::sort is not stable, so a seconds-only tiebreak can
        // evict the just-touched (recent) entry instead of the truly stale one.
        struct entry { std::string path; struct timespec mtime; size_t size; };
        std::vector<entry> entries;
        size_t total = 0;

        DIR * d = opendir(dir.c_str());
        if (!d) return;
        struct dirent * de;
        while ((de = readdir(d)) != nullptr) {
            if (de->d_name[0] == '.') continue;
            const std::string name = de->d_name;
            if (name.size() >= 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) {
                // in-flight write from a concurrent/interrupted put() -- not a real entry.
                continue;
            }
            const std::string path = dir + "/" + name;
            struct stat st;
            if (stat(path.c_str(), &st) != 0) continue;
            entries.push_back({path, st.st_mtim, (size_t) st.st_size});
            total += (size_t) st.st_size;
        }
        closedir(d);

        if (total <= limit_bytes) return;

        std::sort(entries.begin(), entries.end(), [](const entry & a, const entry & b) {
            if (a.mtime.tv_sec != b.mtime.tv_sec) return a.mtime.tv_sec < b.mtime.tv_sec;
            return a.mtime.tv_nsec < b.mtime.tv_nsec; // oldest (least-recently-touched) first
        });

        for (const auto & e : entries) {
            if (total <= limit_bytes) break;
            if (unlink(e.path.c_str()) == 0) {
                total -= e.size;
            }
        }
    }

    std::string dir;
    size_t limit_bytes;
};

} // namespace

std::unique_ptr<chunk_cache_backend> make_disk_chunk_cache_backend(const std::string & dir, size_t limit_mib) {
    return std::make_unique<disk_chunk_cache_backend>(dir, limit_mib);
}
