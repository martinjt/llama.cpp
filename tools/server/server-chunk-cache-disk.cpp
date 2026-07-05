#include "server-chunk-cache.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <iomanip>

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
    explicit disk_chunk_cache_backend(std::string dir) : dir(std::move(dir)) {}

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
    }

private:
    std::string dir;
};

} // namespace

std::unique_ptr<chunk_cache_backend> make_disk_chunk_cache_backend(const std::string & dir) {
    return std::make_unique<disk_chunk_cache_backend>(dir);
}
