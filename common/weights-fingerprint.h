#pragma once
#include <string>

// Returns a stable, hex-encoded 64-bit FNV-1a hash of the file's contents at model_path.
// Used to key cache entries by the model's actual weights rather than its human-assigned
// preset name, so a silent weights-file swap under the same preset name is never served a
// stale cache hit.
std::string common_weights_fingerprint(const std::string & model_path);
