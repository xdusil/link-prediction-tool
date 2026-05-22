#pragma once

#include <string>

namespace utils {

inline void trim_in_place(std::string& value) {
    const std::size_t first = value.find_first_not_of(" \t");
    if (first == std::string::npos) {
        value.clear();
        return;
    }

    const std::size_t last = value.find_last_not_of(" \t");
    value = value.substr(first, last - first + 1);
}

} // namespace utils
