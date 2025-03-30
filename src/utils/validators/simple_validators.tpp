#pragma once

#include "simple_validators.hpp"
#include <algorithm>

namespace validators {

template <typename T>
std::pair<bool, std::string> always_true::operator()(const T &) const {
    return {true, ""};
}

template <typename T>
std::pair<bool, std::string> is_positive::operator()(const T &v) const {
    return {v > T(0), "must be positive"};
}

template <typename T>
std::pair<bool, std::string> is_non_negative::operator()(const T &v) const {
    return {v >= T(0), "must be non-negative"};
}

template <typename T>
std::pair<bool, std::string> is_unit_interval::operator()(const T &v) const {
    return {v >= T(0) && v <= T(1), "must be between 0 and 1"};
}

template <typename T>
std::pair<bool, std::string> is_open_unit_interval::operator()(const T &v) const {
    return {v > T(0) && v < T(1), "must be between 0 and 1 (exclusive)"};
}

template <typename T>
std::pair<bool, std::string>
is_not_empty_positive_vector::operator()(const std::vector<T> &v) const {
    return {!v.empty() &&
                std::all_of(v.begin(), v.end(), [](const T &x) { return x > T(0); }),
            "must be a non-empty vector of positive values"};
}

template <typename T>
std::pair<bool, std::string>
is_not_empty_non_negative_vector::operator()(const std::vector<T> &v) const {
    return {!v.empty() &&
                std::all_of(v.begin(), v.end(), [](const T &x) { return x >= T(0); }),
            "must be a non-empty vector of non-negative values"};
}

} // namespace validators