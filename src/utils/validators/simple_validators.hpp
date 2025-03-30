#pragma once

#include <string>
#include <utility>
#include <vector>

/**
 * @brief Custom validators for various types.
 *
 * These validators are used to check the validity of input values.
 * Each validator is a callable object that takes a value and returns a pair of
 * bool and string. The bool indicates whether the value is valid, and the string
 * contains an error message.
 */
namespace validators {

struct always_true {
    template <typename T>
    std::pair<bool, std::string> operator()(const T &) const;
};

struct is_positive {
    template <typename T>
    std::pair<bool, std::string> operator()(const T &v) const;
};

struct is_non_negative {
    template <typename T>
    std::pair<bool, std::string> operator()(const T &v) const;
};

struct is_unit_interval {
    template <typename T>
    std::pair<bool, std::string> operator()(const T &v) const;
};

struct is_open_unit_interval {
    template <typename T>
    std::pair<bool, std::string> operator()(const T &v) const;
};

struct is_not_empty_positive_vector {
    template <typename T>
    std::pair<bool, std::string> operator()(const std::vector<T> &v) const;
};

struct is_not_empty_non_negative_vector {
    template <typename T>
    std::pair<bool, std::string> operator()(const std::vector<T> &v) const;
};
} // namespace validators

#include "simple_validators.tpp"