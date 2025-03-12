#pragma once

#include "boost/json/object.hpp"
#include <boost/json.hpp>
#include <optional>
#include <string>

namespace json = boost::json;

/**
 * @brief Helper class for extracting values from JSON objects.
 */
class JsonHelper {
public:
    template <typename T>
    inline static std::optional<T> extract_value(const json::object &data,
                                                 const std::string &key) {
        auto it = data.find(key);
        if (it != data.end() && is_type<T>(it->value())) {
            return json::value_to<T>(it->value());
        }
        return std::nullopt;
    }

    /**
     * @brief Extract an array of values from a JSON object.
     *
     * @tparam T The type of the array elements.
     * @param obj The JSON object.
     * @param key The key of the array.
     * @return The array of values if it exists, std::nullopt otherwise.
     */
    template <typename T>
    static std::optional<std::vector<T>> extract_array(const json::object &obj,
                                                       const std::string &key) {
        if (!obj.contains(key) || !obj.at(key).is_array()) {
            return std::nullopt;
        }

        std::vector<T> result;
        const json::array &arr = obj.at(key).as_array();

        for (const auto &elem : arr) {
            if constexpr (std::is_same_v<T, int64_t>) {
                if (is_type<int64_t>(elem)) {
                    result.push_back(elem.as_int64());
                }
            } else if constexpr (std::is_same_v<T, double>) {
                if (is_type<double>(elem)) {
                    result.push_back(elem.as_double());
                } else if (elem.is_int64()) {
                    result.push_back(static_cast<double>(elem.as_int64()));
                }
            } else if constexpr (std::is_same_v<T, bool>) {
                if (is_type<bool>(elem)) {
                    result.push_back(elem.as_bool());
                }
            }
        }

        return result;
    }

    /**
     * @brief Parse a JSON string into a JSON object.
     *
     * @param json The JSON string to parse.
     * @return The parsed JSON object.
     */
    inline static json::object parse_json(const std::string &json) {
        json::value jv = json::parse(json);
        return jv.as_object();
    }

private:
    /**
     * @brief Check if a JSON value is of a specific type.
     *
     * @tparam T The type to check for.
     * @param val The JSON value to check.
     * @return True if the value is of the specified type, false otherwise.
     */
    template <typename T>
    static bool is_type(const json::value &val);
};

// Specializations for is_type
template <>
inline bool JsonHelper::is_type<std::string>(const json::value &val) {
    return val.is_string();
}

template <>
inline bool JsonHelper::is_type<int64_t>(const json::value &val) {
    return val.is_int64();
}

template <>
inline bool JsonHelper::is_type<bool>(const json::value &val) {
    return val.is_bool();
}

template <>
inline bool JsonHelper::is_type<double>(const json::value &val) {
    return val.is_double();
}