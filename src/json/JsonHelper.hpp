#pragma once

#include "exceptions/exceptions.hpp"
#include <boost/json.hpp>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace json = boost::json;

/**
 * @brief Helper class for JSON handling.
 */
class JsonHelper {
public:
    /**
     * @brief Parse a JSON string into a JSON object.
     *
     * @param json_str The JSON string to parse.
     * @return The parsed JSON object.
     * @throws JSONException if the error occurs during parsing.
     */
    static json::object parse_json(const std::string& json_str,
                                   const json::parse_options& opt = {}) {
        try {
            json::value jv = json::parse(json_str, /*storage_ptr*/ {}, opt);
            return jv.as_object();
        } catch (const boost::system::system_error& e) {
            const std::string error_msg = "JSON parse error";
            if (e.code() == boost::json::error::syntax) {
                throw JSONException(error_msg + ": " + "syntax error");
            }
            throw JSONException(error_msg);
        }
    }

    /**
     * @brief Parse a JSON value into a JSON object.
     *
     * @param jv The JSON value to parse.
     * @return The parsed JSON object.
     */
    static json::object parse_json_value(const json::value& jv) { return jv.as_object(); }

    /**
     * @brief Extract a value from a JSON object.
     *
     * @tparam T The type of the value to extract.
     * @tparam ThrowOnConvertError Whether to throw an exception on conversion error.
     * @param data The JSON object.
     * @param key The key of the value to extract.
     * @return The extracted value if it exists and is of the correct type, std::nullopt
     * otherwise.
     * @throws JSONException if the value exists but is of the wrong type - only if
     * ThrowOnConvertError is true.
     */
    template <typename T, bool ThrowOnConvertError = false>
    static std::optional<T> extract_value(const json::object& data,
                                          const std::string& key) {
        auto it = data.find(key);
        if (it == data.end()) {
            return std::nullopt;
        }

        return try_convert<T, ThrowOnConvertError>(it->value());
    }

    /**
     * @brief Extract an array of values from a JSON object.
     *
     * @tparam T The type of the array elements.
     * @tparam ThrowOnConvertError Whether to throw an exception on conversion error.
     * @param obj The JSON object.
     * @param key The key of the array.
     * @return The array of values if it exists, std::nullopt otherwise.
     * @throws JSONException if the array exists but contains invalid values or is of the
     * wrong type - only if ThrowOnConvertError is true.
     */
    template <typename T, bool ThrowOnConvertError = false>
    static std::optional<std::vector<T>> extract_array(const json::object& obj,
                                                       const std::string& key) {
        auto it = obj.find(key);
        if (it == obj.end()) {
            return std::nullopt;
        }

        if (!it->value().is_array()) {
            if constexpr (ThrowOnConvertError) {
                throw JSONException("Invalid array type for parameter: " + key);
            }
            return std::nullopt;
        }

        std::vector<T> result;
        const json::array& arr = obj.at(key).as_array();

        for (const auto& elem : arr) {
            auto converted = try_convert<T, ThrowOnConvertError>(elem);
            if (converted) {
                result.push_back(*converted);
            }
        }

        return result;
    }

    /**
     * @brief Extract a value from a JSON object with validation.
     *
     * @tparam T The type of the value to extract.
     * @param obj The JSON object.
     * @param key The key of the value to extract.
     * @param validator The validation function for the value.
     * @param error_msg The error message to throw if validation fails.
     * @return The extracted value if it exists and passes validation, std::nullopt
     * otherwise.
     * @throws JSONException if the value exists but fails validation or is of the wrong
     * type.
     */
    template <typename T>
    static std::optional<T> extract_validated(const json::object& obj,
                                              const std::string& key,
                                              std::function<bool(const T&)> validator,
                                              const std::string& error_msg) {
        std::optional<T> value = extract_value<T, true>(obj, key);
        if (value && !validator(*value)) {
            throw JSONException(error_msg);
        }
        return value;
    }

    /**
     * @brief Extract an array of values from a JSON object with validation.
     *
     * @tparam T The type of the array elements.
     * @param obj The JSON object.
     * @param key The key of the array.
     * @param validator The validation function for the array.
     * @param error_msg The error message to throw if validation fails.
     * @return The array of values if it exists and passes validation, std::nullopt
     * otherwise.
     * @throws JSONException if the array exists but contains invalid values or is of the
     * wrong type.
     */
    template <typename T>
    static std::optional<std::vector<T>>
    extract_validated_array(const json::object& obj, const std::string& key,
                            std::function<bool(const std::vector<T>&)> validator,
                            const std::string& error_msg) {
        auto arr = extract_array<T, true>(obj, key);

        if (arr && !validator(*arr)) {
            throw JSONException(error_msg);
        }

        return arr;
    }

    /**
     * @brief Check if a JSON object has a specific key.
     *
     * @param obj The JSON object.
     * @param key The key to check for.
     * @return True if the key exists, false otherwise.
     */
    static bool has_key(const json::object& obj, const std::string& key) {
        return obj.contains(key);
    }

private:
    /**
     * @brief Try to convert a JSON value to a specific type.
     *
     * @tparam T The target type.
     * @tparam ThrowOnConvertError Whether to throw an exception on conversion error.
     * @param val The JSON value to convert.
     * @return The converted value if possible, std::nullopt otherwise.
     * @throws JSONException if the conversion fails - only if ThrowOnConvertError is
     * true.
     */
    template <typename T, bool ThrowOnConvertError = false>
    static std::optional<T> try_convert(const json::value& val) {
        if (is_type<T>(val)) {
            return json::value_to<T>(val);
        }

        if constexpr (ThrowOnConvertError) {
            throw JSONException("Invalid type for parameter");
        }

        return std::nullopt;
    }

    /**
     * @brief Check if a JSON value is of a specific type.
     *
     * @tparam T The type to check for.
     * @param val The JSON value to check.
     * @return True if the value is of the specified type, false otherwise.
     */
    template <typename T>
    static bool is_type(const json::value& val);
};

// Specializations for is_type
template <>
inline bool JsonHelper::is_type<std::string>(const json::value& val) {
    return val.is_string();
}

template <>
inline bool JsonHelper::is_type<int>(const json::value& val) {
    return val.is_int64();
}

template <>
inline bool JsonHelper::is_type<int64_t>(const json::value& val) {
    return val.is_int64();
}

template <>
inline bool JsonHelper::is_type<std::size_t>(const json::value& val) {
    return val.is_int64() && val.as_int64() >= 0;
}

template <>
inline bool JsonHelper::is_type<uint16_t>(const json::value& val) {
    return val.is_int64() && val.as_int64() >= 0 &&
           val.as_int64() <= std::numeric_limits<uint16_t>::max();
}

template <>
inline bool JsonHelper::is_type<std::uint32_t>(const json::value& val) {
    return val.is_int64() && val.as_int64() >= 0 &&
           static_cast<std::uint64_t>(val.as_int64()) <=
               std::numeric_limits<std::uint32_t>::max();
}

template <>
inline bool JsonHelper::is_type<bool>(const json::value& val) {
    return val.is_bool();
}

template <>
inline bool JsonHelper::is_type<double>(const json::value& val) {
    return val.is_double();
}
