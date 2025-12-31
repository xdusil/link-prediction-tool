#pragma once

#include <functional>
#include <optional>
#include <unordered_map>
#include <utility>
#include <cstddef>


/**
 * @brief Generic cache for feature computations with per-pair or per-vertex storage.
 *
 * Provides efficient memoization for expensive computations that are called
 * repeatedly with the same arguments. Supports both single-key and pair-key lookups.
 *
 * @tparam Key The type of the key (e.g., Vertex)
 * @tparam Value The type of the cached value (e.g., std::vector<Vertex>, double,
 * std::size_t)
 * @tparam Hash The hash function for the key type
 */
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class FeatureCache {
public:
    /**
     * @brief Clear all cached data.
     */
    void clear() {
        m_single_key_cache.clear();
        m_pair_key_cache.clear();
    }

    /**
     * @brief Get the number of cached entries for single-key lookups.
     */
    std::size_t single_key_size() const { return m_single_key_cache.size(); }

    /**
     * @brief Get the total number of cached entries for pair-key lookups.
     */
    std::size_t pair_key_size() const {
        std::size_t total = 0;
        for (const auto& [key, inner_map] : m_pair_key_cache) {
            total += inner_map.size();
        }
        return total;
    }

    /**
     * @brief Check if a value exists in single-key cache.
     *
     * @param key The key to lookup
     * @return True if the key exists, false otherwise
     */
    bool contains(const Key& key) const {
        return m_single_key_cache.find(key) != m_single_key_cache.end();
    }

    /**
     * @brief Get a value from single-key cache if it exists.
     *
     * @param key The key to lookup
     * @return Optional containing the value if found, empty otherwise
     */
    std::optional<Value> get(const Key& key) const {
        auto it = m_single_key_cache.find(key);
        if (it != m_single_key_cache.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    /**
     * @brief Get or compute a value for single-key cache.
     *
     * If the value exists in cache, return it. Otherwise, compute it using
     * the provided function, store it in cache, and return it.
     *
     * @tparam ComputeFn Function type: Value(const Key&)
     * @param key The key to lookup or compute for
     * @param compute_fn Function to compute the value if not cached
     * @return The cached or newly computed value
     */
    template <typename ComputeFn>
    Value get_or_compute(const Key& key, ComputeFn&& compute_fn) {
        auto it = m_single_key_cache.find(key);
        if (it != m_single_key_cache.end()) {
            return it->second;
        }

        Value result = std::forward<ComputeFn>(compute_fn)(key);
        m_single_key_cache[key] = result;
        return result;
    }

    /**
     * @brief Store a value in single-key cache.
     *
     * @param key The key to store
     * @param value The value to store
     */
    void put(const Key& key, const Value& value) { m_single_key_cache[key] = value; }

    /**
     * @brief Store a value in single-key cache (move version).
     *
     * @param key The key to store
     * @param value The value to store
     */
    void put(const Key& key, Value&& value) {
        m_single_key_cache[key] = std::move(value);
    }

    // ========================================================================
    // Pair-Key Cache Operations
    // ========================================================================

    /**
     * @brief Check if a value exists in pair-key cache.
     *
     * @param key1 First key
     * @param key2 Second key
     * @return True if the key pair exists, false otherwise
     */
    bool contains(const Key& key1, const Key& key2) const {
        auto it = m_pair_key_cache.find(key1);
        if (it == m_pair_key_cache.end()) {
            return false;
        }
        return it->second.find(key2) != it->second.end();
    }

    /**
     * @brief Get a value from pair-key cache if it exists.
     *
     * @param key1 First key
     * @param key2 Second key
     * @return Optional containing the value if found, empty otherwise
     */
    std::optional<Value> get(const Key& key1, const Key& key2) const {
        auto it = m_pair_key_cache.find(key1);
        if (it == m_pair_key_cache.end()) {
            return std::nullopt;
        }

        auto it2 = it->second.find(key2);
        if (it2 != it->second.end()) {
            return it2->second;
        }
        return std::nullopt;
    }

    /**
     * @brief Get or compute a value for pair-key cache.
     *
     * If the value exists in cache, return it. Otherwise, compute it using
     * the provided function, store it in cache, and return it.
     *
     * @tparam ComputeFn Function type: Value(const Key&, const Key&)
     * @param key1 First key
     * @param key2 Second key
     * @param compute_fn Function to compute the value if not cached
     * @return The cached or newly computed value
     */
    template <typename ComputeFn>
    Value get_or_compute(const Key& key1, const Key& key2, ComputeFn&& compute_fn) {
        auto it = m_pair_key_cache.find(key1);
        if (it != m_pair_key_cache.end()) {
            auto it2 = it->second.find(key2);
            if (it2 != it->second.end()) {
                return it2->second;
            }
        }

        Value result = std::forward<ComputeFn>(compute_fn)(key1, key2);
        m_pair_key_cache[key1][key2] = result;
        return result;
    }

    /**
     * @brief Store a value in pair-key cache.
     *
     * @param key1 First key
     * @param key2 Second key
     * @param value The value to store
     */
    void put(const Key& key1, const Key& key2, const Value& value) {
        m_pair_key_cache[key1][key2] = value;
    }

    /**
     * @brief Store a value in pair-key cache (move version).
     *
     * @param key1 First key
     * @param key2 Second key
     * @param value The value to store
     */
    void put(const Key& key1, const Key& key2, Value&& value) {
        m_pair_key_cache[key1][key2] = std::move(value);
    }

private:
    // Single-key cache: Key -> Value
    std::unordered_map<Key, Value, Hash> m_single_key_cache;

    // Pair-key cache: Key1 -> (Key2 -> Value)
    std::unordered_map<Key, std::unordered_map<Key, Value, Hash>, Hash> m_pair_key_cache;
};
