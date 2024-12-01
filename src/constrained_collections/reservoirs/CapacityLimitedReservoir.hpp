#pragma once
#include "ICapacityLimitedReservoir.hpp"
#include <unordered_map>
#include <utility>
#include <vector>
#include <random>

/**
 * @brief A capacity-limited reservoir implementation with standard STL iterators.
 *
 * This class manages a reservoir for storing a fixed number of values per key. 
 * If the number of values exceeds the specified capacity for a key, new values 
 * probabilistically replace existing ones, maintaining a uniform sampling.
 *
 * @tparam Key   The type used as the key in the reservoir.
 * @tparam Value The type of the values associated with each key.
 */
template <typename Key, typename Value>
class CapacityLimitedReservoir : public ICapacityLimitedReservoir<Key, Value> {
    
    /**
     * @brief The internal data structure for storing reservoir values.
     * 
     * The first element of the pair is a vector of values, and the second element
     * is the total number of values seen for the key - used for probabilistic
     * replacement.
     */
    using ReservoirData = std::pair<std::vector<Value>, std::size_t>;

public:
    /**
     * @brief Constructs a reservoir with a fixed capacity for each key.
     *
     * @param capacity The maximum number of values allowed per key.
     */
    explicit CapacityLimitedReservoir(std::size_t capacity);

    /**
     * @brief Adds a value associated with a key to the reservoir.
     *
     * If the number of values for the given key exceeds the capacity, the new value
     * may probabilistically replace an existing value.
     *
     * @param key   The key to associate the value with.
     * @param value The value to add to the reservoir.
     */
    void add(const Key& key, const Value& value) override;

    /**
     * @brief Retrieves all values associated with a specific key.
     *
     * If the key does not exist, exception is thrown.
     *
     * @param key The key to retrieve values for.
     * @return A vector of values associated with the key.
     */
    const std::vector<Value>& get(const Key& key) const override;

    /**
     * @brief Retrieves all keys currently in the reservoir.
     *
     * @return A vector of all keys in the reservoir.
     */
    std::vector<Key> get_keys() const override;

    /**
     * @brief Checks if a key exists in the reservoir.
     *
     * @param key The key to check.
     * @return True if the key exists, false otherwise.
     */
    bool contains(const Key& key) const override;

    /**
     * @brief Gets the number of values currently stored for a specific key.
     *
     * @param key The key to count values for.
     * @return The number of values stored for the key.
     */
    std::size_t size(const Key& key) const override;

    /**
     * @brief Sets the seed for the random number generator.
     *
     * @param seed The seed value to set for the random number generator.
     */
    void set_seed(unsigned int seed) override;

    /**
     * @brief Returns an iterator to the beginning of the reservoir.
     *
     * @return An iterator to the beginning of the reservoir's data.
     */
    auto begin() const -> 
        typename std::unordered_map<Key, ReservoirData>::const_iterator override;

    /**
     * @brief Returns an iterator to the end of the reservoir.
     *
     * @return An iterator to the end of the reservoir's data.
     */
    auto end() const ->
        typename std::unordered_map<Key, ReservoirData>::const_iterator override;

private:
    /**
     * @brief The internal data structure for storing reservoir values.
     */
    std::unordered_map<Key, ReservoirData> m_data;

    /**
     * @brief Maximum capacity of values for each key.
     */
    std::size_t m_capacity;

    /**
     * @brief Random number generator for probabilistic replacement.
     */
    std::mt19937 m_rng;
};

#include "CapacityLimitedReservoir.tpp"
