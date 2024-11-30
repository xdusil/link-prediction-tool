#pragma once

#include "IEvictingCounter.hpp"
#include <unordered_map>

/**
 * @brief An evicting counter that tracks items and their counts with a size
 * constraint.
 *
 * The counter will contain at most `limit` items. When the limit is reached,
 * the counter will decrement counter for each item and remove items with zero
 * counts. - This results in a counter that tracks the items with the highest
 * frequency.
 *
 * @tparam Key The type of the items to count.
 */
template <typename Key> class EvictingCounter : public IEvictingCounter<Key> {
public:
    /**
    * @brief Constructor for EvictingCounter.
    *
    * @param limit The maximum number of items to track.
    */
    explicit EvictingCounter(std::size_t limit);

    /**
    * @brief Add an item or increment its count. If the size limit is reached,
    * decrements and evicts items as needed.
    *
    * @param key The item to add or increment.
    */
    void add_or_decrement(const Key &key) override;

    /**
    * @brief Check if the counter contains a specific key.
    *
    * @param key The item to check.
    * @return True if the item exists, false otherwise.
    */
    bool contains(const Key &key) const override;

    /**
    * @brief Retrieve all items and their counts.
    *
    * @return An unordered map of items and their counts.
    */
    const std::unordered_map<Key, int> &get_items() const override;

    /**
    * @brief Retrieve all items and their counts, and clear the counter.
    *
    * @return An unordered map of items and their counts.
    */
    std::unordered_map<Key, int> get_items_and_clear() override;

    /**
    * @brief Clear the counter.
    */
    void clear() override;

private:
    /**
    * @brief Decrements all counts and removes items with zero counts.
    */
    void decrement_and_remove();

    std::size_t m_limit;                    // The maximum size of the counter
    std::unordered_map<Key, int> m_counter; // Tracks items and their counts
};

#include "EvictingCounter.tpp"