#pragma once

#include "EvictingCounter.hpp"

// Constructor
template <typename Key>
EvictingCounter<Key>::EvictingCounter(std::size_t limit)
    : m_limit(limit) {}

// Add an item or increment its count
template <typename Key>
void EvictingCounter<Key>::add_or_decrement(const Key& key) {
    if (auto it = m_counter.find(key); it != m_counter.end()) {
        it->second++;
    } else if (m_counter.size() < m_limit) {
        m_counter[key] = 1;
    } else {
        decrement_and_remove();
    }
}

// Decrement and remove items with zero counts
template <typename Key>
void EvictingCounter<Key>::decrement_and_remove() {
    for (auto it = m_counter.begin(); it != m_counter.end(); ) {
        auto& [key, value] = *it;
        value--;
        if (value == 0) {
            it = m_counter.erase(it);
        } else {
            ++it;
        }
    }
}

// Check if the counter contains a specific key
template <typename Key>
bool EvictingCounter<Key>::contains(const Key& key) const {
    return m_counter.find(key) != m_counter.end();
}

// Retrieve all items and their counts
template <typename Key>
const std::unordered_map<Key, int>& EvictingCounter<Key>::get_items() const {
    return m_counter;
}

// Retrieve all items and their counts, and clear the counter
template <typename Key>
std::unordered_map<Key, int> EvictingCounter<Key>::get_items_and_clear() {
    std::unordered_map<Key, int> items = std::move(m_counter);
    clear();
    return items;
}

// Get the limit of the counter
template <typename Key>
std::size_t EvictingCounter<Key>::get_limit() const {
    return m_limit;
}

// Get the size of the counter
template <typename Key>
std::size_t EvictingCounter<Key>::get_size() const {
    return m_counter.size();
}

// Clear the counter
template <typename Key>
void EvictingCounter<Key>::clear() {
    m_counter.clear();
}