#pragma once

#include "CapacityLimitedReservoir.hpp"
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>

// Constructor
template <typename Key, typename Value>
CapacityLimitedReservoir<Key, Value>::CapacityLimitedReservoir(std::size_t capacity)
    : m_capacity(capacity), m_rng(std::random_device{}()) {
    if (m_capacity == 0) {
        throw std::invalid_argument("Capacity must be greater than 0");
    }
}

// Add a value for a key
template <typename Key, typename Value>
void CapacityLimitedReservoir<Key, Value>::add(const Key &key, const Value &value) {
    auto &[values, total_seen] = m_data[key];

    if (total_seen == 0) {
        values.reserve(m_capacity);
    }
    total_seen++;

    if (values.size() < m_capacity) {
        values.push_back(value);
        return;
    }

    std::uniform_int_distribution<std::size_t> dist(0, total_seen - 1);
    std::size_t idx = dist(m_rng);
    if (idx < m_capacity) {
        values[idx] = value;
    }
}

// Get values associated with a key
template <typename Key, typename Value>
const std::vector<Value> &
CapacityLimitedReservoir<Key, Value>::get(const Key &key) const {
    auto it = m_data.find(key);
    if (it == m_data.end())
        throw std::out_of_range("Key not found");
    return it->second.first;
}

// Check if a key exists
template <typename Key, typename Value>
bool CapacityLimitedReservoir<Key, Value>::contains(const Key &key) const {
    return m_data.contains(key);
}

// Get the size for a specific key
template <typename Key, typename Value>
std::size_t CapacityLimitedReservoir<Key, Value>::get_size(const Key &key) const {
    auto it = m_data.find(key);
    return it != m_data.end() ? it->second.first.size() : 0;
}

// Get all keys
template <typename Key, typename Value>
std::vector<Key> CapacityLimitedReservoir<Key, Value>::get_keys() const {
    std::vector<Key> keys;
    keys.reserve(m_data.size());
    for (const auto &[key, _] : m_data) {
        keys.push_back(key);
    }
    return keys;
}

// Get the total number of keys
template <typename Key, typename Value>
std::size_t CapacityLimitedReservoir<Key, Value>::get_key_count() const {
    return m_data.size();
}

// Get the total number of values
template <typename Key, typename Value>
std::size_t CapacityLimitedReservoir<Key, Value>::get_total_size() const {
    std::size_t total = 0;
    for (const auto &[_, data] : m_data) {
        total += data.first.size();
    }
    return total;
}

// Get the capacity of the reservoir
template <typename Key, typename Value>
std::size_t CapacityLimitedReservoir<Key, Value>::get_capacity() const {
    return m_capacity;
}

// Set the seed for the random number generator
template <typename Key, typename Value>
void CapacityLimitedReservoir<Key, Value>::set_seed(unsigned int seed) {
    m_rng.seed(seed);
}

// Begin iterator
template <typename Key, typename Value>
auto CapacityLimitedReservoir<Key, Value>::begin() const ->
    typename std::unordered_map<Key, ReservoirData>::const_iterator {
    return m_data.cbegin();
}

// End iterator
template <typename Key, typename Value>
auto CapacityLimitedReservoir<Key, Value>::end() const ->
    typename std::unordered_map<Key, ReservoirData>::const_iterator {
    return m_data.cend();
}