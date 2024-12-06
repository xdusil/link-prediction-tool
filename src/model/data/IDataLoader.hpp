#pragma once

/**
 * @brief Interface for a data loader.
 *
 * A data loader is responsible for loading data in batches.
 *
 * @tparam T The type of data to load.
 */
template <typename T>
class IDataLoader {
public:
    virtual ~IDataLoader() = default;

    /**
     * @brief Get the next batch of data.
     *
     * @return The next batch of data.
     */
    virtual T next_batch() = 0;
};