#pragma once

/**
 * @brief Interface for a data loader.
 *
 * A data loader is responsible for loading data in batches and managing
 * epoch boundaries. Supports iteration through multiple batches per epoch.
 *
 * Usage pattern:
 * @code
 * data_loader.reset();  // Start new epoch
 * while (data_loader.has_next()) {
 *     auto batch = data_loader.next_batch();
 *     // Train on batch
 * }
 * @endcode
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

    /**
     * @brief Check if there are more batches available in current epoch.
     *
     * @return true if next_batch() can be called, false if epoch is complete.
     */
    virtual bool has_next() const = 0;

    /**
     * @brief Reset the data loader to start a new epoch.
     *
     * Regenerates random walks and prepares data for the next epoch.
     */
    virtual void reset() = 0;
};