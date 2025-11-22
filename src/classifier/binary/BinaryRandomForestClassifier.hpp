#pragma once

#include "../generic/RandomForestClassifier.hpp"
#include "IBinaryRandomForestClassifier.hpp"
#include "statistics/metrics.hpp"

/**
 * @brief Specialized Random Forest classifier optimized for binary classification.
 *
 * This class extends RandomForestClassifier with binary-specific functionality
 * such as threshold calibration and optimized binary evaluation.
 *
 * @tparam Features The type of the features.
 * @tparam Labels The type of the labels.
 * @tparam Scaler The type of the scaler.
 */
template <typename Features, typename Labels,
          typename Scaler = mlpack::data::MinMaxScaler>
class BinaryRandomForestClassifier
    : public RandomForestClassifier<Features, Labels, statistics::AverageType::BINARY,
                                    Scaler>,
      public IBinaryRandomForestClassifier<Features, Labels, statistics::Metrics,
                                           arma::mat> {
public:
    // Base class type alias for easier reference
    using Base =
        RandomForestClassifier<Features, Labels, statistics::AverageType::BINARY, Scaler>;

    /**
     * @brief Construct a new Binary Random Forest Classifier object.
     *
     * @param num_trees The number of trees in the forest.
     * @param min_leaf_size The minimum number of points in each tree's leaf nodes.
     * @param min_gain_split The minimum gain for splitting a decision tree node.
     * @param max_depth The maximum depth for the tree.
     * @param use_scaling Whether to use scaling for the features.
     */
    BinaryRandomForestClassifier(std::size_t num_trees = 10,
                                 std::size_t min_leaf_size = 1,
                                 double min_gain_split = 0.0, std::size_t max_depth = 0,
                                 bool use_scaling = true);

    /**
     * @brief Construct a new Binary Random Forest Classifier object.
     *
     * @param params The Random Forest parameters.
     * @param use_scaling Whether to use scaling for the features.
     */
    BinaryRandomForestClassifier(const RandomForestParams &params,
                                 bool use_scaling = true);

    /**
     * @brief Train with automatic threshold calibration
     *
     * This extends the base training with threshold calibration
     * while preserving the original train() behavior.
     *
     * @param features Training features
     * @param labels Training labels
     * @param use_weights Whether to use class weights
     * @param metric The metric to optimize ("f1", "precision", etc.)
     * @param validation_size The size of the validation set
     */
    void train_with_calibration(const Features &features, const Labels &labels,
                                bool use_weights = false,
                                const std::string &metric = "f1",
                                double validation_size = 0.25) override;

    /**
     * @brief Predict using the optimal decision threshold.
     *
     * Overrides base class prediction to use the calibrated threshold
     * for more accurate binary classification.
     *
     * @param features The features.
     * @return The predicted labels.
     */
    Labels predict(const Features &features) const override;

    /**
     * @brief Evaluate model using the calibrated threshold.
     *
     * @param features The features.
     * @param labels The labels.
     * @return Model performance metrics.
     */
    statistics::Metrics evaluate(const Features &features, const Labels &labels) const override;

    /**
     * @brief Get the current threshold.
     *
     * @return Current threshold value (0-1).
     */
    double get_threshold() const override;

    /**
     * @brief Set the threshold manually.
     *
     * @param threshold Value between 0 and 1.
     */
    void set_threshold(double threshold) override;

    /**
     * @brief Calibrate decision threshold to optimize a metric.
     *
     * @param val_features Validation features
     * @param val_labels Validation labels
     * @param metric Metric to optimize (f1, precision, recall, accuracy)
     */
    void calibrate_threshold(const Features &val_features, const Labels &val_labels,
                             const std::string &metric = "f1") override;

    /**
     * @brief Save the model with binary-specific parameters.
     *
     * @param path File path to save to.
     */
    void save(const std::string &path) const override;

    /**
     * @brief Load the model with binary-specific parameters.
     *
     * @param path File path to load from.
     */
    void load(const std::string &path) override;

private:
    double m_binary_threshold = 0.5; // Specialized threshold for binary classification

    /**
     * @brief Serialize the object for saving/loading.
     *
     * @tparam Archive Type of the cereal archive.
     * @param archive The cereal archive.
     */
    template <class Archive>
    void serialize(Archive &archive);

    /**
     * @brief Find the optimal threshold in a specified range.
     *
     * @param val_features Validation features
     * @param val_labels Validation labels
     * @param metric Metric to optimize
     * @param min_threshold Minimum threshold value to try
     * @param max_threshold Maximum threshold value to try
     * @param step Step size between thresholds
     * @return A tuple containing the optimal threshold and the corresponding metrics.
     */
    std::tuple<double, statistics::Metrics>
    find_optimal_threshold(const Features &val_features, const Labels &val_labels,
                           const std::string &metric = "f1", double min_threshold = 0.01,
                           double max_threshold = 0.99, double step = 0.01);

    // Allow cereal access to private members
    friend class cereal::access;
};

#include "BinaryRandomForestClassifier.tpp"