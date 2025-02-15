#pragma once
#include "IRandomForestClassifier.hpp"
#include "mlpack/core/cv/metrics/average_strategy.hpp"
#include "statistics/metrics.hpp"
#include <mlpack/core.hpp>
#include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

/**
 * @brief A Random Forest classifier implementation using mlpack.
 *
 * @tparam Features The type of the features.
 * @tparam Labels The type of the labels.
 */
template <typename Features, typename Labels,
          mlpack::AverageStrategy AvgerageStrategy = mlpack::AverageStrategy::Binary>
class RandomForestClassifier
    : public IRandomForestClassifier<Features, Labels, statistics::Metrics> {
public:
    /**
     * @brief Construct a new Random Forest Classifier object.
     *
     * @param num_classes The number of classes.
     * @param num_trees The number of trees in the forest.
     */
    RandomForestClassifier(std::size_t num_classes = 2, std::size_t num_trees = 10);

    /**
     * @brief Train the classifier with the given features and labels.
     *
     * @param features The features.
     * @param labels The labels.
     */
    void train(const Features &features, const Labels &labels) override;

    /**
     * @brief Predict the labels for the given features.
     *
     * @param features The features.
     * @return The predicted labels.
     */
    Labels predict(const Features &features) const override;

    /**
     * @brief Evaluate the classifier using the given features and labels.
     *
     * @param features The features.
     * @param labels The labels.
     * @return The metrics of the classifier.
     */
    statistics::Metrics evaluate(const Features &features, const Labels &labels) override;

    /**
     * @brief Save the classifier to a file.
     *
     * @param path The file path to save the classifier.
     */
    void save(const std::string &path) const override;

    /**
     * @brief Load the classifier from a file.
     *
     * @param path The file path to load the classifier from.
     */
    void load(const std::string &path) override;

private:
    mlpack::RandomForest<> m_rf;         // The Random Forest model
    mlpack::data::MinMaxScaler m_scaler; // The MinMaxScaler for scaling the features
    std::size_t m_num_classes;           // The number of classes
    std::size_t m_num_trees;             // The number of trees in the forest
};

#include "RandomForestClassifier.tpp"