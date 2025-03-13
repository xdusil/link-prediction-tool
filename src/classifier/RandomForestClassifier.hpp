#pragma once
#include "IRandomForestClassifier.hpp"
#include "mlpack/core/cv/metrics/average_strategy.hpp"
#include "statistics/metrics.hpp"
#include <mlpack/core.hpp>
#include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

/**
 * @brief The parameters for the Random Forest classifier.
 */
struct RandomForestParams {
    std::size_t num_trees;
    std::size_t min_leaf_size;
    double min_gain_split;
    std::size_t max_depth;
};

/**
 * @brief The parameters for the Random Forest grid search.
 */
struct GridSearchParams {
    std::vector<std::size_t> num_trees;
    std::vector<std::size_t> min_leaf_size;
    std::vector<double> min_gain_split;
    std::vector<std::size_t> max_depth;
    double validation_size;
};

/**
 * @brief A Random Forest classifier implementation using mlpack.
 *
 * @tparam Features The type of the features.
 * @tparam Labels The type of the labels.
 * @tparam AvgerageStrategy The average strategy for the metrics.
 * @tparam Scaler The type of the scaler.
 */
template <typename Features, typename Labels,
          mlpack::AverageStrategy AvgerageStrategy = mlpack::AverageStrategy::Binary,
          typename Scaler = mlpack::data::MinMaxScaler>
class RandomForestClassifier
    : public IRandomForestClassifier<Features, Labels, statistics::Metrics, arma::mat> {
public:
    /**
     * @brief Construct a new Random Forest Classifier object.
     *
     * @param num_classes The number of classes.
     * @param num_trees The number of trees in the forest.
     * @param min_leaf_size The minimum number of points in each tree's leaf nodes.
     * @param min_gain_split The minimum gain for splitting a decision tree node.
     * @param max_depth The maximum depth for the tree.
     * @param use_scaling Whether to use scaling for the features.
     */
    RandomForestClassifier(std::size_t num_classes = 2, std::size_t num_trees = 10,
                           std::size_t min_leaf_size = 1, double min_gain_split = 0.0,
                           std::size_t max_depth = 0, bool use_scaling = true);

    /**
     * @brief Construct a new Random Forest Classifier object.
     *
     * @param num_classes The number of classes.
     * @param params The Random Forest parameters.
     * @param use_scaling Whether to use scaling for the features.
     */
    RandomForestClassifier(std::size_t num_classes, const RandomForestParams &params,
                           bool use_scaling = true);

    /**
     * @brief Construct a new Random Forest Classifier object.
     *
     * @param rf The Random Forest model.
     * @param num_classes The number of classes.
     * @param params The Random Forest parameters that were used to train the model.
     * @param use_scaling Whether to use scaling for the features.
     */
    RandomForestClassifier(mlpack::RandomForest<> &&rf, std::size_t num_classes,
                           const RandomForestParams &params, bool use_scaling = true);

    /**
     * @brief Train the classifier with the given features and labels.
     *
     * @param features The features.
     * @param labels The labels.
     * @param use_weights Whether to use weights for the training - according to the
     *                    number of occurrences of each class.
     */
    void train(const Features &features, const Labels &labels,
               bool use_weights = false) override;

    /**
     * @brief Predict the labels for the given features.
     *
     * @param features The features.
     * @return The predicted labels.
     */
    Labels predict(const Features &features) const override;

    /**
     * @brief Predict probabilities for each class for the given features.
     *
     * @param features The features.
     * @return A tuple containing the predicted labels and a matrix of probabilities.
     */
    std::tuple<Labels, arma::mat> predict_proba(const Features &features) const override;

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

    /**
     * @brief Perform a grid search to find the best Random Forest parameters.
     *
     * @param features The features.
     * @param labels The labels.
     * @param num_classes The number of classes.
     * @param num_trees The vector of the number of trees to search.
     * @param min_leaf_size The vector of the minimum leaf size to search.
     * @param min_gain_split The vector of the minimum gain split to search.
     * @param max_depth The vector of the maximum depth to search.
     * @param validation_size The size of the validation set.
     * @param use_scaling Whether to use scaling for the features.
     * @param use_weights Whether to use weights for the training - according to the
     * @return The best Random Forest parameters and the best metric score.
     */
    template <typename Metric>
    static std::tuple<RandomForestParams, double>
    grid_search(const Features &features, const Labels &labels,
                const std::size_t num_classes, const std::vector<std::size_t> &num_trees,
                const std::vector<std::size_t> &min_leaf_size,
                const std::vector<double> &min_gain_split,
                const std::vector<std::size_t> &max_depth,
                const double validation_size = 0.3, bool use_scaling = true,
                bool use_weights = false);

    /**
     * @brief Perform a grid search to find the best Random Forest parameters.
     *
     * @param features The features.
     * @param labels The labels.
     * @param num_classes The number of classes.
     * @param params The grid search parameters.
     * @param use_scaling Whether to use scaling for the features.
     * @param use_weights Whether to use weights for the training - according to the
     *                    number of occurrences of each class.
     * @return The best Random Forest parameters and the best metric score.
     */
    template <typename Metric>
    static std::tuple<RandomForestParams, double>
    grid_search(const Features &features, const Labels &labels,
                const std::size_t num_classes, const GridSearchParams &params,
                bool use_scaling = true, bool use_weights = false);

private:
    mlpack::RandomForest<> m_rf; // The Random Forest model
    mutable Scaler m_scaler;     // Configurable scaler
    bool m_use_scaling;          // Whether to use scaling for the features
    std::size_t m_num_classes;   // The number of classes
    std::size_t m_num_trees;     // The number of trees in the forest
    std::size_t m_min_leaf_size; // The minimum number of points in each tree's leaf nodes
    double m_min_gain_split;     // The minimum gain for splitting a decision tree node
    std::size_t m_max_depth;     // The maximum depth for the tree

    /**
     * @brief Calculate the weights for the training data.
     *
     * @param labels The labels.
     * @param weights The weights.
     * @param num_classes The number of classes.
     */
    static void calculate_weights(const Labels &labels, arma::rowvec &weights, std::size_t num_classes);
};

#include "RandomForestClassifier.tpp"