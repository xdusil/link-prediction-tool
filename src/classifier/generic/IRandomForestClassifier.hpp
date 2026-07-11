#pragma once
#include <string>
#include <vector>

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
 * @brief Interface for a Random Forest classifier.
 *
 * @tparam Features The type of the features.
 * @tparam Labels The type of the labels.
 * @tparam Metrics The type of evaluation metrics.
 * @tparam ProbMatrix The type of probability matrix returned by predict_proba.
 */
template <typename Features, typename Labels, typename Metrics, typename ProbMatrix>
class IRandomForestClassifier {
public:
    virtual ~IRandomForestClassifier() = default;

    /**
     * @brief Train the classifier with the given features and labels.
     *
     * @param features The features.
     * @param labels The labels.
     * @param use_weights Whether to use weights for the training - according to the
     *                    number of occurrences of each class.
     */
    virtual void train(const Features &features, const Labels &labels,
                       bool use_weights) = 0;

    /**
     * @brief Predict the labels for the given features.
     *
     * @param features The features.
     * @return The predicted labels.
     */
    virtual Labels predict(const Features &features) const = 0;

    /**
     * @brief Predict probabilities for each class for the given features.
     *
     * @param features The features.
     * @return A tuple containing the predicted labels and a matrix of probabilities.
     */
    virtual std::tuple<Labels, ProbMatrix>
    predict_proba(const Features &features) const = 0;

    /**
     * @brief Evaluate the classifier using the given features and labels.
     *
     * @param features The features.
     * @param labels The labels.
     * @return The metrics of the classifier.
     */
    virtual Metrics evaluate(const Features &features, const Labels &labels) const = 0;

    /**
     * @brief Save the classifier to a file.
     *
     * @param path The file path to save the classifier.
     */
    virtual void save(const std::string &path) const = 0;

    /**
     * @brief Load the classifier from a file.
     *
     * @param path The file path to load the classifier from.
     */
    virtual void load(const std::string &path) = 0;

    /**
     * @brief Return the parameters of the current trained or loaded model.
     */
    virtual RandomForestParams get_params() const = 0;

    /**
     * @brief Calculate permutation-based feature importance
     * 
     * @param features Validation/test features
     * @param labels Ground truth labels
     * @param feature_names Names of features
     * @param metric Metric to measure ("f1", "accuracy", etc.)
     * @param n_repeats Number of shuffling repeats
     * @return Vector of (feature_name, importance_score) pairs
     */
    virtual std::vector<std::pair<std::string, double>> calculate_feature_importance(
        const Features &features, const Labels &labels,
        const std::vector<std::string> &feature_names,
        const std::string &metric = "f1",
        size_t n_repeats = 5) const = 0;
};