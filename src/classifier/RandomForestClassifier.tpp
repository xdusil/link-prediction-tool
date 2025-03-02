#pragma once

#include "../statistics/metrics.hpp"
#include "RandomForestClassifier.hpp"
#include "exceptions/exceptions.hpp"
#include "mlpack/core/cv/metrics/accuracy.hpp"
#include "mlpack/core/cv/metrics/average_strategy.hpp"
#include "mlpack/core/cv/metrics/metrics.hpp"
#include "mlpack/methods/random_forest/random_forest.hpp"
#include "utils/utils.hpp"
#include <cstddef>
#include <ostream>
#include <tuple>

template <typename Features, typename Labels, mlpack::AverageStrategy AverageStrategy>
RandomForestClassifier<Features, Labels, AverageStrategy>::RandomForestClassifier(
    std::size_t num_classes /*= 2*/, std::size_t num_trees /*= 10*/,
    std::size_t min_leaf_size /*= 1*/, double min_gain_split /*= 0.0*/,
    std::size_t max_depth /*= 0*/)
    : m_num_classes(num_classes), m_num_trees(num_trees), m_min_leaf_size(min_leaf_size),
      m_min_gain_split(min_gain_split), m_max_depth(max_depth) {
    if (AverageStrategy == mlpack::AverageStrategy::Binary && num_classes != 2) {
        throw RandomForestException(
            "Binary average strategy can only be used with 2 classes.");
    }

    if (num_trees < 1) {
        throw RandomForestException("Number of trees must be at least 1.");
    }
}

template <typename Features, typename Labels, mlpack::AverageStrategy AverageStrategy>
RandomForestClassifier<Features, Labels, AverageStrategy>::RandomForestClassifier(
    std::size_t num_classes, const RandomForestParams &params)
    : RandomForestClassifier(num_classes, params.num_trees, params.min_leaf_size,
                             params.min_gain_split, params.max_depth) {}

template <typename Features, typename Labels, mlpack::AverageStrategy AverageStrategy>
RandomForestClassifier<Features, Labels, AverageStrategy>::RandomForestClassifier(
    mlpack::RandomForest<> &&rf, std::size_t num_classes,
    const RandomForestParams &params)
    : RandomForestClassifier(num_classes, params), m_rf(std::move(rf)) {}

template <typename Features, typename Labels, mlpack::AverageStrategy AverageStrategy>
void RandomForestClassifier<Features, Labels, AverageStrategy>::train(
    const Features &features, const Labels &labels) {

    std::cout << "Training Random Forest with " << m_num_classes << " classes."
              << std::endl;
    std::cout << "Features: " << features.n_rows << " x " << features.n_cols << std::endl;

    // Train the Random Forest
    m_rf.Train(features, labels, m_num_classes, m_num_trees, m_min_leaf_size,
               m_min_gain_split, m_max_depth);
}

template <typename Features, typename Labels, mlpack::AverageStrategy AverageStrategy>
Labels RandomForestClassifier<Features, Labels, AverageStrategy>::predict(
    const Features &features) const {
    // Predict using the Random Forest
    Labels predictions;
    m_rf.Classify(features, predictions);

    return predictions;
}

template <typename Features, typename Labels, mlpack::AverageStrategy AverageStrategy>
statistics::Metrics RandomForestClassifier<Features, Labels, AverageStrategy>::evaluate(
    const Features &features, const Labels &labels) {

    // Predict using the Random Forest
    Labels predictions;
    m_rf.Classify(features, predictions);

    statistics::Metrics metrics;
    metrics.accuracy = mlpack::Accuracy::Evaluate(m_rf, features, labels);
    metrics.precision =
        mlpack::Precision<AverageStrategy>::Evaluate(m_rf, features, labels);
    metrics.recall = mlpack::Recall<AverageStrategy>::Evaluate(m_rf, features, labels);
    metrics.f1_score = mlpack::F1<AverageStrategy>::Evaluate(m_rf, features, labels);
    return metrics;
}

template <typename Features, typename Labels, mlpack::AverageStrategy AverageStrategy>
void RandomForestClassifier<Features, Labels, AverageStrategy>::save(
    const std::string &path) const {
    bool res = mlpack::data::Save(path, "rf_model", m_rf);
    if (!res) {
        throw RandomForestException("Failed to save the Random Forest model to " + path);
    }
}

template <typename Features, typename Labels, mlpack::AverageStrategy AverageStrategy>
void RandomForestClassifier<Features, Labels, AverageStrategy>::load(
    const std::string &path) {
    bool res = mlpack::data::Load(path, "rf_model", m_rf);
    if (!res) {
        throw RandomForestException("Failed to load the Random Forest model from " +
                                    path);
    }
}

template <typename Features, typename Labels, mlpack::AverageStrategy AverageStrategy>
template <typename Metric>
std::tuple<RandomForestParams, double>
RandomForestClassifier<Features, Labels, AverageStrategy>::grid_search(
    const Features &features, const Labels &labels, const std::size_t num_classes,
    const std::vector<std::size_t> &num_trees,
    const std::vector<std::size_t> &min_leaf_size,
    const std::vector<double> &min_gain_split, const std::vector<std::size_t> &max_depth,
    const double validation_size /*= 0.3*/) {
    std::cout << "Performing grid search for Random Forest hyperparameters." << std::endl;
    RandomForestParams best_params;

    if (num_classes < 2) {
        throw RandomForestException("Number of classes must be at least 2.");
    }

    if (num_trees.empty() || min_leaf_size.empty() || min_gain_split.empty() ||
        max_depth.empty()) {
        throw RandomForestException(
            "Grid search requires non-empty hyperparameter lists.");
    }

    arma::mat double_features;
    if constexpr (std::is_same_v<Features, arma::fmat>) {
        std::cout
            << "Converting features from float to double precision for grid search..."
            << std::endl;
        double_features = arma::conv_to<arma::mat>::from(features);
    }

    const auto &features_to_use = std::is_same_v<Features, arma::fmat>
                                      ? static_cast<const arma::mat &>(double_features)
                                      : features;

    mlpack::HyperParameterTuner<mlpack::RandomForest<>, Metric, mlpack::SimpleCV,
                                ens::GridSearch>
        tuner(features_to_use, labels, static_cast<size_t>(num_classes));

    std::tie(best_params.num_trees, best_params.min_leaf_size, best_params.min_gain_split,
             best_params.max_depth) =
        tuner.Optimize(num_trees, min_leaf_size, min_gain_split, max_depth);

    const double best_result = tuner.BestObjective();

    std::cout << "Best hyperparameters:\n";
    std::cout << "  numTrees:     " << best_params.num_trees << "\n";
    std::cout << "  minLeafSize:  " << best_params.min_leaf_size << "\n";
    std::cout << "  maxDepth:     " << best_params.max_depth << "\n";
    std::cout << "  minGainSplit: " << best_params.min_gain_split << "\n";
    std::cout << "  Best objective: " << best_result << "\n";

    return std::make_tuple(best_params, best_result);
}