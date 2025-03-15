#pragma once

#include "../statistics/metrics.hpp"
#include "RandomForestClassifier.hpp"
#include "exceptions/exceptions.hpp"
#include "mlpack/core/cv/metrics/accuracy.hpp"
#include "mlpack/core/cv/metrics/average_strategy.hpp"
#include "mlpack/core/cv/metrics/f1.hpp"
#include "mlpack/core/cv/metrics/metrics.hpp"
#include "mlpack/methods/random_forest/random_forest.hpp"
#include "utils/utils.hpp"
#include <cstddef>
#include <exception>
#include <ostream>
#include <tuple>

template <typename Features, typename Labels, statistics::AverageType AverageType,
          typename Scaler>
RandomForestClassifier<Features, Labels, AverageType, Scaler>::RandomForestClassifier(
    std::size_t num_classes /*= 2*/, std::size_t num_trees /*= 10*/,
    std::size_t min_leaf_size /*= 1*/, double min_gain_split /*= 0.0*/,
    std::size_t max_depth /*= 0*/, bool use_scaling /*= true*/)
    : m_num_classes(num_classes), m_num_trees(num_trees), m_min_leaf_size(min_leaf_size),
      m_min_gain_split(min_gain_split), m_max_depth(max_depth),
      m_use_scaling(use_scaling) {
    if (AverageType == statistics::AverageType::BINARY && num_classes != 2) {
        throw RandomForestException(
            "Binary average strategy can only be used with 2 classes.");
    }

    if (num_trees < 1) {
        throw RandomForestException("Number of trees must be at least 1.");
    }
}

template <typename Features, typename Labels, statistics::AverageType AverageType,
          typename Scaler>
RandomForestClassifier<Features, Labels, AverageType, Scaler>::RandomForestClassifier(
    std::size_t num_classes, const RandomForestParams &params,
    bool use_scaling /*= true*/)
    : RandomForestClassifier(num_classes, params.num_trees, params.min_leaf_size,
                             params.min_gain_split, params.max_depth, use_scaling) {}

template <typename Features, typename Labels, statistics::AverageType AverageType,
          typename Scaler>
RandomForestClassifier<Features, Labels, AverageType, Scaler>::RandomForestClassifier(
    mlpack::RandomForest<> &&rf, std::size_t num_classes,
    const RandomForestParams &params, bool use_scaling /*= true*/)
    : RandomForestClassifier(num_classes, params, use_scaling) {
    m_rf = std::move(rf);
}

template <typename Features, typename Labels, statistics::AverageType AverageType,
          typename Scaler>
void RandomForestClassifier<Features, Labels, AverageType, Scaler>::calculate_weights(
    const Labels &labels, arma::rowvec &weights, std::size_t num_classes) {
    if (num_classes == 0) {
        throw RandomForestException("Number of classes must be greater than 0.");
    }
    // Count the occurrences of each class
    arma::Row<size_t> counts(num_classes, arma::fill::zeros);
    for (size_t i = 0; i < labels.n_elem; ++i) {
        counts[labels[i]]++;
    }

    // Calculate total number of samples
    const size_t total_samples = labels.n_elem;

    // Calculate balanced weights (inverse of frequency)
    // Higher weight for minority class, lower weight for majority class
    arma::Row<double> class_weights(num_classes);
    for (size_t i = 0; i < num_classes; ++i) {
        class_weights[i] = counts[i] > 0
                               ? static_cast<double>(total_samples) /
                                     (static_cast<double>(num_classes) * counts[i])
                               : 0.0;
    }

    std::cout << "Class distribution: ";
    for (size_t i = 0; i < num_classes; ++i) {
        std::cout << "Class " << i << ": " << counts[i] << " ("
                  << 100.0 * counts[i] / total_samples << "%), ";
    }
    std::cout << std::endl;

    std::cout << "Class weights: ";
    for (size_t i = 0; i < num_classes; ++i) {
        std::cout << "Class " << i << ": " << class_weights[i] << ", ";
    }
    std::cout << std::endl;

    // Apply class weights to each sample based on its class
    weights.set_size(total_samples);
    for (size_t i = 0; i < total_samples; ++i) {
        weights[i] = class_weights[labels[i]];
    }
}

template <typename Features, typename Labels, statistics::AverageType AverageType,
          typename Scaler>
void RandomForestClassifier<Features, Labels, AverageType, Scaler>::train(
    const Features &features, const Labels &labels, bool use_weights /*= false*/) {
    if (features.n_cols != labels.n_elem) {
        throw RandomForestException("Number of features and labels must be equal.");
    }

    std::cout << "Training Random Forest with " << m_num_classes << " classes.\n";
    std::cout << "Features: " << features.n_rows << " x " << features.n_cols << std::endl;

    arma::mat feats = arma::conv_to<arma::mat>::from(features);
    if (m_use_scaling) {
        m_scaler.Fit(feats);
        m_scaler.Transform(feats, feats);
    }

    if (use_weights) {
        arma::rowvec weights;
        calculate_weights(labels, weights, m_num_classes);
        m_rf.Train(feats, labels, m_num_classes, weights, m_num_trees, m_min_leaf_size,
                   m_min_gain_split, m_max_depth);
        return;
    }

    m_rf.Train(feats, labels, m_num_classes, m_num_trees, m_min_leaf_size,
               m_min_gain_split, m_max_depth);
}

template <typename Features, typename Labels, statistics::AverageType AverageType,
          typename Scaler>
Labels RandomForestClassifier<Features, Labels, AverageType, Scaler>::predict(
    const Features &features) const {
    auto feats = arma::conv_to<arma::mat>::from(features);
    if (m_use_scaling) {
        m_scaler.Transform(feats, feats);
    }

    // Predict using the Random Forest
    Labels predictions;
    m_rf.Classify(feats, predictions);

    return predictions;
}

template <typename Features, typename Labels, statistics::AverageType AverageType,
          typename Scaler>
std::tuple<Labels, arma::mat>
RandomForestClassifier<Features, Labels, AverageType, Scaler>::predict_proba(
    const Features &features) const {
    // Predict using the Random Forest
    Labels predictions;
    arma::mat probabilities;
    arma::mat feats = arma::conv_to<arma::mat>::from(features);
    if (m_use_scaling) {
        m_scaler.Transform(feats, feats);
    }
    m_rf.Classify(feats, predictions, probabilities);

    return {predictions, probabilities};
}

template <typename Features, typename Labels, statistics::AverageType AverageType,
          typename Scaler>
statistics::Metrics
RandomForestClassifier<Features, Labels, AverageType, Scaler>::evaluate(
    const Features &features, const Labels &labels) {

    // Get predictions and probabilities
    auto [predictions, probabilities] = predict_proba(features);

    // Calculate metrics
    const arma::rowvec positive_scores = probabilities.row(1);
    return statistics::calculate_metrics(predictions, labels, positive_scores,
                                         AverageType, m_num_classes);
}

template <typename Features, typename Labels, statistics::AverageType AverageType,
          typename Scaler>
void RandomForestClassifier<Features, Labels, AverageType, Scaler>::save(
    const std::string &path) const {
    try {
        // Open a single output archive
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) {
            throw RandomForestException("Could not open file " + path + " for writing");
        }

        cereal::BinaryOutputArchive archive(ofs);

        // Save all data to the same archive
        archive(CEREAL_NVP(m_rf), CEREAL_NVP(m_scaler), CEREAL_NVP(m_num_classes),
                CEREAL_NVP(m_num_trees), CEREAL_NVP(m_min_leaf_size),
                CEREAL_NVP(m_min_gain_split), CEREAL_NVP(m_max_depth),
                CEREAL_NVP(m_use_scaling));
    } catch (const std::exception &e) {
        std::throw_with_nested(
            RandomForestException("Failed to save the Random Forest model to " + path));
    }

    std::cout << "Classifier saved to " << path << std::endl;
}

template <typename Features, typename Labels, statistics::AverageType AverageType,
          typename Scaler>
void RandomForestClassifier<Features, Labels, AverageType, Scaler>::load(
    const std::string &path) {
    try {
        // Open a single input archive
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) {
            throw RandomForestException("Could not open file " + path + " for reading");
        }

        cereal::BinaryInputArchive archive(ifs);

        // Load all data from the same archive
        archive(CEREAL_NVP(m_rf), CEREAL_NVP(m_scaler), CEREAL_NVP(m_num_classes),
                CEREAL_NVP(m_num_trees), CEREAL_NVP(m_min_leaf_size),
                CEREAL_NVP(m_min_gain_split), CEREAL_NVP(m_max_depth),
                CEREAL_NVP(m_use_scaling));
    } catch (const std::exception &e) {
        std::throw_with_nested(
            RandomForestException("Failed to load the Random Forest model from " + path));
    }

    if (AverageType == statistics::AverageType::BINARY && m_num_classes != 2) {
        throw RandomForestException(
            "Binary average strategy can only be used with 2 classes.");
    }

    std::cout << "Classifier loaded from " << path << "\n"
              << "with " << m_num_trees << " trees, min leaf size " << m_min_leaf_size
              << ", min gain split " << m_min_gain_split << ", and max depth "
              << m_max_depth << std::endl;
}

template <typename Features, typename Labels, statistics::AverageType AverageType,
          typename Scaler>
template <typename Metric>
std::tuple<RandomForestParams, double>
RandomForestClassifier<Features, Labels, AverageType, Scaler>::grid_search(
    const Features &features, const Labels &labels, const std::size_t num_classes,
    const std::vector<std::size_t> &num_trees,
    const std::vector<std::size_t> &min_leaf_size,
    const std::vector<double> &min_gain_split, const std::vector<std::size_t> &max_depth,
    const double validation_size /*= 0.3*/, bool use_scaling /*= true*/,
    bool use_weights /*= false*/) {
    std::cout << "Performing grid search for Random Forest hyperparameters." << std::endl;
    RandomForestParams best_params;
    double best_result = 0;

    if (num_classes < 2) {
        throw RandomForestException("Number of classes must be at least 2.");
    }

    if (num_trees.empty() || min_leaf_size.empty() || min_gain_split.empty() ||
        max_depth.empty()) {
        throw RandomForestException(
            "Grid search requires non-empty hyperparameter lists.");
    }

    arma::mat feats = arma::conv_to<arma::mat>::from(features);
    if (use_scaling) {
        Scaler scaler;
        scaler.Fit(feats);
        scaler.Transform(feats, feats);
    }

    arma::rowvec weights;
    if (use_weights) {
        calculate_weights(labels, weights, num_classes);

        // Create tuner with weights
        mlpack::HyperParameterTuner<mlpack::RandomForest<>, Metric, mlpack::SimpleCV,
                                    ens::GridSearch>
            tuner(validation_size, feats, labels, static_cast<size_t>(num_classes),
                  weights);

        std::tie(best_params.num_trees, best_params.min_leaf_size,
                 best_params.min_gain_split, best_params.max_depth) =
            tuner.Optimize(num_trees, min_leaf_size, min_gain_split, max_depth);
        best_result = tuner.BestObjective();
    } else {
        // Regular tuner without weights
        mlpack::HyperParameterTuner<mlpack::RandomForest<>, Metric, mlpack::SimpleCV,
                                    ens::GridSearch>
            tuner(validation_size, feats, labels, static_cast<size_t>(num_classes));

        std::tie(best_params.num_trees, best_params.min_leaf_size,
                 best_params.min_gain_split, best_params.max_depth) =
            tuner.Optimize(num_trees, min_leaf_size, min_gain_split, max_depth);
        best_result = tuner.BestObjective();
    }

    std::cout << "Best hyperparameters:\n";
    std::cout << "  numTrees:     " << best_params.num_trees << "\n";
    std::cout << "  minLeafSize:  " << best_params.min_leaf_size << "\n";
    std::cout << "  maxDepth:     " << best_params.max_depth << "\n";
    std::cout << "  minGainSplit: " << best_params.min_gain_split << "\n";
    std::cout << "  Best objective: " << best_result << "\n";

    return std::make_tuple(best_params, best_result);
}

template <typename Features, typename Labels, statistics::AverageType AverageType,
          typename Scaler>
template <typename Metric>
std::tuple<RandomForestParams, double>
RandomForestClassifier<Features, Labels, AverageType, Scaler>::grid_search(
    const Features &features, const Labels &labels, const std::size_t num_classes,
    const GridSearchParams &params, bool use_scaling /*= true*/,
    bool use_weights /*= false*/) {
    return grid_search<Metric>(features, labels, num_classes, params.num_trees,
                               params.min_leaf_size, params.min_gain_split,
                               params.max_depth, params.validation_size, use_scaling,
                               use_weights);
}