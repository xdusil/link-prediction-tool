#pragma once

#include "../statistics/metrics.hpp"
#include "RandomForestClassifier.hpp"
#include "exceptions/exceptions.hpp"
#include "mlpack/core/cv/metrics/accuracy.hpp"
#include "mlpack/core/cv/metrics/average_strategy.hpp"
#include "mlpack/core/cv/metrics/metrics.hpp"
#include "utils/utils.hpp"
#include <ostream>

template <typename Features, typename Labels, mlpack::AverageStrategy AverageStrategy>
RandomForestClassifier<Features, Labels, AverageStrategy>::RandomForestClassifier(
    std::size_t num_classes /*= 2*/, std::size_t num_trees /*= 10*/, std::size_t min_leaf_size /*= 1*/,
    double min_gain_split /*= 0.0*/, std::size_t max_depth /*= 0*/)
    : m_num_classes(num_classes), m_num_trees(num_trees), m_min_leaf_size(min_leaf_size),
      m_min_gain_split(min_gain_split), m_max_depth(max_depth) {
    if (AverageStrategy == mlpack::AverageStrategy::Binary && num_classes != 2) {
        throw RandomForestException(
            "Binary average strategy can only be used with 2 classes.");
    }
}

template <typename Features, typename Labels, mlpack::AverageStrategy AverageStrategy>
RandomForestClassifier<Features, Labels, AverageStrategy>::RandomForestClassifier(
    const RandomForestParams &params)
    : RandomForestClassifier(params.num_classes, params.num_trees, params.min_leaf_size,
                             params.min_gain_split, params.max_depth) {}


template <typename Features, typename Labels, mlpack::AverageStrategy AverageStrategy>
void RandomForestClassifier<Features, Labels, AverageStrategy>::train(
    const Features &features, const Labels &labels) {

    std::cout << "Training Random Forest with " << m_num_classes << " classes."
              << std::endl;
    std::cout << "Features: " << features.n_rows << " x " << features.n_cols << std::endl;

    // Train the Random Forest
    m_rf.Train(features, labels, m_num_classes, m_num_trees, m_min_leaf_size, m_min_gain_split,
               m_max_depth);
}

template <typename Features, typename Labels, mlpack::AverageStrategy AverageStrategy>
Labels RandomForestClassifier<Features, Labels, AverageStrategy>::predict(const Features &features) const {
    // Predict using the Random Forest
    Labels predictions;
    m_rf.Classify(features, predictions);

    return predictions;
}


template <typename Features, typename Labels, mlpack::AverageStrategy AverageStrategy>
statistics::Metrics RandomForestClassifier<Features, Labels, AverageStrategy>::evaluate(const Features &features,
                                                          const Labels &labels) {

    // Predict using the Random Forest
    Labels predictions;
    m_rf.Classify(features, predictions);

    statistics::Metrics metrics;
    metrics.accuracy = mlpack::Accuracy::Evaluate(m_rf, features, labels);
    metrics.precision = mlpack::Precision<AverageStrategy>::Evaluate(m_rf, features, labels);
    metrics.recall = mlpack::Recall<AverageStrategy>::Evaluate(m_rf, features, labels);
    metrics.f1_score = mlpack::F1<AverageStrategy>::Evaluate(m_rf, features, labels);
    return metrics;
}

template <typename Features, typename Labels, mlpack::AverageStrategy AverageStrategy>
void RandomForestClassifier<Features, Labels, AverageStrategy>::save(const std::string &path) const {
    bool res = mlpack::data::Save(path, "rf_model", m_rf);
    if (!res) {
        throw RandomForestException("Failed to save the Random Forest model to " + path);
    }
}

template <typename Features, typename Labels, mlpack::AverageStrategy AverageStrategy>
void RandomForestClassifier<Features, Labels, AverageStrategy>::load(const std::string &path) {
    bool res = mlpack::data::Load(path, "rf_model", m_rf);
    if (!res) {
        throw RandomForestException("Failed to load the Random Forest model from " + path);
    }
}