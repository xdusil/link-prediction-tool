#pragma once

#include "BinaryRandomForestClassifier.hpp"
#include "exceptions/exceptions.hpp"
#include "statistics/metrics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

template <typename Features, typename Labels, typename Scaler>
BinaryRandomForestClassifier<Features, Labels, Scaler>::BinaryRandomForestClassifier(
    std::size_t num_trees /*= 10*/, std::size_t min_leaf_size /*= 1*/,
    double min_gain_split /*= 0.0*/, std::size_t max_depth /*= 0*/,
    bool use_scaling /*= true*/)
    : Base(2, num_trees, min_leaf_size, min_gain_split, max_depth, use_scaling),
      m_binary_threshold(0.5) {
    // Binary classifier always has exactly 2 classes
    // Base constructor already sets up everything else
}

template <typename Features, typename Labels, typename Scaler>
BinaryRandomForestClassifier<Features, Labels, Scaler>::BinaryRandomForestClassifier(
    const RandomForestParams &params, bool use_scaling /*= true*/)
    : Base(2, params, use_scaling), m_binary_threshold(0.5) {
    // Binary classifier always has exactly 2 classes
    // Base constructor already sets up everything else
}

template <typename Features, typename Labels, typename Scaler>
void BinaryRandomForestClassifier<Features, Labels, Scaler>::train_with_calibration(
    const Features &features,
    const Labels &labels,
    bool use_weights,
    const std::string &metric /*= f1*/,
    std::size_t num_folds /*= 5*/) {
    if (features.n_cols != labels.n_elem) {
        throw RandomForestException("Number of features and labels must be equal.");
    }
    if (num_folds < 2) {
        throw RandomForestException("Threshold calibration requires at least two folds.");
    }

    std::array<arma::uvec, 2> class_indices = {
        arma::find(labels == 0),
        arma::find(labels == 1)};
    if (class_indices[0].n_elem + class_indices[1].n_elem != labels.n_elem) {
        throw RandomForestException("Binary labels must be either 0 or 1.");
    }
    if (class_indices[0].is_empty() || class_indices[1].is_empty()) {
        throw RandomForestException(
            "Threshold calibration requires both binary classes.");
    }

    num_folds = std::min<std::size_t>(
        num_folds,
        std::min(class_indices[0].n_elem, class_indices[1].n_elem));
    std::vector<std::vector<arma::uword>> fold_indices(num_folds);
    for (arma::uvec &indices : class_indices) {
        indices = arma::shuffle(indices);
        for (arma::uword i = 0; i < indices.n_elem; ++i) {
            fold_indices[i % num_folds].push_back(indices[i]);
        }
    }

    std::cout << "Binary classifier: Generating " << num_folds
              << "-fold out-of-fold probabilities for threshold calibration..."
              << std::endl;

    arma::rowvec positive_scores(labels.n_elem);
    arma::uvec scored(labels.n_elem, arma::fill::zeros);
    const RandomForestParams params{
        this->m_num_trees,
        this->m_min_leaf_size,
        this->m_min_gain_split,
        this->m_max_depth};

    for (std::size_t fold = 0; fold < num_folds; ++fold) {
        arma::uvec validation_indices(fold_indices[fold]);
        validation_indices = arma::sort(validation_indices);

        arma::uvec is_validation(labels.n_elem, arma::fill::zeros);
        is_validation.elem(validation_indices).ones();
        const arma::uvec training_indices = arma::find(is_validation == 0);

        BinaryRandomForestClassifier fold_classifier(params, this->m_use_scaling);
        fold_classifier.train(
            features.cols(training_indices),
            labels.cols(training_indices),
            use_weights);
        const auto [_, probabilities] =
            fold_classifier.predict_proba(features.cols(validation_indices));
        positive_scores.elem(validation_indices) = probabilities.row(1);
        scored.elem(validation_indices).ones();
    }

    if (arma::any(scored == 0)) {
        throw RandomForestException(
            "Out-of-fold threshold calibration did not score every sample.");
    }

    const auto [threshold, metrics] =
        find_optimal_threshold(positive_scores, labels, metric);
    m_binary_threshold = threshold;

    std::cout << "Binary classifier: OOF calibration complete with threshold = "
              << m_binary_threshold << " (Accuracy = " << metrics.accuracy
              << ", Precision = " << metrics.precision << ", Recall = " << metrics.recall
              << ", F1 = " << metrics.f1_score << ")\n"
              << "Binary classifier: Training final model on all samples..." << std::endl;
    Base::train(features, labels, use_weights);
}

template <typename Features, typename Labels, typename Scaler>
std::tuple<Labels, arma::mat>
BinaryRandomForestClassifier<Features, Labels, Scaler>::predict_proba(
    const Features &features) const {
    auto [_, probabilities] = Base::predict_proba(features);
    if (probabilities.n_rows != 2) {
        throw RandomForestException(
            "Binary classifier expected probabilities for two classes.");
    }

    Labels predictions(probabilities.n_cols);

    // Apply binary threshold to the positive class probability (class 1)
    for (size_t i = 0; i < probabilities.n_cols; ++i) {
        predictions[i] = (probabilities(1, i) > m_binary_threshold) ? 1 : 0;
    }

    return {std::move(predictions), std::move(probabilities)};
}

template <typename Features, typename Labels, typename Scaler>
Labels BinaryRandomForestClassifier<Features, Labels, Scaler>::predict(
    const Features &features) const {
    auto [predictions, _] = predict_proba(features);
    std::cout << "Binary classifier: Using optimized binary threshold: " << m_binary_threshold
              << std::endl;
    return predictions;
}

template <typename Features, typename Labels, typename Scaler>
statistics::Metrics
BinaryRandomForestClassifier<Features, Labels, Scaler>::evaluate(const Features &features,
                                                                 const Labels &labels) const {
    auto [predictions, probabilities] = predict_proba(features);

    // Calculate metrics including ROC AUC
    const arma::rowvec positive_scores = probabilities.row(1);
    return statistics::calculate_metrics(predictions, labels, positive_scores,
                                         statistics::AverageType::BINARY, 2);
}

template <typename Features, typename Labels, typename Scaler>
double BinaryRandomForestClassifier<Features, Labels, Scaler>::get_threshold() const {
    return m_binary_threshold;
}

template <typename Features, typename Labels, typename Scaler>
void BinaryRandomForestClassifier<Features, Labels, Scaler>::set_threshold(
    double threshold) {
    if (threshold < 0.0 || threshold > 1.0) {
        throw RandomForestException("Threshold must be between 0 and 1");
    }

    m_binary_threshold = threshold;
    std::cout << "Binary classifier: Binary threshold manually set to: " << threshold << std::endl;
}

template <typename Features, typename Labels, typename Scaler>
void BinaryRandomForestClassifier<Features, Labels, Scaler>::calibrate_threshold(
    const Features &val_features, const Labels &val_labels,
    const std::string &metric /*= f1*/) {

    auto [threshold, metrics] =
        find_optimal_threshold(val_features, val_labels, metric);
    m_binary_threshold = threshold;

    std::cout << "Binary classifier: Calibration complete with threshold = "
            << m_binary_threshold << " (Accuracy = " << metrics.accuracy
            << ", Precision = " << metrics.precision
            << ", Recall = " << metrics.recall
            << ", F1 = " << metrics.f1_score << ")" << std::endl;
}

template <typename Features, typename Labels, typename Scaler>
std::tuple<double, statistics::Metrics> BinaryRandomForestClassifier<
    Features,
    Labels,
    Scaler>::
    find_optimal_threshold(
        const arma::rowvec& positive_scores,
        const Labels& labels,
        const std::string& metric) const {
    if (positive_scores.n_elem != labels.n_elem || labels.is_empty()) {
        throw RandomForestException(
            "Threshold calibration scores must match non-empty labels.");
    }

    if (!positive_scores.is_finite() || arma::any(positive_scores < 0.0) ||
        arma::any(positive_scores > 1.0)) {
        throw RandomForestException(
            "Threshold calibration requires finite probabilities in [0, 1].");
    }

    std::vector<std::size_t> sorted_indices(labels.n_elem);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(
        sorted_indices.begin(),
        sorted_indices.end(),
        [&positive_scores](std::size_t lhs, std::size_t rhs) {
            if (positive_scores[lhs] == positive_scores[rhs]) {
                return lhs < rhs;
            }
            return positive_scores[lhs] < positive_scores[rhs];
        });

    double best_value = -std::numeric_limits<double>::infinity();
    double best_threshold = 0.5;
    std::size_t true_positives = arma::accu(labels == 1);
    std::size_t false_positives = labels.n_elem - true_positives;
    std::size_t true_negatives = 0;
    std::size_t false_negatives = 0;
    std::size_t position = 0;

    std::vector<double> thresholds;
    thresholds.reserve(labels.n_elem + 2);
    thresholds.push_back(0.0);
    for (const std::size_t index : sorted_indices) {
        thresholds.push_back(positive_scores[index]);
    }
    thresholds.push_back(1.0);
    std::sort(thresholds.begin(), thresholds.end());
    thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());

    for (const double threshold : thresholds) {
        while (position < sorted_indices.size() &&
               positive_scores[sorted_indices[position]] <= threshold) {
            if (labels[sorted_indices[position]] == 1) {
                --true_positives;
                ++false_negatives;
            } else {
                --false_positives;
                ++true_negatives;
            }
            ++position;
        }

        const statistics::Metrics metrics = statistics::calculate_binary_metrics(
            true_positives, false_positives, true_negatives, false_negatives);
        double value;
        if (metric == "f1") {
            value = metrics.f1_score;
        } else if (metric == "precision") {
            value = metrics.precision;
        } else if (metric == "recall") {
            value = metrics.recall;
        } else if (metric == "accuracy") {
            value = metrics.accuracy;
        } else {
            throw RandomForestException(
                "Unknown threshold calibration metric: " + metric);
        }

        // Prefer the threshold closest to 0.5 when the objective is tied.
        if (value > best_value ||
            (value == best_value &&
             std::abs(threshold - 0.5) < std::abs(best_threshold - 0.5))) {
            best_value = value;
            best_threshold = threshold;
        }
    }

    Labels predictions(labels.n_elem);
    for (std::size_t i = 0; i < labels.n_elem; ++i) {
        predictions[i] = positive_scores[i] > best_threshold ? 1 : 0;
    }
    const statistics::Metrics best = statistics::calculate_metrics(
        predictions,
        labels,
        arma::rowvec(),
        statistics::AverageType::BINARY,
        2);
    return {best_threshold, best};
}

template <typename Features, typename Labels, typename Scaler>
std::tuple<double, statistics::Metrics> BinaryRandomForestClassifier<
    Features,
    Labels,
    Scaler>::
    find_optimal_threshold(
        const Features& val_features,
        const Labels& val_labels,
        const std::string& metric) const {
    const auto [_, probabilities] = Base::predict_proba(val_features);
    if (probabilities.n_rows != 2) {
        throw RandomForestException(
            "Binary classifier expected probabilities for two classes.");
    }
    return find_optimal_threshold(probabilities.row(1), val_labels, metric);
}

template <typename Features, typename Labels, typename Scaler>
void BinaryRandomForestClassifier<Features, Labels, Scaler>::save(
    const std::string &path) const {
    try {
        // Open output file
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) {
            throw RandomForestException("Could not open file " + path + " for writing");
        }

        // Create archive
        cereal::BinaryOutputArchive archive(ofs);

        // Use cereal to serialize
        // This will call our serialize method which handles both base and derived class
        archive(*this);

        std::cout << "Binary classifier saved to " << path << std::endl;

    } catch (const std::exception &e) {
        std::throw_with_nested(
            RandomForestException("Failed to save binary classifier to " + path));
    }
}

template <typename Features, typename Labels, typename Scaler>
void BinaryRandomForestClassifier<Features, Labels, Scaler>::load(
    const std::string &path) {
    try {
        // Open input file
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) {
            throw RandomForestException("Could not open file " + path + " for reading");
        }

        // Create archive
        cereal::BinaryInputArchive archive(ifs);

        // Use cereal to deserialize
        // This will call our serialize method which handles both base and derived class
        archive(*this);

        std::cout << "Binary classifier loaded from " << path
                  << " with threshold = " << m_binary_threshold << ", "
                  << "num trees = " << this->m_num_trees
                  << ", min leaf size = " << this->m_min_leaf_size
                  << ", min gain split = " << this->m_min_gain_split
                  << ", max depth = " << this->m_max_depth
                  << ", with scaling: " << this->m_use_scaling << std::endl;

    } catch (const std::exception &e) {
        std::throw_with_nested(
            RandomForestException("Failed to load binary classifier from " + path));
    }
}

template <typename Features, typename Labels, typename Scaler>
template <class Archive>
void BinaryRandomForestClassifier<Features, Labels, Scaler>::serialize(Archive &archive) {
    // First serialize the base class
    archive(cereal::base_class<Base>(this));

    // Then serialize our members
    archive(CEREAL_NVP(m_binary_threshold));
}
