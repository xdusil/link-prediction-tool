#pragma once

#include "BinaryRandomForestClassifier.hpp"
#include "exceptions/exceptions.hpp"
#include "statistics/metrics.hpp"

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
    const Features &features, const Labels &labels, bool use_weights,
    const std::string &metric /*= f1*/, double validation_size /*= 0.25*/) {

    Features train_features, test_features;
    Labels train_labels, test_labels;
    mlpack::data::StratifiedSplit(features, labels, train_features, test_features,
                                  train_labels, test_labels, validation_size);

    // First call the base class training
    Base::train(train_features, train_labels, use_weights);

    std::cout << "Binary classifier: Training complete, now calibrating threshold..."
              << std::endl;

    // After training, calibrate the threshold
    find_optimal_threshold(test_features, test_labels, metric, 0.01, 0.99, 0.02);

    std::cout << "Binary classifier: Calibration complete with threshold = "
              << m_binary_threshold << std::endl;
}

template <typename Features, typename Labels, typename Scaler>
Labels BinaryRandomForestClassifier<Features, Labels, Scaler>::predict(
    const Features &features) const {

    // Get probabilities for each class
    auto [_, probabilities] = Base::predict_proba(features);
    Labels predictions(probabilities.n_cols);

    // Apply binary threshold to the positive class probability (class 1)
    for (size_t i = 0; i < probabilities.n_cols; ++i) {
        predictions[i] = (probabilities(1, i) > m_binary_threshold) ? 1 : 0;
    }

    std::cout << "==>Using optimized binary threshold: " << m_binary_threshold
              << std::endl;
    return predictions;
}

template <typename Features, typename Labels, typename Scaler>
statistics::Metrics
BinaryRandomForestClassifier<Features, Labels, Scaler>::evaluate(const Features &features,
                                                                 const Labels &labels) {

    // Get probabilities for each class
    auto [_, probabilities] = Base::predict_proba(features);
    Labels predictions(probabilities.n_cols);

    // Apply binary threshold to the positive class probability
    for (size_t i = 0; i < predictions.n_elem; ++i) {
        predictions[i] = (probabilities(1, i) > m_binary_threshold) ? 1 : 0;
    }

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
    std::cout << "Binary threshold manually set to: " << threshold << std::endl;
}

template <typename Features, typename Labels, typename Scaler>
void BinaryRandomForestClassifier<Features, Labels, Scaler>::calibrate_threshold(
    const Features &val_features, const Labels &val_labels, const std::string &metric /*= f1*/) {

    // Default calibration range with 0.05 steps
    find_optimal_threshold(val_features, val_labels, metric, 0.05, 0.95, 0.05);
}

template <typename Features, typename Labels, typename Scaler>
void BinaryRandomForestClassifier<Features, Labels, Scaler>::find_optimal_threshold(
    const Features &val_features, const Labels &val_labels, const std::string &metric,
    double min_threshold /*= 0.01 */, double max_threshold /*= 0.99 */, double step /*= 0.01*/) {

    // Get probabilities for each class
    auto [_, probabilities] = Base::predict_proba(val_features);

    // Binary classification threshold optimization
    double best_value = 0.0;
    double best_threshold = 0.5;
    statistics::Metrics best{};

    // Try different thresholds
    for (double t = min_threshold; t <= max_threshold; t += step) {
        Labels preds(val_labels.n_elem);

        // Apply current threshold
        for (size_t i = 0; i < val_labels.n_elem; ++i) {
            preds[i] = (probabilities(1, i) > t) ? 1 : 0;
        }

        // Calculate metrics
        statistics::Metrics eval = statistics::calculate_metrics(
            preds, val_labels, arma::rowvec(), statistics::AverageType::BINARY, 2);

        // Update best threshold based on chosen metric
        double current_value = 0.0;

        if (metric == "f1") {
            current_value = eval.f1_score;
        } else if (metric == "precision") {
            current_value = eval.precision;
        } else if (metric == "recall") {
            current_value = eval.recall;
        } else if (metric == "accuracy") {
            current_value = eval.accuracy;
        } else {
            current_value = eval.accuracy; // Default to accuracy
        }

        if (current_value > best_value) {
            best_value = current_value;
            best_threshold = t;
            best = eval;
        }
    }

    // Save the best threshold
    m_binary_threshold = best_threshold;

    std::cout << "\n=== Binary Classifier Threshold Calibration ===" << std::endl;
    std::cout << "---Calibrated threshold: " << best_threshold << " (" << metric << "="
              << best_value << ")" << std::endl;
    std::cout << "---Accuracy: " << best.accuracy << std::endl;
    std::cout << "---Precision: " << best.precision << std::endl;
    std::cout << "---Recall: " << best.recall << std::endl;
    std::cout << "---F1 score: " << best.f1_score << std::endl;
    if (best.roc_auc.has_value()) {
        std::cout << "---ROC-AUC: " << best.roc_auc.value() << std::endl;
    }
    std::cout << "================================================\n" << std::endl;
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
                  << " with threshold = " << m_binary_threshold << ", " << 
                    "num trees = " << this->m_num_trees << ", min leaf size = " <<
                    this->m_min_leaf_size << ", min gain split = " << this->m_min_gain_split <<
                    ", max depth = " << this->m_max_depth << std::endl;

    } catch (const std::exception &e) {
        std::throw_with_nested(
            RandomForestException("Failed to load binary classifier from " + path));
    }
}