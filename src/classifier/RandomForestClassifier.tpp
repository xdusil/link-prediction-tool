#pragma once

#include "RandomForestClassifier.hpp"
#include "../statistics/metrics.hpp"
#include <ostream>

// Train the classifier with the given features and labels.
template <typename Features, typename Labels>
void RandomForestClassifier<Features, Labels>::train(const Features& features,
                                                     const Labels& labels) {
    // Scale the features
    arma::mat scaled_features;
    m_scaler.Fit(features);
    m_scaler.Transform(features, scaled_features);

    // Train the Random Forest
    m_rf.Train(scaled_features, labels, m_num_classes, m_num_trees);
}

// Predict the labels for the given features.
template <typename Features, typename Labels>
Labels RandomForestClassifier<Features, Labels>::predict(const Features& features) {
    // Scale the features
    arma::mat scaled_features;
    m_scaler.Transform(features, scaled_features);

    // Predict using the Random Forest
    Labels predictions;
    m_rf.Classify(scaled_features, predictions);

    return predictions;
}

// Evaluate the classifier using the given features and labels.
template <typename Features, typename Labels>
double RandomForestClassifier<Features, Labels>::evaluate(const Features& features, const Labels& labels) {
    // Scale the features
    arma::mat scaled_features;
    m_scaler.Transform(features, scaled_features);

    // Predict using the Random Forest
    Labels predictions;
    m_rf.Classify(scaled_features, predictions);

    // Calculate accuracy
    return statistics::calculate_accuracy(predictions, labels);
}

// Save the classifier to a file.
template <typename Features, typename Labels>
void RandomForestClassifier<Features, Labels>::save(const std::string& path) const {
    mlpack::data::Save(path, "rf_model", m_rf, true);
}

// Load the classifier from a file.
template <typename Features, typename Labels>
void RandomForestClassifier<Features, Labels>::load(const std::string& path) {
    mlpack::data::Load(path, "rf_model", m_rf, true);
}