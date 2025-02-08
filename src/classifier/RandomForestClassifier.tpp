#pragma once

#include "../statistics/metrics.hpp"
#include "RandomForestClassifier.hpp"
#include "utils/utils.hpp"
#include <ostream>

// Train the classifier with the given features and labels.
template <typename Features, typename Labels>
void RandomForestClassifier<Features, Labels>::train(const Features &features,
                                                     const Labels &labels) {

    std::cout << "Training Random Forest with " << m_num_classes << " classes."
              << std::endl;
    std::cout << "Features: " << features.n_rows << " x " << features.n_cols
              << std::endl;

    // Train the Random Forest
    m_rf.Train(features, labels, m_num_classes, m_num_trees);
}

// Predict the labels for the given features.
template <typename Features, typename Labels>
Labels RandomForestClassifier<Features, Labels>::predict(const Features &features) {
    // Predict using the Random Forest
    Labels predictions;
    m_rf.Classify(features, predictions);

    return predictions;
}

// Evaluate the classifier using the given features and labels.
template <typename Features, typename Labels>
double RandomForestClassifier<Features, Labels>::evaluate(const Features &features,
                                                          const Labels &labels) {

    // Predict using the Random Forest
    Labels predictions;
    m_rf.Classify(features, predictions);

    // Calculate accuracy TODO
    return 0;
}

// Save the classifier to a file.
template <typename Features, typename Labels>
void RandomForestClassifier<Features, Labels>::save(const std::string &path) const {
    mlpack::data::Save(path, "rf_model", m_rf, true);
}

// Load the classifier from a file.
template <typename Features, typename Labels>
void RandomForestClassifier<Features, Labels>::load(const std::string &path) {
    mlpack::data::Load(path, "rf_model", m_rf, true);
}