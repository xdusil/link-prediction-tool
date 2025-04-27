#pragma once

#include "../generic/IRandomForestClassifier.hpp"

/**
 * @brief Interface for a Binary Random Forest classifier with threshold optimization.
 *
 * @tparam Features The type of the features.
 * @tparam Labels The type of the labels.
 * @tparam Metrics The type of evaluation metrics.
 * @tparam ProbMatrix The type of probability matrix returned by predict_proba.
 */
template <typename Features, typename Labels, typename Metrics, typename ProbMatrix>
class IBinaryRandomForestClassifier
    : public virtual IRandomForestClassifier<Features, Labels, Metrics, ProbMatrix> {
public:
    virtual ~IBinaryRandomForestClassifier() = default;

    /**
     * @brief Train with automatic threshold calibration
     *
     * This extends the base training with threshold calibration
     * while preserving the original train() behavior.
     *
     * @param features Training features
     * @param labels Training labels
     * @param use_weights Whether to use class weights
     * @param metric The metric to optimize ("f1", "precision", etc.)
     * @param validation_size The size of the validation set
     */
    virtual void train_with_calibration(const Features &features, const Labels &labels,
                                        bool use_weights = false,
                                        const std::string &metric = "f1", double validation_size = 0.25) = 0;

    /**
     * @brief Get the current decision threshold.
     *
     * @return Current threshold value between 0 and 1.
     */
    virtual double get_threshold() const = 0;

    /**
     * @brief Set the decision threshold manually.
     *
     * @param threshold Threshold value between 0 and 1.
     */
    virtual void set_threshold(double threshold) = 0;

    /**
     * @brief Calibrate the decision threshold on validation data.
     *
     * @param val_features Validation features
     * @param val_labels Validation labels
     * @param metric Metric to optimize (f1, precision, recall, accuracy)
     */
    virtual void calibrate_threshold(const Features &val_features,
                                     const Labels &val_labels,
                                     const std::string &metric = "f1") = 0;
};