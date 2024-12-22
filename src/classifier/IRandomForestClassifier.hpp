#pragma once
#include <string>
#include <string>
#include <vector>

/**
 * @brief Interface for a Random Forest classifier.
 *
 * @tparam Features The type of the features.
 * @tparam Labels The type of the labels.
 */
template <typename Features, typename Labels>
class IRandomForestClassifier {
public:
    virtual ~IRandomForestClassifier() = default;

    /**
     * @brief Train the classifier with the given features and labels.
     *
     * @param features The features.
     * @param labels The labels.
     */
    virtual void train(const Features& features,
                       const Labels& labels) = 0;

    /**
     * @brief Predict the labels for the given features.
     *
     * @param features The features.
     * @return The predicted labels.
     */
    virtual Labels predict(const Features& features) = 0;

    /**
     * @brief Evaluate the classifier using the given features and labels.
     *
     * @param features The features.
     * @param labels The labels.
     * @return The accuracy of the classifier.
     */
    virtual double evaluate(const Features& features, const Labels& labels) = 0;

    /**
     * @brief Save the classifier to a file.
     *
     * @param path The file path to save the classifier.
     */
    virtual void save(const std::string& path) const = 0;

    /**
     * @brief Load the classifier from a file.
     *
     * @param path The file path to load the classifier from.
     */
    virtual void load(const std::string& path) = 0;
};