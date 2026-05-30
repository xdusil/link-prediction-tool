#include "metrics.hpp"
#include "exceptions/exceptions.hpp"
#include "mlpack/core/cv/metrics/roc_auc_score.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace statistics {

namespace {

constexpr std::array<std::size_t, 3> DEFAULT_RANKING_CUTOFFS = {10, 50, 100};

std::vector<std::size_t> sorted_score_indices_desc(const arma::rowvec &scores) {
    std::vector<std::size_t> indices(scores.n_elem);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&scores](std::size_t lhs, std::size_t rhs) {
        if (scores[lhs] == scores[rhs]) {
            return lhs < rhs;
        }
        return scores[lhs] > scores[rhs];
    });
    return indices;
}

std::optional<double> precision_at_k(const std::vector<std::size_t> &ranked_indices,
                                     const arma::Row<size_t> &labels,
                                     std::size_t k) {
    const std::size_t limit = std::min(k, ranked_indices.size());
    if (limit == 0) {
        return std::nullopt;
    }

    std::size_t positives = 0;
    for (std::size_t i = 0; i < limit; ++i) {
        const std::size_t index = ranked_indices[i];
        assert(index < labels.n_elem);
        positives += labels[index] == 1 ? 1 : 0;
    }

    return static_cast<double>(positives) / static_cast<double>(limit);
}

std::optional<double> recall_at_k(const std::vector<std::size_t> &ranked_indices,
                                  const arma::Row<size_t> &labels, std::size_t k) {
    const std::size_t total_positives = arma::sum(labels == 1);
    const std::size_t limit = std::min(k, ranked_indices.size());
    if (limit == 0 || total_positives == 0) {
        return std::nullopt;
    }

    std::size_t positives = 0;
    for (std::size_t i = 0; i < limit; ++i) {
        const std::size_t index = ranked_indices[i];
        assert(index < labels.n_elem);
        positives += labels[index] == 1 ? 1 : 0;
    }

    return static_cast<double>(positives) / static_cast<double>(total_positives);
}

std::optional<double> average_precision(
    const std::vector<std::size_t> &ranked_indices,
    const arma::Row<size_t> &labels) {
    const std::size_t total_positives = arma::sum(labels == 1);
    if (ranked_indices.empty() || total_positives == 0) {
        return std::nullopt;
    }

    double precision_sum = 0.0;
    std::size_t true_positives = 0;

    for (std::size_t rank = 0; rank < ranked_indices.size(); ++rank) {
        const std::size_t index = ranked_indices[rank];
        assert(index < labels.n_elem);
        if (labels[index] != 1) {
            continue;
        }

        ++true_positives;
        precision_sum += static_cast<double>(true_positives) /
                         static_cast<double>(rank + 1);
    }

    return precision_sum / static_cast<double>(total_positives);
}

std::optional<double> reciprocal_rank(
    const std::vector<std::size_t>& ranked_indices,
    const arma::Row<std::size_t>& labels) {
    for (std::size_t rank = 0; rank < ranked_indices.size(); ++rank) {
        const std::size_t index = ranked_indices[rank];
        assert(index < labels.n_elem);
        if (labels[index] == 1) {
            return 1.0 / static_cast<double>(rank + 1);
        }
    }

    return std::nullopt;
}

void fill_score_based_metrics(Metrics &metrics, const arma::rowvec &scores,
                              const arma::Row<size_t> &labels) {
    if (scores.is_empty()) {
        return;
    }

    if (scores.n_elem != labels.n_elem) {
        throw std::invalid_argument(
            "Positive score count does not match label count.");
    }

    metrics.roc_auc = calculate_roc_auc(scores, labels);

    const auto ranked_indices = sorted_score_indices_desc(scores);
    metrics.average_precision = average_precision(ranked_indices, labels);
    metrics.mean_reciprocal_rank = reciprocal_rank(ranked_indices, labels);
    metrics.ranking_at_k.reserve(DEFAULT_RANKING_CUTOFFS.size());
    for (const std::size_t k : DEFAULT_RANKING_CUTOFFS) {
        metrics.ranking_at_k.push_back(
            {k, precision_at_k(ranked_indices, labels, k),
             recall_at_k(ranked_indices, labels, k)});
    }
}

} // namespace

Metrics calculate_metrics(const arma::Row<size_t> &predictions,
                          const arma::Row<size_t> &labels,
                          const arma::rowvec &positive_scores /*= arma::rowvec()*/,
                          AverageType avg_type /*= AverageType::BINARY*/,
                          size_t num_classes /*= 2*/) {
    Metrics metrics{};
    if (labels.is_empty()) {
        return metrics;
    }
    if (predictions.n_elem != labels.n_elem) {
        throw std::invalid_argument("Prediction count does not match label count.");
    }

    // Handle simple binary case with direct calculation
    if (avg_type == AverageType::BINARY && num_classes == 2) {
        size_t tp = arma::sum((labels == 1) % (predictions == 1));
        size_t fp = arma::sum((labels == 0) % (predictions == 1));
        size_t tn = arma::sum((labels == 0) % (predictions == 0));
        size_t fn = arma::sum((labels == 1) % (predictions == 0));

        metrics.accuracy = static_cast<double>(tp + tn) / labels.n_elem;
        metrics.precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
        metrics.recall = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
        metrics.f1_score = (metrics.precision + metrics.recall > 0)
                               ? 2.0 * metrics.precision * metrics.recall /
                                     (metrics.precision + metrics.recall)
                               : 0.0;

        fill_score_based_metrics(metrics, positive_scores, labels);
        return metrics;
    }

    // Determine number of classes if not provided
    if (num_classes == 0) {
        num_classes = arma::max(labels) + 1;
    }

    // Calculate accuracy (same for all averaging methods)
    metrics.accuracy =
        arma::accu(predictions == labels) / static_cast<double>(labels.n_elem);

    // For multiclass metrics, we need to calculate per-class values
    std::vector<double> class_precision(num_classes, 0.0);
    std::vector<double> class_recall(num_classes, 0.0);
    std::vector<double> class_f1(num_classes, 0.0);

    // For micro-averaging
    size_t total_tp = 0;
    size_t total_fp = 0;
    size_t total_fn = 0;

    // Calculate metrics for each class
    for (size_t c = 0; c < num_classes; c++) {
        // One-vs-rest approach
        size_t tp = arma::sum((labels == c) % (predictions == c));
        size_t fp = arma::sum((labels != c) % (predictions == c));
        size_t fn = arma::sum((labels == c) % (predictions != c));

        // Per-class metrics
        class_precision[c] = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
        class_recall[c] = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
        class_f1[c] = (class_precision[c] + class_recall[c] > 0)
                          ? 2.0 * class_precision[c] * class_recall[c] /
                                (class_precision[c] + class_recall[c])
                          : 0.0;

        // Add to totals for micro-average
        total_tp += tp;
        total_fp += fp;
        total_fn += fn;
    }

    switch (avg_type) {
    case AverageType::MICRO:
        metrics.precision = (total_tp + total_fp > 0)
                                ? static_cast<double>(total_tp) / (total_tp + total_fp)
                                : 0.0;
        metrics.recall = (total_tp + total_fn > 0)
                             ? static_cast<double>(total_tp) / (total_tp + total_fn)
                             : 0.0;
        metrics.f1_score = (metrics.precision + metrics.recall > 0)
                               ? 2.0 * metrics.precision * metrics.recall /
                                     (metrics.precision + metrics.recall)
                               : 0.0;
        break;

    case AverageType::MACRO:
        // Simple average across classes
        metrics.precision =
            std::accumulate(class_precision.begin(), class_precision.end(), 0.0) /
            num_classes;
        metrics.recall =
            std::accumulate(class_recall.begin(), class_recall.end(), 0.0) / num_classes;
        metrics.f1_score =
            std::accumulate(class_f1.begin(), class_f1.end(), 0.0) / num_classes;
        break;

    default:
        throw NotSupportedException(
            "Unsupported averaging strategy for multiclass metrics.");
    }

    return metrics;
}

std::optional<double> calculate_roc_auc(const arma::rowvec &scores,
                                        const arma::Row<size_t> &labels) {
    if (scores.n_elem != labels.n_elem) {
        throw std::invalid_argument("Score count does not match label count.");
    }

    const std::size_t positives = arma::sum(labels == 1);
    if (scores.is_empty() || positives == 0 || positives == labels.n_elem) {
        return std::nullopt;
    }

    return mlpack::ROCAUCScore<>::Evaluate(labels, scores);
}

} // namespace statistics
