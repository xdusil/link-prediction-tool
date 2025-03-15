#include "metrics.hpp"
#include "exceptions/exceptions.hpp"
#include "mlpack/core/cv/metrics/roc_auc_score.hpp"

namespace statistics {
Metrics calculate_metrics(const arma::Row<size_t> &predictions,
                          const arma::Row<size_t> &labels,
                          const arma::rowvec &positive_scores /*= arma::rowvec()*/,
                          AverageType avg_type /*= AverageType::BINARY*/,
                          size_t num_classes /*= 2*/) {
    Metrics metrics{};

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

        if (!positive_scores.is_empty()) {
            metrics.roc_auc = calculate_roc_auc(positive_scores, labels);
        }
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

double calculate_roc_auc(const arma::rowvec &scores, const arma::Row<size_t> &labels) {
    return mlpack::ROCAUCScore<>::Evaluate(labels, scores);
}
} // namespace statistics