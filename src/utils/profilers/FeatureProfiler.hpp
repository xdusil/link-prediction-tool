#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace profilers {

/**
 * @brief Profiler for measuring individual feature computation times.
 *
 * Accumulates timing data for each feature across all node pairs,
 * then prints sorted results showing the most expensive features.
 */
class FeatureProfiler {
public:
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::micro>; // microseconds

    /**
     * @brief Start timing a feature.
     * @param feature_name Name of the feature being computed.
     */
    void start(const std::string& feature_name) {
        m_current_feature = feature_name;
        m_start_time = Clock::now();
    }

    /**
     * @brief Stop timing the current feature and accumulate the duration.
     */
    void stop() {
        auto end_time = Clock::now();
        Duration elapsed = end_time - m_start_time;

        m_timings[m_current_feature] += elapsed.count();
        m_counts[m_current_feature]++;
    }

    /**
     * @brief Print all timing results sorted by total time (descending).
     */
    void print_results() const {
        if (m_timings.empty()) {
            std::cout << "No profiling data collected.\n";
            return;
        }

        // Sort by total time descending
        std::vector<std::pair<std::string, double>> sorted_timings(m_timings.begin(),
                                                                   m_timings.end());

        std::sort(sorted_timings.begin(), sorted_timings.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        double total_time = 0.0;
        for (const auto& [_, time] : m_timings) {
            total_time += time;
        }

        std::cout << "\n=== Feature Profiling Results ===\n";
        std::cout << "Total time: " << (total_time / 1000.0) << " ms\n\n";
        std::cout << "Rank | Feature Name                              | Total (ms) | "
                     "Avg (μs) | Count    | % Time\n";
        std::cout << "-----+-------------------------------------------+------------+----"
                     "------+----------+-------\n";

        int rank = 1;
        for (const auto& [name, time_us] : sorted_timings) {
            size_t count = m_counts.at(name);
            double avg_us = time_us / count;
            double pct = (time_us / total_time) * 100.0;

            printf("%4d | %-41s | %10.3f | %8.3f | %8zu | %5.1f%%\n", rank++,
                   name.c_str(), time_us / 1000.0, avg_us, count, pct);
        }
        std::cout << "=== End Profiling Results ===\n\n";
    }

    /**
     * @brief Reset all collected timing data.
     */
    void reset() {
        m_timings.clear();
        m_counts.clear();
    }

private:
    std::map<std::string, double> m_timings; // Accumulated time in microseconds
    std::map<std::string, size_t> m_counts;  // Number of times each feature was computed
    std::string m_current_feature;
    Clock::time_point m_start_time;
};

/**
 * @brief RAII helper for automatic start/stop timing.
 */
class ScopedFeatureTimer {
public:
    ScopedFeatureTimer(FeatureProfiler& profiler, const std::string& feature_name)
        : m_profiler(profiler) {
        m_profiler.start(feature_name);
    }

    ~ScopedFeatureTimer() { m_profiler.stop(); }

private:
    FeatureProfiler& m_profiler;
};

} // namespace profilers

// Convenience macro for timing
#define PROFILE_FEATURE(profiler, name) profilers::ScopedFeatureTimer _timer##__LINE__(profiler, name)