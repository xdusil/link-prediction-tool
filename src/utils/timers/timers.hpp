#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace utils {

/**
 * @brief Simple timer class for measuring execution time
 */
class Timer {
public:
    /**
     * @brief Construct a new Timer object
     *
     * @param name Name of the timer for identification in output
     * @param auto_start Whether to start the timer immediately
     * @param auto_print Whether to print the duration when stopped
     */
    explicit Timer(std::string name = "Timer", bool auto_start = true,
                   bool auto_print = true);

    /**
     * @brief Start the timer
     */
    void start();

    /**
     * @brief Stop the timer and optionally print the duration
     *
     * @param print_duration Whether to print the duration
     * @return double Duration in seconds
     */
    double stop(bool print_duration = true);

    /**
     * @brief Get the elapsed time without stopping the timer
     *
     * @return double Elapsed time in seconds
     */
    double elapsed() const;

    /**
     * @brief Format a duration in seconds to a readable string
     *
     * @param seconds Duration in seconds
     * @return std::string Formatted duration string
     */
    static std::string format_duration(double seconds);

    /**
     * @brief Destructor - stops the timer if still running
     */
    ~Timer();

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
    std::string m_name;
    bool m_is_running = false;
    bool m_auto_print;
};

/**
 * @brief Timer that checks a global verbose state
 */
class VerboseTimer {
public:
    /**
     * @brief Set global verbose state
     */
    static void set_verbose(bool verbose);

    /**
     * @brief Create a timer that only activates if verbose mode is enabled
     */
    explicit VerboseTimer(const std::string &name);

    ~VerboseTimer() = default;

private:
    
    inline static bool m_verbose_enabled = false;
    std::unique_ptr<Timer> m_timer;
};
} // namespace utils