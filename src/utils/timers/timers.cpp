#include "timers.hpp"

namespace utils {

Timer::Timer(std::string name /*= "Timer"*/, bool auto_start /*= true*/,
             bool auto_print /*= true*/)
    : m_name(std::move(name)), m_auto_print(auto_print) {
    if (auto_start) {
        start();
    }
}

void Timer::start() {
    m_start = std::chrono::high_resolution_clock::now();
    m_is_running = true;
}

double Timer::stop(bool print_duration /*= true*/) {
    auto duration = elapsed();
    m_is_running = false;

    if (print_duration && m_auto_print) {
        std::cout << m_name << ": " << format_duration(duration) << std::endl;
    }

    return duration;
}

double Timer::elapsed() const {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - m_start;
    return diff.count();
}

std::string Timer::format_duration(double seconds) {
    std::stringstream ss;

    if (seconds < 0.001) {
        // Microseconds
        ss << std::fixed << std::setprecision(2) << (seconds * 1000000) << " μs";
    } else if (seconds < 1.0) {
        // Milliseconds
        ss << std::fixed << std::setprecision(2) << (seconds * 1000) << " ms";
    } else if (seconds < 60.0) {
        // Seconds
        ss << std::fixed << std::setprecision(2) << seconds << " s";
    } else if (seconds < 3600.0) {
        // Minutes and seconds
        int minutes = static_cast<int>(seconds) / 60;
        double secs = seconds - (minutes * 60);
        ss << minutes << "m " << std::fixed << std::setprecision(2) << secs << "s";
    } else {
        // Hours, minutes and seconds
        int hours = static_cast<int>(seconds) / 3600;
        int minutes = (static_cast<int>(seconds) % 3600) / 60;
        double secs = seconds - (hours * 3600) - (minutes * 60);
        ss << hours << "h " << minutes << "m " << std::fixed << std::setprecision(2)
           << secs << "s";
    }

    return ss.str();
}

Timer::~Timer() {
    if (m_is_running) {
        stop();
    }
}

void VerboseTimer::set_verbose(bool verbose) { m_verbose_enabled = verbose; }

VerboseTimer::VerboseTimer(const std::string &name) {
    if (m_verbose_enabled) {
        m_timer = std::make_unique<Timer>(name, true, true);
    }
}
} // namespace utils