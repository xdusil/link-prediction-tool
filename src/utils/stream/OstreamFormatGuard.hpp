#pragma once

#include <ios>
#include <ostream>

namespace utils {

/**
 * @brief Restores an output stream's formatting state when leaving scope.
 *
 * Use this around code that temporarily changes flags or precision on shared streams
 * such as std::cout.
 */
class OstreamFormatGuard {
public:
    explicit OstreamFormatGuard(std::ostream& stream)
        : m_stream(stream), m_flags(stream.flags()), m_precision(stream.precision()) {}

    OstreamFormatGuard(const OstreamFormatGuard&) = delete;
    OstreamFormatGuard& operator=(const OstreamFormatGuard&) = delete;

    ~OstreamFormatGuard() {
        m_stream.flags(m_flags);
        m_stream.precision(m_precision);
    }

private:
    std::ostream& m_stream;
    std::ios_base::fmtflags m_flags;
    std::streamsize m_precision;
};

} // namespace utils
