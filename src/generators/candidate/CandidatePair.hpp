#pragma once

#include "Types.hpp"

namespace candidate {

/**
 * Directed source-destination pair selected for classifier evaluation.
 */
struct CandidatePair {
    IPAddress src;
    IPAddress dst;
};

} // namespace candidate
