#include "FileReader.hpp"

// Constructor
FileReader::FileReader(const std::string& filename)
    : m_file(filename) {
    if (!m_file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
}