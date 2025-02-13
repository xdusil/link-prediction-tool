#include "FileReader.hpp"
#include "../exceptions/exceptions.hpp"

// Constructor
FileReader::FileReader(const std::string& filename)
    : m_file(filename) {
    if (!m_file.is_open()) {
        throw FileReaderException("Could not open file: " + filename);
    }
}