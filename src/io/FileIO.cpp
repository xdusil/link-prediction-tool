#include "FileIO.hpp"
#include "../exceptions/exceptions.hpp"

FileIO::FileIO(const std::string& filename, std::ios_base::openmode mode) 
    : m_file(filename, mode) {
    if (!m_file.is_open()) {
        throw FileException("Could not open file: " + filename);
    }
}