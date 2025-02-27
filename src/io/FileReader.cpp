#include "FileReader.hpp"
#include "../exceptions/exceptions.hpp"

FileReader::FileReader(const std::string& filename)
    : FileIO(filename, std::ios::in) {
}

void FileReader::read_all(std::string& content) {
    // Get file size
    m_file.seekg(0, std::ios::end);
    std::streamsize size = m_file.tellg();
    m_file.seekg(0, std::ios::beg);

    if (size <= 0) {
        return;
    }

    // Reserve space
    content.resize(static_cast<size_t>(size));

    // Read directly into string's buffer
    m_file.read(&content[0], size);
    
    if (has_error()) {
        throw FileReaderException("Error reading file: " + get_error_message());
    }
}