#include "FileWriter.hpp"
#include "../exceptions/exceptions.hpp"

FileWriter::FileWriter(const std::string& filename, bool append)
    : FileIO(filename, std::ios::out | (append ? std::ios::app : std::ios::trunc)) {
}

void FileWriter::write(const std::string& content) {
    m_file << content;
    if (has_error()) {
        throw FileWriterException("Error writing to file: " + get_error_message());
    }
}

void FileWriter::write_line(const std::string& line) {
    m_file << line << std::endl;
    if (has_error()) {
        throw FileWriterException("Error writing to file: " + get_error_message());
    }
}