#include "FileWriter.hpp"
#include "../exceptions/exceptions.hpp"

FileWriter::FileWriter(const std::string& filename, bool append)
    : FileIO(filename, std::ios::out | (append ? std::ios::app : std::ios::trunc)) {
}

bool FileWriter::write(const std::string& content) {
    m_file << content;
    return !has_error();
}

bool FileWriter::write_line(const std::string& line) {
    m_file << line << std::endl;
    return !has_error();
}