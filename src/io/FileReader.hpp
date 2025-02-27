#pragma once

#include "FileIO.hpp"
#include <string>

/**
 * @brief Class for reading from a file
 */
class FileReader : public FileIO {
public:
    /**
     * @brief Constructor
     *
     * @param filename The name of the file to read from
     */
    FileReader(const std::string& filename);

    /**
     * @brief Get the next line from the file
     *
     * @param line The line read from the file
     * @return True if a line was read, false if the end of the file was reached
     * or an error occurred
     */
    inline bool get_next_line(std::string& line) {
        return static_cast<bool>(std::getline(m_file, line));
    }

    /**
     * @brief Read the entire file
     *
     * @param content The content of the file
     * @throws FileReaderException if an error occurs
     */
    void read_all(std::string& content);
};