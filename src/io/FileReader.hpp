#pragma once

#include <fstream>
#include <string>
#include <stdexcept>
#include <iostream>
#include <sstream>

/**
 * @brief Class for reading from a file
 */
class FileReader {
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
     * @brief Close the file
     */
    inline void close() {
        m_file.close();
    }

    /**
     * @brief Read the entire file
     *
     * @param content The content of the file
     * @return True if the file was read successfully, false otherwise
     */
    inline bool read_all(std::string& content) {
        // Get file size
        m_file.seekg(0, std::ios::end);
        std::streamsize size = m_file.tellg();
        m_file.seekg(0, std::ios::beg);

        // Reserve space
        content.resize(static_cast<size_t>(size));

        // Read directly into string's buffer
        return m_file.read(&content[0], size) ? true : false;
    }

    /**
     * @brief Check if there was any error in the file stream.
     *
     * @return True if an error is present, false otherwise.
     */
    inline bool has_error() const {
        return m_file.fail() || m_file.bad();
    }

        /**
     * @brief Get a brief description of the current error state.
     *
     * @return A string message describing the error.
     */
    inline std::string get_error_message() const {
        if (m_file.bad()) {
            return "Critical I/O error in stream.";
        } else if (m_file.fail()) {
            return "Non-critical failure in stream.";
        }
        return "No error.";
    }

private:
    std::ifstream m_file; // The file stream
};
