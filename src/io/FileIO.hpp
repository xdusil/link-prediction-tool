#pragma once

#include <fstream>
#include <string>
#include <stdexcept>
#include <iostream>
#include <sstream>

/**
 * @brief Base class for common file I/O operations
 */
class FileIO {
public:
    /**
     * @brief Virtual destructor
     */
    virtual ~FileIO() {
        close();
    }

    /**
     * @brief Close the file
     */
    inline void close() {
        if (m_file.is_open()) {
            m_file.close();
        }
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

protected:
    /**
     * @brief Constructor (protected, to be used by derived classes)
     *
     * @param filename The name of the file to work with
     * @param mode The file open mode
     */
    FileIO(const std::string& filename, std::ios_base::openmode mode);

    std::fstream m_file; // The file stream, protected so derived classes can access it
};