#pragma once

#include "FileIO.hpp"
#include <string>

/**
 * @brief Class for writing to a file
 */
class FileWriter : public FileIO {
public:
    /**
     * @brief Constructor
     *
     * @param filename The name of the file to write to
     * @param append Whether to append to the file (true) or overwrite (false)
     */
    FileWriter(const std::string& filename, bool append = false);

    /**
     * @brief Write a string to the file
     * 
     * @param content The string to write
     * @throws FileWriterException if an error occurs
     */
    void write(const std::string& content);
    
    /**
     * @brief Write a line to the file (adds newline)
     * 
     * @param line The line to write
     * @throws FileWriterException if an error occurs
     */
    void write_line(const std::string& line);
};