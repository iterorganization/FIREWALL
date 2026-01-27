#include "ConfigParser.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

// Helper to trim whitespace
static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

void ConfigParser::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open config file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        // Remove comments
        size_t commentPos = line.find('#');
        if (commentPos != std::string::npos) {
            line = line.substr(0, commentPos);
        }

        line = trim(line);
        if (line.empty()) continue;

        size_t delimiterPos = line.find(':');  // Use ':' for YAML-like look
        if (delimiterPos == std::string::npos) {
            delimiterPos = line.find('=');  // Also support '='
        }

        if (delimiterPos != std::string::npos) {
            std::string key = trim(line.substr(0, delimiterPos));
            std::string value = trim(line.substr(delimiterPos + 1));
            data[key] = value;
        }
    }
}

double ConfigParser::getDouble(const std::string& key, double defaultValue) const {
    auto it = data.find(key);
    if (it != data.end()) {
        try {
            return std::stod(it->second);
        } catch (...) {
            std::cerr << "Warning: Failed to parse double for key '" << key << "', using default.\n";
        }
    }
    return defaultValue;
}

int ConfigParser::getInt(const std::string& key, int defaultValue) const {
    auto it = data.find(key);
    if (it != data.end()) {
        try {
            return std::stoi(it->second);
        } catch (...) {
            std::cerr << "Warning: Failed to parse int for key '" << key << "', using default.\n";
        }
    }
    return defaultValue;
}

std::string ConfigParser::getString(const std::string& key, const std::string& defaultValue) const {
    auto it = data.find(key);
    if (it != data.end()) {
        return it->second;
    }
    return defaultValue;
}
