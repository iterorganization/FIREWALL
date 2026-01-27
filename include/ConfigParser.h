#ifndef INCLUDE_CONFIGPARSER_H_
#define INCLUDE_CONFIGPARSER_H_

#include <map>
#include <string>

class ConfigParser {
   public:
    // Load configuration from a file
    void load(const std::string& filename);

    // Get a value as a specific type
    double getDouble(const std::string& key, double defaultValue) const;
    int getInt(const std::string& key, int defaultValue) const;
    std::string getString(const std::string& key, const std::string& defaultValue) const;

   private:
    std::map<std::string, std::string> data;
};

#endif  // INCLUDE_CONFIGPARSER_H_
