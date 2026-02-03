#ifndef INCLUDE_CONFIGPARSER_H_
#define INCLUDE_CONFIGPARSER_H_

#include <map>
#include <string>

/**
 * @brief Simple configuration parser.
 *
 * Loads key-value pairs from a file and provides typed accessors.
 */
class ConfigParser {
   public:
    /**
     * @brief Loads configuration from a file.
     *
     * @param filename The path to the configuration file.
     */
    void load(const std::string& filename);

    /**
     * @brief Gets a double value from the configuration.
     *
     * @param key The key to look up.
     * @param defaultValue The value to return if the key is not found.
     * @return double The value associated with the key, or defaultValue.
     */
    double getDouble(const std::string& key, double defaultValue) const;

    /**
     * @brief Gets an integer value from the configuration.
     *
     * @param key The key to look up.
     * @param defaultValue The value to return if the key is not found.
     * @return int The value associated with the key, or defaultValue.
     */
    int getInt(const std::string& key, int defaultValue) const;

    /**
     * @brief Gets a string value from the configuration.
     *
     * @param key The key to look up.
     * @param defaultValue The value to return if the key is not found.
     * @return std::string The value associated with the key, or defaultValue.
     */
    std::string getString(const std::string& key, const std::string& defaultValue) const;

   private:
    std::map<std::string, std::string> data; ///< Map storing the configuration key-value pairs.
};

#endif  // INCLUDE_CONFIGPARSER_H_
