#ifndef ARG_PARSER_H
#define ARG_PARSER_H

#include <string>
#include <vector>

/**
 * @brief Argument parser for the benchmark executable.
 *
 * Handles parsing of command-line arguments for the benchmark configuration.
 */
class BenchArgParser {
public:
    /**
     * @brief Structure containing the parsed arguments.
     */
    struct Args {
        std::string configPath; ///< Path to the configuration file.
        std::string partPath;   ///< Path to the particles data file.
        std::string interpPath; ///< Path to the interpolation data file.
        std::string outPath;    ///< Path to the output file.
        bool help = false;      ///< Flag indicating if help message was requested.
    };

    /**
     * @brief Parses command line arguments.
     *
     * @param argc The number of arguments.
     * @param argv The array of arguments.
     * @return Args The struct containing parsed argument values.
     * @throws std::runtime_error if arguments are invalid.
     */
    static Args parse(int argc, char* argv[]);

    /**
     * @brief Prints the usage message to the standard output.
     *
     * @param progName The name of the program.
     */
    static void printUsage(const char* progName);
};

/**
 * @brief Argument parser for the main FIREWALL executable.
 *
 * Extends BenchArgParser to include wall-specific arguments.
 */
class ArgParser : public BenchArgParser {
public:
    /**
     * @brief Structure containing the parsed arguments for the main executable.
     */
    struct Args : public BenchArgParser::Args {
        std::string wallPath;     ///< Path to the wall geometry file.
        std::vector<int> wallIds; ///< List of wall IDs to process.
    };

    /**
     * @brief Parses command line arguments.
     *
     * @param argc The number of arguments.
     * @param argv The array of arguments.
     * @return Args The struct containing parsed argument values.
     * @throws std::runtime_error if arguments are invalid.
     */
    static Args parse(int argc, char* argv[]);

    /**
     * @brief Prints the usage message to the standard output.
     *
     * @param progName The name of the program.
     */
    static void printUsage(const char* progName);
};

#endif // ARG_PARSER_H
