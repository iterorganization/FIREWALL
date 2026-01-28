#ifndef ARG_PARSER_H
#define ARG_PARSER_H

#include <string>

class ArgParser {
public:
    struct Args {
        std::string configPath;
        std::string wallPath;
        std::string partPath;
        std::string interpPath;
        std::string outPath;
        bool help = false;
    };

    /**
     * Parses command line arguments.
     * Throws std::runtime_error if arguments are invalid.
     */
    static Args parse(int argc, char* argv[]);

    static void printUsage(const char* progName);
};

#endif // ARG_PARSER_H
