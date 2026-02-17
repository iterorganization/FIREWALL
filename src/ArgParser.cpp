#include "ArgParser.h"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <fstream>

// Default paths
constexpr char DEFAULT_CONFIG[] = "../examples/config.txt";
constexpr char DEFAULT_WALL[] = "./data/wall.h5";
constexpr char DEFAULT_PART[] = "./data/particles.h5";
constexpr char DEFAULT_INTERP[] = "./data/interpolation.h5";
constexpr char DEFAULT_OUT[] = "results.h5";

// --- BenchArgParser ---

void BenchArgParser::printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n"
              << "Options:\n"
              << "  --config <path>    Path to configuration file (default: " << DEFAULT_CONFIG << ")\n"
              << "  --part <path>      Path to particles HDF5 file (default: " << DEFAULT_PART << ")\n"
              << "  --interp <path>    Path to interpolation data HDF5 file (default: " << DEFAULT_INTERP << ")\n"
              << "  --out <path>       Path to output HDF5 file (default: " << DEFAULT_OUT << ")\n"
              << "  --help, -h         Show this help message\n";
}

BenchArgParser::Args BenchArgParser::parse(int argc, char* argv[]) {
    Args args;
    // Defaults
    args.configPath = DEFAULT_CONFIG;
    args.partPath = DEFAULT_PART;
    args.interpPath = DEFAULT_INTERP;
    args.outPath = DEFAULT_OUT;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config") {
            if (i + 1 < argc) {
                args.configPath = argv[++i];
            } else {
                throw std::runtime_error("Error: --config requires a path argument.");
            }
        } else if (arg == "--part") {
            if (i + 1 < argc) {
                args.partPath = argv[++i];
            } else {
                throw std::runtime_error("Error: --part requires a path argument.");
            }
        } else if (arg == "--interp") {
            if (i + 1 < argc) {
                args.interpPath = argv[++i];
            } else {
                throw std::runtime_error("Error: --interp requires a path argument.");
            }
        } else if (arg == "--out") {
            if (i + 1 < argc) {
                args.outPath = argv[++i];
            } else {
                throw std::runtime_error("Error: --out requires a path argument.");
            }
        } else if (arg == "--help" || arg == "-h") {
            args.help = true;
            return args;
        } else {
             throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return args;
}

// --- ArgParser ---

void ArgParser::printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n"
              << "Options:\n"
              << "  --config <path>    Path to configuration file (default: " << DEFAULT_CONFIG << ")\n"
              << "  --wall <path>      Path to wall HDF5 file (default: " << DEFAULT_WALL << ")\n"
              << "  --part <path>      Path to particles HDF5 file (default: " << DEFAULT_PART << ")\n"
              << "  --interp <path>    Path to interpolation data HDF5 file (default: " << DEFAULT_INTERP << ")\n"
              << "  --out <path>       Path to output HDF5 file (default: " << DEFAULT_OUT << ")\n"
              << "  --walls <list>     Comma-separated list of wall IDs to process (default: all)\n"
              << "  --full_profile_walls <list>     Comma-separated list of wall IDs to store full profiles for (default: none)\n"
              << "  --help, -h         Show this help message\n";
}

ArgParser::Args ArgParser::parse(int argc, char* argv[]) {
    Args args;
    // Defaults
    args.configPath = DEFAULT_CONFIG;
    args.wallPath = DEFAULT_WALL;
    args.partPath = DEFAULT_PART;
    args.interpPath = DEFAULT_INTERP;
    args.outPath = DEFAULT_OUT;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config") {
            if (i + 1 < argc) {
                args.configPath = argv[++i];
            } else {
                throw std::runtime_error("Error: --config requires a path argument.");
            }
        } else if (arg == "--wall") {
            if (i + 1 < argc) {
                args.wallPath = argv[++i];
            } else {
                throw std::runtime_error("Error: --wall requires a path argument.");
            }
        } else if (arg == "--part") {
            if (i + 1 < argc) {
                args.partPath = argv[++i];
            } else {
                throw std::runtime_error("Error: --part requires a path argument.");
            }
        } else if (arg == "--interp") {
            if (i + 1 < argc) {
                args.interpPath = argv[++i];
            } else {
                throw std::runtime_error("Error: --interp requires a path argument.");
            }
        } else if (arg == "--out") {
            if (i + 1 < argc) {
                args.outPath = argv[++i];
            } else {
                throw std::runtime_error("Error: --out requires a path argument.");
            }
        } else if (arg == "--walls") {
            if (i + 1 < argc) {
                std::string val = argv[++i];
                std::string content;

                // Check if the argument is a file path
                std::ifstream file(val);
                if (file.is_open()) {
                    // Read entire file content into a string, 
                    // replacing newlines/spaces with commas for uniform parsing
                    std::string line;
                    while (std::getline(file, line)) {
                        content += line + ",";
                    }
                    file.close();
                } else {
                    // Not a file, assume it's a raw comma-separated string
                    content = val;
                }

                // Parse the resulting string (comma or space separated)
                std::stringstream ss(content);
                std::string segment;
                // Use a delimiter set that handles commas, spaces, or tabs
                while (std::getline(ss, segment, ',')) {
                    // Trim whitespace and handle empty segments
                    segment.erase(0, segment.find_first_not_of(" \t\r\n"));
                    segment.erase(segment.find_last_not_of(" \t\r\n") + 1);
                    
                    if (!segment.empty()) {
                        try {
                            args.wallIds.push_back(std::stoi(segment));
                        } catch (...) {
                            throw std::runtime_error("Error: Invalid wall ID: '" + segment + "'");
                        }
                    }
                }
            } else {
                throw std::runtime_error("Error: --walls requires a list or a file path.");
            }
        } else if (arg == "--full_profile_walls"){
            if (i + 1 < argc) {
                std::string val = argv[++i];
                std::string content;

                // Check if the argument is a file path
                std::ifstream file(val);
                if (file.is_open()) {
                    // Read entire file content into a string,
                    // replacing newlines/spaces with commas for uniform parsing
                    std::string line;
                    while (std::getline(file, line)) {
                        content += line + ",";
                    }
                    file.close();
                } else {
                    // Not a file, assume it's a raw comma-separated string
                    content = val;
                }

                // Parse the resulting string (comma or space separated)
                std::stringstream ss(content);
                std::string segment;
                // Use a delimiter set that handles commas, spaces, or tabs
                while (std::getline(ss, segment, ',')) {
                    // Trim whitespace and handle empty segments
                    segment.erase(0, segment.find_first_not_of(" \t\r\n"));
                    segment.erase(segment.find_last_not_of(" \t\r\n") + 1);

                    if (!segment.empty()) {
                        try {
                            args.fullProfileWallIds.push_back(std::stoi(segment));
                        } catch (...) {
                            throw std::runtime_error("Error: Invalid wall ID: '" + segment + "'");
                        }
                    }
                }
            } else {
                throw std::runtime_error("Error: --walls requires a list or a file path.");
            }
        } else if (arg == "--help" || arg == "-h") {
            args.help = true;
            return args;
        } else {
             throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return args;
}
