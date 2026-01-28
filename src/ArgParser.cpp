#include "ArgParser.h"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <sstream>

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
                std::stringstream ss(val);
                std::string segment;
                while (std::getline(ss, segment, ',')) {
                    if (!segment.empty()) {
                        try {
                            args.wallIds.push_back(std::stoi(segment));
                        } catch (...) {
                            throw std::runtime_error("Error: Invalid wall ID in list: " + segment);
                        }
                    }
                }
            } else {
                throw std::runtime_error("Error: --walls requires a comma-separated list of IDs.");
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
