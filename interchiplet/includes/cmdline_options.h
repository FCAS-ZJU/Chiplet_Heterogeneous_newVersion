#pragma once

#include <iostream>

#include "CLI/CLI.hpp"

/**
 * @defgroup cmdline
 * @brief Command line parser.
 * @{
 */
/**
 * @brief Options from command line.
 */
class CmdLineOptions {
   public:
    /**
     * @brief Constructor.
     */
    CmdLineOptions()
        : m_bench(), m_cwd(), m_timeout_threshold(5), m_err_rate_threshold(0.005), m_debug(false) {}

    /**
     * @brief Read options from command line.
     * @param argc Number of argument.
     * @param argv String of argument.
     */
    int parse(int argc, const char* argv[]) {
        CLI::App app{"Lego Chiplet Simulator"};
        app.add_option("bench", m_bench, "Benchmark configuration file (.yml)")
            ->required()
            ->check(CLI::ExistingFile);
        app.add_option("-t,--timeout", m_timeout_threshold, "Time out threshold, in time of round.")
            ->check(CLI::PositiveNumber);
        app.add_option("-e,--error", m_err_rate_threshold, "Error rate when quit simulation.");
        app.add_option("--cwd", m_cwd, "Woring directory for simulation.")
            ->check(CLI::ExistingPath);
        app.add_flag("--debug", m_debug, "Print debug information.");

        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError& e) {
            int ret = app.exit(e);
            exit(ret);
        }

        return 0;
    }

   public:
    /**
     * @brief Path of benchmark configuration yaml.
     */
    std::string m_bench;
    /**
     * @brief New working directory.
     */
    std::string m_cwd;

    /**
     * @brief Timeout threshold, in term of round.
     */
    long m_timeout_threshold;
    /**
     * @brief Error rate threshold, used to quit iteration.
     */
    double m_err_rate_threshold;

    /**
     * @brief Print debug information.
     */
    bool m_debug;
};
/**
 * @}
 */
