
# File cmdline\_options.h

[**File List**](files.md) **>** [**includes**](dir_943fa6db2bfb09b7dcf1f02346dde40e.md) **>** [**cmdline\_options.h**](cmdline__options_8h.md)

[Go to the documentation of this file.](cmdline__options_8h.md) 

```C++

#pragma once

#include <iostream>

#include "CLI/CLI.hpp"

class CmdLineOptions {
   public:
    CmdLineOptions()
        : m_bench(), m_cwd(), m_timeout_threshold(5), m_err_rate_threshold(0.005), m_debug(false) {}

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
    std::string m_bench;
    std::string m_cwd;

    long m_timeout_threshold;
    double m_err_rate_threshold;

    bool m_debug;
};

```