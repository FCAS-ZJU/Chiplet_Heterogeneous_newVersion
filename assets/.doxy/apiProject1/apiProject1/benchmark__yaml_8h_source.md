
# File benchmark\_yaml.h

[**File List**](files.md) **>** [**includes**](dir_943fa6db2bfb09b7dcf1f02346dde40e.md) **>** [**benchmark\_yaml.h**](benchmark__yaml_8h.md)

[Go to the documentation of this file.](benchmark__yaml_8h.md) 

```C++

#pragma once

#include <boost/filesystem.hpp>
#include <iostream>

#include "yaml-cpp/yaml.h"
namespace fs = boost::filesystem;

class ProcessConfig {
   public:
    ProcessConfig(const std::string& __cmd, const std::vector<std::string>& __args,
                  const std::string& __log, bool __to_stdout, double __clock_rate,
                  const std::string& __pre_copy)
        : m_command(__cmd),
          m_args(__args),
          m_log_file(__log),
          m_to_stdout(__to_stdout),
          m_clock_rate(__clock_rate),
          m_pre_copy(__pre_copy) {}

   public:
    std::string m_command;
    std::vector<std::string> m_args;
    std::string m_log_file;
    bool m_to_stdout;
    double m_clock_rate;
    std::string m_pre_copy;
};

class BenchmarkConfig {
   public:
    BenchmarkConfig(const std::string& file_name) {
        // Get environment variables.
        m_benchmark_root = fs::canonical(fs::path(file_name)).parent_path().string();
        if (getenv("SIMULATOR_ROOT") == NULL) {
            std::cerr << "The environment variable SIMULATOR_ROOT is not defined.\n";
            exit(EXIT_FAILURE);
        } else {
            m_simulator_root = getenv("SIMULATOR_ROOT");
        }

        // Parse YAML file.
        YAML::Node config;
        try {
            config = YAML::LoadFile(file_name);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            exit(EXIT_FAILURE);
        }

        // Parse YAML Tree.
        yaml_parse(config);

        // Extend environment variables.
        extend_env_var();
    }

    void yaml_parse(const YAML::Node& config) {
        m_phase1_proc_cfg_list = yaml_parse_phase(config["phase1"]);
        m_phase2_proc_cfg_list = yaml_parse_phase(config["phase2"]);
    }

   private:
    std::vector<ProcessConfig> yaml_parse_phase(const YAML::Node& config) {
        std::vector<ProcessConfig> proc_list;
        for (YAML::const_iterator it = config.begin(); it != config.end(); it++) {
            proc_list.push_back(yaml_parse_process(*it));
        }
        return proc_list;
    }

    ProcessConfig yaml_parse_process(const YAML::Node& config) {
        std::string pre_copy;
        if (config["pre_copy"]) {
            pre_copy = config["pre_copy"].as<std::string>();
        }
        return ProcessConfig(config["cmd"].as<std::string>(),
                             config["args"].as<std::vector<std::string> >(),
                             config["log"].as<std::string>(), config["is_to_stdout"].as<bool>(),
                             config["clock_rate"].as<double>(), pre_copy);
    }

    void extend_env_var() {
        for (ProcessConfig& config : m_phase1_proc_cfg_list) {
            extend_env_var_proc(config);
        }
        for (ProcessConfig& config : m_phase2_proc_cfg_list) {
            extend_env_var_proc(config);
        }
    }

    void extend_env_var_proc(ProcessConfig& proc_config) {
        extend_env_var_string(proc_config.m_command);
        extend_env_var_string(proc_config.m_log_file);
        for (std::string& arg : proc_config.m_args) {
            extend_env_var_string(arg);
        }
    }

    void extend_env_var_string(std::string& __str) {
        std::size_t find_pos;
        while ((find_pos = __str.find("$SIMULATOR_ROOT")) != std::string::npos) {
            __str = __str.replace(find_pos, 15, m_simulator_root);
        }
        while ((find_pos = __str.find("$BENCHMARK_ROOT")) != std::string::npos) {
            __str = __str.replace(find_pos, 15, m_benchmark_root);
        }
    }

   public:
    std::string m_benchmark_root;
    std::string m_simulator_root;
    std::vector<ProcessConfig> m_phase1_proc_cfg_list;
    std::vector<ProcessConfig> m_phase2_proc_cfg_list;
};

```