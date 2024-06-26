#pragma once

#include <boost/filesystem.hpp>
#include <iostream>

#include "yaml-cpp/yaml.h"
namespace fs = boost::filesystem;

/**
 * @defgroup benchmark_yaml
 * @brief YAML configuration file interface.
 * @{
 */
/**
 * @brief Data structure to configure one simulation process.
 */
class ProcessConfig {
   public:
    /**
     * @brief Construct ProcessConfig.
     * @param __cmd Command of simulation process.
     * @param __args Arguments of simulation process.
     * @param __log Path of logging name.
     * @param __to_stdout True means redirect output of this process to standard output.
     * @param __clock_rate the rate of inter-simulator cycle convert.
     * @param __pre_copy Files copy to sub-directory of simulator before executing.
     */
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
    /**
     * @brief Command of simulation process.
     */
    std::string m_command;
    /**
     * @brief Arguments of simulation process.
     */
    std::vector<std::string> m_args;
    /**
     * @brief Path of logging name.
     */
    std::string m_log_file;
    /**
     * @brief True means redirect output of this process to standard output.
     */
    bool m_to_stdout;
    /**
     * @brief the rate of inter-simulator cycle convert.
     */
    double m_clock_rate;
    /**
     * @brief Files copy to sub-directory of simulator before executing.
     */
    std::string m_pre_copy;
};

/**
 * @brief Benchmark configuration structure.
 */
class BenchmarkConfig {
   public:
    /**
     * @brief Parse YAML configuration file to get benchmark configuration.
     * @param file_name Path of YAML configuration file.
     */
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

    /**
     * @brief Parse YAML configuration tree.
     * @param config Top node of YAML Tree.
     */
    void yaml_parse(const YAML::Node& config) {
        m_phase1_proc_cfg_list = yaml_parse_phase(config["phase1"]);
        m_phase2_proc_cfg_list = yaml_parse_phase(config["phase2"]);
    }

   private:
    /**
     * @brief Parse YAML configuration tree from "phase1" or "phase2".
     * @param config "phase1" or "phase2" node of YAML.
     */
    std::vector<ProcessConfig> yaml_parse_phase(const YAML::Node& config) {
        std::vector<ProcessConfig> proc_list;
        for (YAML::const_iterator it = config.begin(); it != config.end(); it++) {
            proc_list.push_back(yaml_parse_process(*it));
        }
        return proc_list;
    }

    /**
     * @brief Parse YAML configuration tree below "phase1" or "phase2".
     * @param config node below "phase1" or "phase2" of YAML.
     */
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

    /**
     * @brief Extend Environment Variables.
     *
     * Replace $SIMULATOR_ROOT and $BENCHMARK_ROOT with absoluate address.
     */
    void extend_env_var() {
        for (ProcessConfig& config : m_phase1_proc_cfg_list) {
            extend_env_var_proc(config);
        }
        for (ProcessConfig& config : m_phase2_proc_cfg_list) {
            extend_env_var_proc(config);
        }
    }

    /**
     * @brief Extend Environment Variables in one process configuration.
     *
     * Replace $SIMULATOR_ROOT and $BENCHMARK_ROOT with absoluate address.
     */
    void extend_env_var_proc(ProcessConfig& proc_config) {
        extend_env_var_string(proc_config.m_command);
        extend_env_var_string(proc_config.m_log_file);
        for (std::string& arg : proc_config.m_args) {
            extend_env_var_string(arg);
        }
    }

    /**
     * @brief Extend Environment Variables in one string.
     *
     * Replace $SIMULATOR_ROOT and $BENCHMARK_ROOT with absoluate address.
     */
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
    /**
     * @brief Environments.
     */
    std::string m_benchmark_root;
    std::string m_simulator_root;
    /**
     * @brief List of configuration structures of phase 1.
     */
    std::vector<ProcessConfig> m_phase1_proc_cfg_list;
    /**
     * @brief List of configuration structures of phase 2.
     */
    std::vector<ProcessConfig> m_phase2_proc_cfg_list;
};
/**
 * @}
 */
