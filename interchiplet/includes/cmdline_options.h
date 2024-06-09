#pragma once

#include <boost/program_options.hpp>
#include <iostream>

namespace po = boost::program_options;

namespace InterChiplet {
/**
 * @brief Options from command line.
 */
class CmdLineOptions {
   public:
    /**
     * @brief Read options from command line.
     * @param argc Number of argument.
     * @param argv String of argument.
     */
    CmdLineOptions(int argc, const char* argv[])
        : m_has_cwd(false), m_cwd(), m_timeout_threshold(5), m_err_rate_threshold(0.005) {
        try {
            // Setup command line.
            po::options_description desc{"Program Usage"};
            desc.add_options()("help,h", "Help screen")("bench,b", po::value<std::string>(),
                                                        "Path to benchmark.")(
                "timeout,t", po::value<int>()->default_value(m_timeout_threshold),
                "Time out threshold, in time of round.")(
                "error,e", po::value<double>()->default_value(m_err_rate_threshold),
                "Error rate when quit simulation.")("cwd",
                                                    po::value<std::string>()->default_value(""),
                                                    "Woring directory for simulation.");

            // Positional options.
            po::positional_options_description pos_desc;
            pos_desc.add("bench", -1);

            // Setup and execute parser.
            po::command_line_parser parser{argc, argv};
            parser.options(desc).positional(pos_desc).allow_unregistered();
            po::parsed_options parsed_options = parser.run();

            // Build up variable map.
            po::variables_map vm;
            po::store(parsed_options, vm);
            po::notify(vm);

            // Handle options.
            if (vm.count("help")) {
                std::cout << desc << '\n';
                exit(EXIT_SUCCESS);
            }
            // mandatory option.
            if (vm.count("bench")) {
                m_bench = vm["bench"].as<std::string>();
            } else {
                std::cout << desc << '\n';
                exit(EXIT_SUCCESS);
            }

            // optional option.
            if (vm.count("timeout")) {
                m_timeout_threshold = vm["timeout"].as<int>();
            }
            if (vm.count("error")) {
                m_err_rate_threshold = vm["error"].as<double>();
            }
            if (vm.count("cwd")) {
                m_cwd = vm["cwd"].as<std::string>();
                m_has_cwd = !m_cwd.empty();
            }
        } catch (std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            exit(EXIT_FAILURE);
        } catch (...) {
            std::cerr << "Unknown error!" << "\n";
            exit(EXIT_FAILURE);
        }
    }

   public:
    /**
     * @brief Path of benchmark configuration yaml.
     */
    std::string m_bench;

    /**
     * @brief Timeout threshold, in term of round.
     */
    int m_timeout_threshold;
    /**
     * @brief Error rate threshold, used to quit iteration.
     */
    double m_err_rate_threshold;

    /**
     * @brief Flag to change working directory.
     */
    bool m_has_cwd;
    /**
     * @brief New working directory if m_has_cwd is True.
     */
    std::string m_cwd;
};
}  // namespace InterChiplet
