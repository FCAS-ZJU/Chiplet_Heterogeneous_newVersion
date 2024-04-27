
#include "sync_protocol.h"
#include "net_bench.h"
#include "net_delay.h"
#include "cmdline_options.h"
#include "benchmark_yaml.h"

#include <ctime>

#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>

/**
 * @brief Data structure of synchronize operation.
 */
class SyncStruct
{
public:
    /**
     * @brief Construct synchronize stucture.
     * 
     * Initializete mutex.
     */
    SyncStruct()
        : m_cycle(0)
    {
        if (pthread_mutex_init(&m_mutex, NULL) < 0)
        {
            perror("pthread_mutex_init");
            exit(EXIT_FAILURE);
        }
    }

    /**
     * @brief Destory synchronize structure.
     * 
     * Destory mutex.
     */
    ~SyncStruct()
    {
        pthread_mutex_destroy(&m_mutex);
    }

public:
    /**
     * @brief Mutex to access this structure.
     */
    pthread_mutex_t m_mutex;

    /**
     * @brief Global simulation cycle, which is the largest notified cycle count.
     */
    InterChiplet::TimeType m_cycle;

    /**
     * @brief Benchmark list, recording the communication transactions have sent out.
     */
    InterChiplet::NetworkBenchList m_bench_list;
    /**
     * @brief Delay list, recoding the delay of each communication transactions
     */
    InterChiplet::NetworkDelayList m_delay_list;

    /**
     * @brief List of PIPE file names.
     */
    std::vector<std::string> m_fifo_list;
    /**
     * @brief List of Read commands, used to pair with write commands.
     */
    std::vector<InterChiplet::SyncCommand> m_read_cmd_list;
    /**
     * @brief List of Write commands, used to pair with write commands.
     */
    std::vector<InterChiplet::SyncCommand> m_write_cmd_list;
};

/**
 * @brief Data structure of process configuration.
 */
class ProcessStruct
{
public:
    ProcessStruct(const InterChiplet::ProcessConfig& __config)
        : m_command(__config.m_command)
        , m_args(__config.m_args)
        , m_log_file(__config.m_log_file)
        , m_to_stdout(__config.m_to_stdout)
        , m_pre_copy(__config.m_pre_copy)
        , m_unfinished_line()
        , m_thread_id()
        , m_pid(-1)
        , m_pid2(-1)
        , m_sync_struct(NULL)
    {}

public:
    // Configuration.
    std::string m_command;
    std::vector<std::string> m_args;
    std::string m_log_file;
    bool m_to_stdout;
    std::string m_pre_copy;

    std::string m_unfinished_line;

    // Indentify
    int m_round;
    int m_phase;
    int m_thread;

    pthread_t m_thread_id;
    int m_pid;
    int m_pid2;

    /**
     * @brief Pointer to synchronize structure.
     */
    SyncStruct* m_sync_struct;
};

/**
 * @brief  Create FIFO with specified name.
 * @param __fifo_name Specified name for PIPE.
 * @retval 0 Operation success. PIPE file is existed or created.
 * @retval -1 Operation fail. PIPE file is missing.
 */
int create_fifo(std::string __fifo_name)
{
    if (access(__fifo_name.c_str(), F_OK) == -1)
    {
        // Report error if FIFO file does not exist and mkfifo error.
        if (mkfifo(__fifo_name.c_str(), 0664) == -1)
        {
            std::cerr << "Cannot create FIFO file " << __fifo_name << "." << std::endl;
            return -1;
        }
        // Report success.
        else
        {
            std::cout << "Create FIFO file " << __fifo_name << "." << std::endl;
            return 0;
        }
    }
    // Reuse exist FIFO and reports.
    else
    {
        std::cout << "Reuse exist FIFO file " << __fifo_name << "." << std::endl;
        return 0;
    }
}

#define PIPE_BUF_SIZE 1024

/**
 * @brief Handle PIPE command.
 * @param __cmd Command to handle.
 * @param __sync_struct Pointer to global synchronize structure.
 */
void handle_pipe_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct)
{
    std::cout << "[INTERCMD DBG] PIPE command"
        << " from " << __cmd.m_src_x << " " << __cmd.m_src_y
        << " to " << __cmd.m_dst_x << " " << __cmd.m_dst_y << ". ";

    // Create Pipe file and add the file to 
    std::string file_name = InterChiplet::SyncProtocol::pipeNameMaster(
        __cmd.m_src_x, __cmd.m_src_y, __cmd.m_dst_x, __cmd.m_dst_y);
    if (create_fifo(file_name.c_str()) == 0)
    {
        __sync_struct->m_fifo_list.push_back(file_name);
    }
    std::cout << "Create pipe file " << file_name << "." << std::endl;

    // Send synchronize command.
    std::stringstream ss;
    ss << "[INTERCMD] SYNC " << 0 << std::endl;
    write(__cmd.m_stdin_fd, ss.str().c_str(), ss.str().size());
}

/**
 * @brief Handle READ command.
 * @param __cmd Command to handle.
 * @param __sync_struct Pointer to global synchronize structure.
 */
void handle_read_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct)
{
    std::cout << "[INTERCMD DBG] READ command at " << __cmd.m_cycle << " cycle"
        << " from " << __cmd.m_src_x << " " << __cmd.m_src_y
        << " to " << __cmd.m_dst_x << " " << __cmd.m_dst_y << ". ";

    // Check for paired write command.
    bool has_write_cmd = false;
    InterChiplet::SyncCommand write_cmd;
    for (std::size_t i = 0; i < __sync_struct->m_write_cmd_list.size(); i ++)
    {
        InterChiplet::SyncCommand& __write_cmd = __sync_struct->m_write_cmd_list[i];
        if (__write_cmd.m_src_x == __cmd.m_src_x && __write_cmd.m_src_y == __cmd.m_src_y &&
            __write_cmd.m_dst_x == __cmd.m_dst_x && __write_cmd.m_dst_y == __cmd.m_dst_y &&
            __write_cmd.m_nbytes == __cmd.m_nbytes)
        {
            has_write_cmd = true;
            write_cmd = __write_cmd;
            __sync_struct->m_write_cmd_list.erase(__sync_struct->m_write_cmd_list.begin() + i);
            break;
        }
    }

    if (!has_write_cmd)
    {
        // If there is no paired write command, add this command to read command queue to wait.
        __sync_struct->m_read_cmd_list.push_back(__cmd);
        std::cout << "Register READ command to pair with WRITE command." << std::endl;
    }
    else
    {
        // If there is a paired write command, get the end cycle of transaction.
        InterChiplet::TimeType end_cycle =
            __sync_struct->m_delay_list.getEndCycle(write_cmd, __cmd);
        std::cout << "Pair with WRITE command and transation ends at "
            << end_cycle << " cycle." << std::endl;

        // Send synchronize command to response READ command.
        std::stringstream ss;
        ss << "[INTERCMD] SYNC " << end_cycle << std::endl;
        write(__cmd.m_stdin_fd, ss.str().c_str(), ss.str().size());

        // Send synchronize command to response WRITE command.
        ss.clear();
        ss << "[INTERCMD] SYNC " << end_cycle << std::endl;
        write(write_cmd.m_stdin_fd, ss.str().c_str(), ss.str().size());
    }
}

/**
 * @brief Handle WRITE command.
 * @param __cmd Command to handle.
 * @param __sync_struct Pointer to global synchronize structure.
 */
void handle_write_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct)
{
    std::cout << "[INTERCMD DBG] WRITE command at " << __cmd.m_cycle << " cycle"
        << " from " << __cmd.m_src_x << " " << __cmd.m_src_y
        << " to " << __cmd.m_dst_x << " " << __cmd.m_dst_y << ". ";

    // Insert write event to benchmark list.
    InterChiplet::NetworkBenchItem bench_item(__cmd);
    __sync_struct->m_bench_list.insert(bench_item);

    // Check for paired read command.
    bool has_read_cmd = false;
    InterChiplet::SyncCommand read_cmd;
    for (std::size_t i = 0; i < __sync_struct->m_read_cmd_list.size(); i ++)
    {
        InterChiplet::SyncCommand& __read_cmd = __sync_struct->m_read_cmd_list[i];
        if (__read_cmd.m_src_x == __cmd.m_src_x && __read_cmd.m_src_y == __cmd.m_src_y &&
            __read_cmd.m_dst_x == __cmd.m_dst_x && __read_cmd.m_dst_y == __cmd.m_dst_y &&
            __read_cmd.m_nbytes == __cmd.m_nbytes)
        {
            has_read_cmd = true;
            read_cmd = __read_cmd;
            __sync_struct->m_read_cmd_list.erase(__sync_struct->m_read_cmd_list.begin() + i);
            break;
        }
    }

    if (!has_read_cmd)
    {
        // If there is no paired read command, add this command to write command queue to wait.
        __sync_struct->m_write_cmd_list.push_back(__cmd);
        std::cout << "Register WRITE command to pair with READ command." << std::endl;
    }
    else
    {
        // If there is a paired read command, get the end cycle of transaction.
        InterChiplet::TimeType end_cycle =
            __sync_struct->m_delay_list.getEndCycle(__cmd, read_cmd);
        std::cout << "Pair with READ command and transation ends at "
            << end_cycle << " cycle." << std::endl;

        // Send synchronize command to response WRITE command.
        std::stringstream ss;
        ss << "[INTERCMD] SYNC " << end_cycle << std::endl;
        write(__cmd.m_stdin_fd, ss.str().c_str(), ss.str().size());

        // Send synchronize command to response READ command.
        ss.clear();
        ss << "[INTERCMD] SYNC " << end_cycle << std::endl;
        write(read_cmd.m_stdin_fd, ss.str().c_str(), ss.str().size());
    }
}

/**
 * @brief Handle CYCLE command.
 * @param __cmd Command to handle.
 * @param __sync_struct Pointer to global synchronize structure.
 */
void handle_cycle_cmd(const InterChiplet::SyncCommand& __cmd, SyncStruct* __sync_struct)
{
    std::cout << "[INTERCMD DBG] CYCLE command at " << __cmd.m_cycle << " cycle" << ".";

    // Update global cycle.
    InterChiplet::TimeType new_cycle = __cmd.m_cycle;
    if (__sync_struct->m_cycle < new_cycle)
    {
        __sync_struct->m_cycle = new_cycle;
    }
}

void parse_command(char* __pipe_buf, ProcessStruct* __proc_struct, int __stdin_fd)
{
    // Split line by '\n'
    std::string line = std::string(__pipe_buf);
    std::vector<std::string> lines;

    int start_idx = 0;
    for (std::size_t i = 0; i < line.size(); i ++)
    {
        if (line[i] =='\n')
        {
            std::string l = line.substr(start_idx, i + 1 - start_idx);
            start_idx = i + 1;
            lines.push_back(l);
        }
    }
    if (start_idx < line.size())
    {
        std::string l = line.substr(start_idx, line.size() - start_idx);
        lines.push_back(l);
    }

    // Unfinished line.
    if (__proc_struct->m_unfinished_line.size() > 0)
    {
        lines[0] = __proc_struct->m_unfinished_line + lines[0];
        __proc_struct->m_unfinished_line = "";
    }
    if (lines[lines.size() - 1].find('\n') == -1)
    {
        __proc_struct->m_unfinished_line = lines[lines.size() - 1];
        lines.pop_back();
    }

    // Get line start with [INTERCMD]
    for (std::size_t i = 0; i < lines.size(); i ++)
    {
        std::string l = lines[i];
        if (l.substr(0, 10) == "[INTERCMD]")
        {
            InterChiplet::SyncCommand cmd = InterChiplet::SyncProtocol::parseCmd(l);
            cmd.m_stdin_fd = __stdin_fd;

            pthread_mutex_lock(&__proc_struct->m_sync_struct->m_mutex);

            // Call functions to handle corresponding command.
            switch(cmd.m_type)
            {
            case InterChiplet::SC_CYCLE:
                handle_cycle_cmd(cmd, __proc_struct->m_sync_struct);
                break;
            case InterChiplet::SC_PIPE:
                handle_pipe_cmd(cmd, __proc_struct->m_sync_struct);
                break;
            case InterChiplet::SC_READ:
                handle_read_cmd(cmd, __proc_struct->m_sync_struct);
                break;
            case InterChiplet::SC_WRITE:
                handle_write_cmd(cmd, __proc_struct->m_sync_struct);
                break;
            default:
                break;
            }

            pthread_mutex_unlock(&__proc_struct->m_sync_struct->m_mutex);
        }
    }
}

void *bridge_thread(void * __args_ptr)
{
    ProcessStruct* proc_struct = (ProcessStruct*)__args_ptr;

    int pipe_stdin[2]; // Pipe to send data to child process
    int pipe_stdout[2]; // Pipe to receive data from child process
    int pipe_stderr[2]; // Pipe to receive data from child process

    // Create pipes
    if (pipe(pipe_stdin) == -1 || pipe(pipe_stdout) == -1 || pipe(pipe_stderr) == -1)
    {
        perror("pipe");
        exit(EXIT_FAILURE);
    }

    // Create sub directory for subprocess.
    char* sub_dir_path = new char[128];
    sprintf(sub_dir_path, "./proc_r%d_p%d_t%d",
        proc_struct->m_round, proc_struct->m_phase, proc_struct->m_thread);
    if (access(sub_dir_path, F_OK) == -1)
    {
        mkdir(sub_dir_path, 0775);
    }

    std::cout << "Enter bridge_thread " << proc_struct->m_command << std::endl;
    // Fork a child process
    pid_t pid = fork();
    if (pid == -1)
    {
        perror("fork");
        exit(EXIT_FAILURE);
    }

    if (pid == 0)
    { // Child process
        // Close unnecessary pipe ends
        close(pipe_stdin[1]);
        close(pipe_stdout[0]);
        close(pipe_stderr[0]);

        // Redirect stdin and stdout to pipes
        dup2(pipe_stdin[0], STDIN_FILENO);
        dup2(pipe_stdout[1], STDOUT_FILENO);
        dup2(pipe_stderr[1], STDERR_FILENO);

        // Change directory to sub-process.
        chdir(sub_dir_path);
        perror("chdir");
        // TODO: Copy necessary configuration file.
        if (!proc_struct->m_pre_copy.empty())
        {
            std::string cp_cmd = std::string("cp ") + proc_struct->m_pre_copy + " .";
            system(cp_cmd.c_str());
            perror("system");
        }

        std::cout << "CWD: " << get_current_dir_name() << std::endl;

        // Build arguments.
        int argc = proc_struct->m_args.size();
        char** args_list = new char*[argc + 2];
        args_list[0] = new char[proc_struct->m_command.size() + 1];
        strcpy(args_list[0], proc_struct->m_command.c_str());
        args_list[0][proc_struct->m_command.size()] = '\0';
        for (int i = 0; i < argc; i ++)
        {
            int arg_len = proc_struct->m_args[i].size();
            args_list[i + 1] = new char[arg_len + 1];
            strcpy(args_list[i + 1], proc_struct->m_args[i].c_str());
            args_list[i + 1][arg_len] = '\0';
        }
        args_list[argc + 1] = NULL;

        // Execute the child program
        std::cout << "Exec: ";
        for (int i = 0; i < proc_struct->m_args.size() + 1; i ++)
        {
            std::cout << " " << args_list[i];
        }
        std::cout << std::endl;
        execvp(args_list[0], args_list);

        // If execl fails, it means the child process couldn't be started
        perror("execvp");
        exit(EXIT_FAILURE);
    }
    else
    { // Parent process
        std::cout << "Start simulation process " << pid << std::endl;
        proc_struct->m_pid2 = pid;

        // Close unnecessary pipe ends
        close(pipe_stdin[0]);
        close(pipe_stdout[1]);
        close(pipe_stderr[1]);
        int stdin_fd = pipe_stdin[1];
        int stdout_fd = pipe_stdout[0];
        int stderr_fd = pipe_stderr[0];

        pollfd fd_list[2] = {{fd: stdout_fd, events: POLL_IN},
                             {fd: stderr_fd, events: POLL_IN}};

        // Move log to subfolder.
        std::ofstream log_file(std::string(sub_dir_path) + "/" + proc_struct->m_log_file);

        // Write execution start time to log file.
        std::time_t t = std::time(0);
        std::tm* now = std::localtime(&t);
        log_file << "Execution starts at "
            << (now->tm_year + 1900) << "-" << (now->tm_mon + 1) << "-" << now->tm_mday << "  "
            << (now->tm_hour) << ":" << (now->tm_min) << ":" << (now->tm_sec) << std::endl;

        char* pipe_buf = new char[PIPE_BUF_SIZE + 1];
        bool to_stdout = proc_struct->m_to_stdout;
        int res = 0;
        while (true)
        {
            int res = poll(fd_list, 2, 1000);
            if (res == -1)
            {
                perror("poll");
                break;
            }

            bool has_stdout = false;
            if (fd_list[0].revents & POLL_IN)
            {
                has_stdout = true;
                int res = read(stdout_fd, pipe_buf, PIPE_BUF_SIZE);
                if (res <= 0) break;
                pipe_buf[res] = '\0';
                // log redirection.
                log_file.write(pipe_buf, res).flush();
                if (to_stdout)
                {
                    std::cout.write(pipe_buf, res);
                    std::cout.flush();
                }
                // Parse command in pipe_buf
                parse_command(pipe_buf, proc_struct, stdin_fd);
            }
            if (fd_list[1].revents & POLL_IN)
            {
                has_stdout = true;
                int res = read(stderr_fd, pipe_buf, PIPE_BUF_SIZE);
                if (res <= 0) break;
                pipe_buf[res] = '\0';
                // log redirection.
                log_file.write(pipe_buf, res).flush();
                if (to_stdout)
                {
                    std::cerr.write(pipe_buf, res);
                    std::cerr.flush();
                }
                // Parse command in pipe_buf
                parse_command(pipe_buf, proc_struct, stdin_fd);
            }

            // Check the status of child process and quit.
            int status;
            if (!has_stdout && (waitpid(pid, &status, WNOHANG) > 0))
            {
                // Optionally handle child process termination status
                std::cout << "Simulation process " << proc_struct->m_pid2
                    << " terminate with status = " << status << "." << std::endl;
                break;
            }
        }

        delete pipe_buf;
    }

    return 0;
}

InterChiplet::TimeType __loop_phase_one(
    int __round,
    const std::vector<InterChiplet::ProcessConfig>& __proc_cfg_list)
{
    // Create synchronize data structure.
    SyncStruct* g_sync_structure = new SyncStruct();

    // Load delay record.
    g_sync_structure->m_delay_list.load_delay("delayInfo.txt");
    std::cout << "Load " << g_sync_structure->m_delay_list.size() << " delay records." << std::endl;

    // Create multi-thread.
    int thread_i = 0;
    std::vector<ProcessStruct*> proc_struct_list;
    for (auto& proc_cfg: __proc_cfg_list)
    {
        ProcessStruct* proc_struct = new ProcessStruct(proc_cfg);
        proc_struct->m_round = __round;
        proc_struct->m_phase = 1;
        proc_struct->m_thread = thread_i;
        proc_struct->m_sync_struct = g_sync_structure;
        int res = pthread_create(
            &(proc_struct->m_thread_id), NULL, bridge_thread, (void*)proc_struct);
        if (res < 0)
        {
            perror("pthread_create");
            exit(EXIT_FAILURE);
        }

        proc_struct_list.push_back(proc_struct);
        thread_i ++;
    }

    // Wait threads to finish.
    for (auto& proc_struct: proc_struct_list)
    {
        pthread_join(proc_struct->m_thread_id, NULL);
        delete proc_struct;
    }
    std::cout << "All process has exit." << std::endl;

    // Remove file.
    for (auto& __name: g_sync_structure->m_fifo_list)
    {
        remove(__name.c_str());
    }

    // Dump benchmark record.
    g_sync_structure->m_bench_list.dump_bench("bench.txt");
    std::cout << "Dump " << g_sync_structure->m_bench_list.size() << " bench records." << std::endl;

    // Destory global synchronize structure, and return total cycle.
    InterChiplet::TimeType res_cycle = g_sync_structure->m_cycle;
    delete g_sync_structure;
    return res_cycle;
}

void __loop_phase_two(
    int __round,
    const std::vector<InterChiplet::ProcessConfig>& __proc_cfg_list)
{
    // Create synchronize data structure.
    SyncStruct* g_sync_structure = new SyncStruct();

    // Create multi-thread.
    int thread_i = 0;
    std::vector<ProcessStruct*> proc_struct_list;
    for (auto& proc_cfg: __proc_cfg_list)
    {
        ProcessStruct* proc_struct = new ProcessStruct(proc_cfg);
        proc_struct->m_round = __round;
        proc_struct->m_phase = 2;
        proc_struct->m_thread = thread_i;
        thread_i ++;
        proc_struct->m_sync_struct = g_sync_structure;
        int res = pthread_create(
            &(proc_struct->m_thread_id), NULL, bridge_thread, (void*)proc_struct);
        if (res < 0)
        {
            perror("pthread_create");
            exit(EXIT_FAILURE);
        }

        proc_struct_list.push_back(proc_struct);
        thread_i ++;
    }

    // Wait threads to finish.
    for (auto& proc_struct: proc_struct_list)
    {
        pthread_join(proc_struct->m_thread_id, NULL);
        delete proc_struct;
    }
    std::cout << "All process has exit." << std::endl;

    // Destory global synchronize structure.
    delete g_sync_structure;
}

int main(int argc, const char* argv[])
{
    // Parse command line.
    InterChiplet::CmdLineOptions options(argc, argv);

    // Change working directory if --cwd is specified.
    if (options.m_has_cwd)
    {
        if (access(options.m_cwd.c_str(), F_OK) == 0)
        {
            chdir(options.m_cwd.c_str());
            std::cout << "[INTERCMD] Change working directory " << get_current_dir_name() << ".\n";
        }
    }

    // Check exist of benchmark configuration yaml.
    if (access(options.m_bench.c_str(), F_OK) < 0)
    {
        std::cerr << "Error: Cannot find benchmark " << options.m_bench << "." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Load benchmark configuration.
    InterChiplet::BenchmarkConfig configs(options.m_bench);
    std::cout << "[INTERCMD] Load benchmark configuration from " << options.m_bench << ".\n";

    // Get start time of simulation.
	struct timeval simstart, simend, roundstart, roundend;
	gettimeofday(&simstart, 0);

    long long sim_cycle = 0;
    for (int round = 1; round <= options.m_timeout_threshold; round ++)
    {
        // Get start time of one round.
	    gettimeofday(&roundstart, 0);
        std::cout << "[COMMBRIDGE] *** Round " << round << " Phase 1 ***" << std::endl;
        InterChiplet::TimeType round_cycle = __loop_phase_one(round, configs.m_phase1_proc_cfg_list);

        // Get simulation cycle.
        // If simulation cycle this round is close to the previous one, quit iteration.
        std::cout << "[COMMBRIDGE] Benchmark elapses " << round_cycle << " cycle." << std::endl;
        if (round > 1)
        {
            // Calculate error of simulation cycle.
            double err_rate = ((double)round_cycle - (double)sim_cycle) / (double)round_cycle;
            err_rate = err_rate < 0 ? - err_rate : err_rate;
            std::cout << "[COMMBRIDGE] Difference related to pervious round is "
                << err_rate * 100 << "%." << std::endl;
            // If difference is small enough, quit loop.
            if (err_rate < options.m_err_rate_threshold)
            {
                std::cout << "[COMMBRIDGE] Quit simulation because simulation cycle has converged."
                    << std::endl;
                sim_cycle = round_cycle;
                break;
            }
        }
        sim_cycle = round_cycle;

        std::cout << "[COMMBRIDGE] *** Round " << round << " Phase 2 ***" << std::endl;
        __loop_phase_two(round, configs.m_phase2_proc_cfg_list);

        // Get end time of one round.
        gettimeofday(&roundend, 0);
        unsigned long elaped_sec = roundend.tv_sec - roundstart.tv_sec;
        std::cout << "[COMMBRIDGE] Round " << round << " elapses "
            << elaped_sec / 3600 / 24 << "d "
            << (elaped_sec / 3600) % 24 << "h "
            << (elaped_sec / 60) % 60 << "m "
            << elaped_sec % 60 << "s." << std::endl;
    }

    // Get end time of simulation.
    std::cout << "[COMMBRIDGE] *** End of Simulation ***" << std::endl;
    gettimeofday(&simend, 0);
    unsigned long elaped_sec = simend.tv_sec - simstart.tv_sec;
    std::cout << "[COMMBRIDGE] Benchmark elapses " << sim_cycle << " cycle." << std::endl;
    std::cout << "[COMMBRIDGE] Simulation elapseds " 
        << elaped_sec / 3600 / 24 << "d "
        << (elaped_sec / 3600) % 24 << "h "
        << (elaped_sec / 60) % 60 << "m "
        << elaped_sec % 60 << "s." << std::endl;
}
