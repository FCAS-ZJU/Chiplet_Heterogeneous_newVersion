
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fstream>
#include <iostream>
#include <vector>
#include <set>
#include <sstream>
#include <string>

#include <error.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "intercomm.h"
#include "network_interface.h"

/**
 * @brief Data structure of synchronize operation.
 */
class SyncStruct
{
public:
    SyncStruct()
        : m_cycle(0)
    {
    }

public:
    pthread_mutex_t m_mutex;

    long long int m_cycle;

    BenchList m_bench_list;
    BenchList m_delay_list;

    std::vector<std::string> m_fifo_set;
    std::vector<nsInterchiplet::SyncCommand> m_read_cmd_set;
    std::vector<nsInterchiplet::SyncCommand> m_write_cmd_set;
};

/**
 * @brief Data structure of process configuration.
 */
class ProcessConfig
{
public:
    ProcessConfig(const std::string& __cmd,
                  const std::vector<std::string>& __args,
                  const std::string& __log,
                  bool __to_stdout)
        : m_command(__cmd)
        , m_args(__args)
        , m_log_file(__log)
        , m_to_stdout(__to_stdout)
        , m_unfinished_line()
        , m_thread_id()
        , m_pid(-1)
        , m_pid2(-1)
        , m_sync_struct(NULL)
    {}

public:
    std::string m_command;
    std::vector<std::string> m_args;
    std::string m_log_file;
    bool m_to_stdout;

    std::string m_unfinished_line;

    pthread_t m_thread_id;
    int m_pid;
    int m_pid2;
    SyncStruct* m_sync_struct;
};

// Create FIFO with specified name.
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

void handle_pipe_cmd(const nsInterchiplet::SyncCommand& __cmd,
                     SyncStruct* __sync_struct,
                     int __stdin_fd)
{
    std::cout << "From pipe " << __stdin_fd
        << " Handle PIPE command from " << __cmd.m_src_x << " " << __cmd.m_src_y
        << " to " << __cmd.m_dst_x << " " << __cmd.m_dst_y << std::endl;
    // Check pipe file.
    std::string file_name = nsInterchiplet::SyncProtocol::pipeNameString(
        __cmd.m_src_x, __cmd.m_src_y, __cmd.m_dst_x, __cmd.m_dst_y);

    bool has_fifo_file = false;
    for (auto& __name: __sync_struct->m_fifo_set)
    {
        if (__name == file_name)
        {
            has_fifo_file = true;
            break;
        }
    }
    
    if (!has_fifo_file)
    {
        if (create_fifo(file_name.c_str()) == 0)
        {
            __sync_struct->m_fifo_set.push_back(file_name);
        }
    }

    std::stringstream ss;
    ss << "[INTERCMD] SYNC " << 0 << std::endl;
    write(__stdin_fd, ss.str().c_str(), ss.str().size());
}

void handle_read_cmd(const nsInterchiplet::SyncCommand& __cmd,
                     SyncStruct* __sync_struct,
                     int __stdin_fd)
{
    std::cout << "From pipe " << __stdin_fd
        << " Handle READ command from " << __cmd.m_src_x << " " << __cmd.m_src_y
        << " to " << __cmd.m_dst_x << " " << __cmd.m_dst_y << std::endl;
    // Check pipe file.
    std::string file_name = nsInterchiplet::SyncProtocol::pipeNameString(
        __cmd.m_src_x, __cmd.m_src_y, __cmd.m_dst_x, __cmd.m_dst_y);
    
    bool has_write_cmd = false;
    nsInterchiplet::SyncCommand write_cmd;
    for (std::size_t i = 0; i < __sync_struct->m_write_cmd_set.size(); i ++)
    {
        nsInterchiplet::SyncCommand& __write_cmd = __sync_struct->m_write_cmd_set[i];
        if (__write_cmd.m_src_x == __cmd.m_src_x && __write_cmd.m_src_y == __cmd.m_src_y &&
            __write_cmd.m_dst_x == __cmd.m_dst_x && __write_cmd.m_dst_y == __cmd.m_dst_y &&
            __write_cmd.m_nbytes == __cmd.m_nbytes)
        {
            has_write_cmd = true;
            write_cmd = __write_cmd;
            __sync_struct->m_write_cmd_set.erase(__sync_struct->m_write_cmd_set.begin() + i);
            break;
        }
    }

    if (!has_write_cmd)
    {
        __sync_struct->m_read_cmd_set.push_back(__cmd);
        __sync_struct->m_read_cmd_set[__sync_struct->m_read_cmd_set.size() - 1].m_stdin_fd =
            __stdin_fd;
        std::cout << "Register READ command to pair with WRITE command." << std::endl;
    }
    else
    {
        std::cout << "Pair with WRITE command." << std::endl;

        long long int end_cycle = __sync_struct->m_delay_list.getEndCycle(write_cmd, __cmd);

        // Send sync command.
        std::stringstream ss;
        ss << "[INTERCMD] SYNC " << end_cycle << std::endl;
        write(__stdin_fd, ss.str().c_str(), ss.str().size());

        ss.clear();
        ss << "[INTERCMD] SYNC " << end_cycle << std::endl;
        write(write_cmd.m_stdin_fd, ss.str().c_str(), ss.str().size());
    }
}

void handle_write_cmd(const nsInterchiplet::SyncCommand& __cmd,
                      SyncStruct* __sync_struct,
                      int __stdin_fd)
{
    std::cout << "From pipe " << __stdin_fd
        << " WRITE command from " << __cmd.m_src_x << " " << __cmd.m_src_y
        << " to " << __cmd.m_dst_x << " " << __cmd.m_dst_y << std::endl;
    // Check pipe file.
    std::string file_name = nsInterchiplet::SyncProtocol::pipeNameString(
        __cmd.m_src_x, __cmd.m_src_y, __cmd.m_dst_x, __cmd.m_dst_y);

    // Insert benchmark;        
    BenchItem bench_item(__cmd);
    __sync_struct->m_bench_list.insert(bench_item);

    bool has_read_cmd = false;
    nsInterchiplet::SyncCommand read_cmd;
    for (std::size_t i = 0; i < __sync_struct->m_read_cmd_set.size(); i ++)
    {
        nsInterchiplet::SyncCommand& __read_cmd = __sync_struct->m_read_cmd_set[i];
        if (__read_cmd.m_src_x == __cmd.m_src_x && __read_cmd.m_src_y == __cmd.m_src_y &&
            __read_cmd.m_dst_x == __cmd.m_dst_x && __read_cmd.m_dst_y == __cmd.m_dst_y &&
            __read_cmd.m_nbytes == __cmd.m_nbytes)
        {
            has_read_cmd = true;
            read_cmd = __read_cmd;
            __sync_struct->m_read_cmd_set.erase(__sync_struct->m_read_cmd_set.begin() + i);
            break;
        }
    }

    if (!has_read_cmd)
    {
        __sync_struct->m_write_cmd_set.push_back(__cmd);
        __sync_struct->m_write_cmd_set[__sync_struct->m_write_cmd_set.size() - 1].m_stdin_fd =
            __stdin_fd;
        std::cout << "Register WRITE command to pair with READ command." << std::endl;
    }
    else
    {
        std::cout << "Pair with READ command." << std::endl;

        long long int end_cycle = __sync_struct->m_delay_list.getEndCycle(__cmd, read_cmd);

        // Send sync command.
        std::stringstream ss;
        ss << "[INTERCMD] SYNC " << end_cycle << std::endl;
        write(__stdin_fd, ss.str().c_str(), ss.str().size());

        ss.clear();
        ss << "[INTERCMD] SYNC " << end_cycle << std::endl;
        write(read_cmd.m_stdin_fd, ss.str().c_str(), ss.str().size());
    }
}

void handle_cycle_cmd(const nsInterchiplet::SyncCommand& __cmd,
                      SyncStruct* __sync_struct,
                      int __stdin_fd)
{
    long long int new_cycle = __cmd.m_cycle;
    if (__sync_struct->m_cycle < new_cycle)
    {
        __sync_struct->m_cycle = new_cycle;
    }
}

void handle_command(const std::string& __cmd, SyncStruct* __sync_struct, int __stdin_fd)
{
    nsInterchiplet::SyncCommand cmd = nsInterchiplet::SyncProtocol::parseCmd(__cmd);

    pthread_mutex_lock(&__sync_struct->m_mutex);

    switch(cmd.m_type)
    {
    case nsInterchiplet::SC_CYCLE:
        handle_cycle_cmd(cmd, __sync_struct, __stdin_fd);
        break;
    case nsInterchiplet::SC_PIPE:
        handle_pipe_cmd(cmd, __sync_struct, __stdin_fd);
        break;
    case nsInterchiplet::SC_READ:
        handle_read_cmd(cmd, __sync_struct, __stdin_fd);
        break;
    case nsInterchiplet::SC_WRITE:
        handle_write_cmd(cmd, __sync_struct, __stdin_fd);
        break;
    default:
        break;
    }

    pthread_mutex_unlock(&__sync_struct->m_mutex);
}

void parse_command(char* __pipe_buf, ProcessConfig* __proc_cfg, int __stdin_fd)
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
    if (__proc_cfg->m_unfinished_line.size() > 0)
    {
        lines[0] = __proc_cfg->m_unfinished_line + lines[0];
        __proc_cfg->m_unfinished_line = "";
    }
    if (lines[lines.size() - 1].find('\n') == -1)
    {
        __proc_cfg->m_unfinished_line = lines[lines.size() - 1];
        lines.pop_back();
    }

    // Get line start with [INTERCMD]
    for (std::size_t i = 0; i < lines.size(); i ++)
    {
        std::string l = lines[i];
        if (l.substr(0, 10) == "[INTERCMD]")
        {
            handle_command(l, __proc_cfg->m_sync_struct, __stdin_fd);
        }
    }
}

void *bridge_thread(void * __args_ptr)
{
    ProcessConfig* proc_cfg = (ProcessConfig*)__args_ptr;

    int pipe_stdin[2]; // Pipe to send data to child process
    int pipe_stdout[2]; // Pipe to receive data from child process
    int pipe_stderr[2]; // Pipe to receive data from child process

    // Create pipes
    if (pipe(pipe_stdin) == -1 || pipe(pipe_stdout) == -1 || pipe(pipe_stderr) == -1)
    {
        perror("pipe");
        exit(EXIT_FAILURE);
    }

    std::cout << "Enter bridge_thread " << proc_cfg->m_command << std::endl;
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

        // Execute the child program
        int argc = proc_cfg->m_args.size();
        char** args_list = new char*[argc + 2];
        args_list[0] = new char[proc_cfg->m_command.size() + 1];
        strcpy(args_list[0], proc_cfg->m_command.c_str());
        args_list[0][proc_cfg->m_command.size()] = '\0';
        for (int i = 0; i < argc; i ++)
        {
            int arg_len = proc_cfg->m_args[i].size();
            args_list[i + 1] = new char[arg_len + 1];
            strcpy(args_list[i + 1], proc_cfg->m_args[i].c_str());
            args_list[i + 1][arg_len] = '\0';
        }
        args_list[argc + 1] = NULL;

        std::cout << "Exec:";
        for (int i = 0; i < proc_cfg->m_args.size() + 1; i ++)
        {
            std::cout << " " << args_list[i];
        }
        std::cout << std::endl;
        execvp(proc_cfg->m_command.c_str(), args_list);

        // If execl fails, it means the child process couldn't be started
        perror("execvp");
        exit(EXIT_FAILURE);
    }
    else
    { // Parent process
        std::cout << "Start simulation process " << pid << std::endl;
        proc_cfg->m_pid2 = pid;

        // Close unnecessary pipe ends
        close(pipe_stdin[0]);
        close(pipe_stdout[1]);
        close(pipe_stderr[1]);
        int stdin_fd = pipe_stdin[1];
        int stdout_fd = pipe_stdout[0];
        int stderr_fd = pipe_stderr[0];

        pollfd fd_list[2] = {{fd: stdout_fd, events: POLL_IN},
                             {fd: stderr_fd, events: POLL_IN}};

        char* pipe_buf = new char[PIPE_BUF_SIZE + 1];
        std::ofstream log_file(proc_cfg->m_log_file);

        bool to_stdout = proc_cfg->m_to_stdout;
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
                parse_command(pipe_buf, proc_cfg, stdin_fd);
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
                parse_command(pipe_buf, proc_cfg, stdin_fd);
            }

            // Check the status of child process and quit.
            int status;
            if (!has_stdout && (waitpid(pid, &status, WNOHANG) > 0))
            {
                // Optionally handle child process termination status
                std::cout << "Simulation process " << proc_cfg->m_pid2
                    << " terminate with status = " << status << "." << std::endl;
                break;
            }
        }

        delete pipe_buf;
    }

    return 0;
}

long long int __loop_phase_one(const std::vector<ProcessConfig*>& __proc_cfg_list)
{
    // Create synchronize data structure.
    SyncStruct* g_sync_structure = new SyncStruct();
    if (pthread_mutex_init(&g_sync_structure->m_mutex, NULL) < 0)
    {
        perror("pthread_mutex_init");
        exit(EXIT_FAILURE);
    }

    // Load delay record.
    g_sync_structure->m_delay_list.load_delay();
    std::cout << "Load " << g_sync_structure->m_delay_list.size() << " delay records." << std::endl;

    // Create multi-thread.
    for (auto& proc_cfg: __proc_cfg_list)
    {
        proc_cfg->m_sync_struct = g_sync_structure;
        int res = pthread_create(&(proc_cfg->m_thread_id), NULL, bridge_thread, (void*)proc_cfg);
        if (res < 0)
        {
            perror("pthread_create");
            exit(EXIT_FAILURE);
        }
    }

    // Wait threads to finish.
    for (auto& proc_cfg: __proc_cfg_list)
    {
        pthread_join(proc_cfg->m_thread_id, NULL);
    }
    std::cout << "All process has exit." << std::endl;

    // End handle.
    pthread_mutex_destroy(&g_sync_structure->m_mutex);

    // Remove file.
    for (auto& __name: g_sync_structure->m_fifo_set)
    {
        remove(__name.c_str());
    }

    // Dump benchmark record.
    g_sync_structure->m_bench_list.dump_bench();
    std::cout << "Dump " << g_sync_structure->m_bench_list.size() << " bench records." << std::endl;

    long long int res_cycle = g_sync_structure->m_cycle;
    delete g_sync_structure;
    return res_cycle;
}

void __loop_phase_two(const std::vector<ProcessConfig*>& __proc_cfg_list)
{
    // Create synchronize data structure.
    SyncStruct* g_sync_structure = new SyncStruct();
    if (pthread_mutex_init(&g_sync_structure->m_mutex, NULL) < 0)
    {
        perror("pthread_mutex_init");
        exit(EXIT_FAILURE);
    }

    // Create multi-thread.
    for (auto& proc_cfg: __proc_cfg_list)
    {
        proc_cfg->m_sync_struct = g_sync_structure;
        int res = pthread_create(&(proc_cfg->m_thread_id), NULL, bridge_thread, (void*)proc_cfg);
        if (res < 0)
        {
            perror("pthread_create");
            exit(EXIT_FAILURE);
        }
    }

    // Wait threads to finish.
    for (auto& proc_cfg: __proc_cfg_list)
    {
        pthread_join(proc_cfg->m_thread_id, NULL);
    }
    std::cout << "All process has exit." << std::endl;

    // End handle.
    pthread_mutex_destroy(&g_sync_structure->m_mutex);
    delete g_sync_structure;
}

int main(int argc, char** argv)
{
    // Parse commandline arguments.

    // Parse process configuration.
    std::vector<ProcessConfig*> phase1_proc_cfg_list;
    phase1_proc_cfg_list.push_back(
        new ProcessConfig("./bin/matmul_cu", {"0", "1"}, "gpgpusim.0.1.log", false));
    phase1_proc_cfg_list.push_back(
        new ProcessConfig("./bin/matmul_cu", {"1", "0"}, "gpgpusim.1.0.log", false));
    phase1_proc_cfg_list.push_back(
        new ProcessConfig("./bin/matmul_cu", {"1", "1"}, "gpgpusim.1.1.log", false));
    phase1_proc_cfg_list.push_back(new ProcessConfig(
        "../../snipersim/run-sniper", {"--", "./bin/matmul_cpu", "0", "0"}, "sniper.0.0.log", false));

    std::vector<ProcessConfig*> phase2_proc_cfg_list;
    // ./popnet -A 9 -c 2 -V 3 -B 12 -O 12 -F 4 -L 1000 -T 20000 -r 1 -I ./bench.txt -R 0
    phase2_proc_cfg_list.push_back(
        new ProcessConfig("../../popnet/popnet",
            {"-A", "9", "-c", "2", "-V", "3", "-B", "12", "-O", "12", "-F", "4",
             "-L", "1000", "-T", "10000000", "-r", "1", "-I", "./bench.txt", "-R", "0"},
            "popnet.log", false));

    int timeout_round = 2;
    double err_rate_threshold = 0.005;

	struct timeval simstart, simend, roundstart, roundend;
	gettimeofday(&simstart, 0);

    long long sim_cycle = 0;
    for (int round = 1; round <= timeout_round; round ++)
    {
	    gettimeofday(&roundstart, 0);
        std::cout << "[COMMBRIDGE] *** Round " << round << " Phase 1 ***" << std::endl;
        long long int round_cycle = __loop_phase_one(phase1_proc_cfg_list);

        std::cout << "[COMMBRIDGE] Benchmark elapses " << round_cycle << " cycle." << std::endl;
        if (round > 1)
        {
            double err_rate = ((double)round_cycle - (double)sim_cycle) / (double)round_cycle;
            err_rate = err_rate < 0 ? - err_rate : err_rate;
            std::cout << "[COMMBRIDGE] Error related to pervious round is " << err_rate * 100 << "%." << std::endl;
            if (err_rate < err_rate_threshold)
            {
                std::cout << "[COMMBRIDGE] Quit simulation because simulation cycle has converged." << std::endl;
                sim_cycle = round_cycle;
                break;
            }
        }
        sim_cycle = round_cycle;

        std::cout << "[COMMBRIDGE] *** Round " << round << " Phase 2 ***" << std::endl;
        __loop_phase_two(phase2_proc_cfg_list);

        gettimeofday(&roundend, 0);

        unsigned long elaped_sec = roundend.tv_sec - roundstart.tv_sec;
        std::cout << "[COMMBRIDGE] Round " << round << " elapses "
            << elaped_sec / 3600 / 24 << "d "
            << (elaped_sec / 3600) % 24 << "h "
            << (elaped_sec / 60) % 60 << "m "
            << elaped_sec % 60 << "s." << std::endl;
    }

    std::cout << "[COMMBRIDGE] *** End of Simulation ***" << std::endl;
    gettimeofday(&simend, 0);
    unsigned long elaped_sec = simend.tv_sec - simstart.tv_sec;
    std::cout << "[COMMBRIDGE] Benchmark elapses " << sim_cycle << " cycle." << std::endl;
    std::cout << "[COMMBRIDGE] Simulation elapseds " 
        << elaped_sec / 3600 / 24 << "d "
        << (elaped_sec / 3600) % 24 << "h "
        << (elaped_sec / 60) % 60 << "m "
        << elaped_sec % 60 << "s." << std::endl;

    for (auto& item: phase1_proc_cfg_list)
    {
        delete item;
    }
    phase1_proc_cfg_list.clear();
    for (auto& item: phase2_proc_cfg_list)
    {
        delete item;
    }
    phase2_proc_cfg_list.clear();
}
