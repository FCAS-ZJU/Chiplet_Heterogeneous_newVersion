
# Benchmarks

You can learn the following topics from this page:

- Create or modify the source code of a benchmark to execute on the LegoSim.
- Create the configuration file (YAML format) of a novel benchmark.

## Create Source Codes of Benchmarks

Considering the complexity of the software stack of heterogeneous systems, it should not be expected that there will be a standard software stack available for every experimental platform on LegoSim. Hence, task partitioning and task management should be done manually. Each simulator should have one individual executable file. For example, different executable files should be provided to SniperSim and GPGPUSim. Moreover, imported simulators can share the same executable file if they perform the same task on various datasets, like Same-Task-Multiple-Data (STMD).

Take the *matmul* benchmark as an example. In *matmul* benchmark, the CPU generates source matrixes and sends data to GPGPUs, while three GPGPUs perform the matrix multiplex operation. Each GPGPU handles a part of the data. The *matmul* benchmark provides two executable files for SniperSim and GPGPUSim. Simulation processes of GPGPUSim share the same executable file.

Several APIs should be added to benchmarks to communicate and synchronize processes. In reality, similar APIs are also injected by one complete software stack (like CUDA).

## API List on CPU

APIs on CPUs are implemented by system calls. System Calls can have a number of arguments. The mapping between APIs and system calls is listed below:

| API | System Call ID |
| ---- | ---- |
| `launch`         | `SYSCALL_LAUNCH`       |
| `waitLaunch`     | `SYSCALL_WAITLAUNCH`   |
| `lock`           | `SYSCALL_LOCK`         |
| `unlock`         | `SYSCALL_UNLOCK`       |
| `barrier`        | `SYSCALL_BARRIER`      |
| `sendMessage`    | `SYSCALL_REMOTE_WRITE` |
| `receiveMessage` | `SYSCALL_REMOTE_READ`  |

Functions return the result of the operation. 0 means the transmission operation succeeds, while 1 means the transmission operation fails.

**Communication**

```cpp
syscall_return_t sendMessage(int64_t __dst_x, int64_t __dst_y, int64_t __src_x, int64_t __src_y, void* __addr, int64_t __nbyte);
syscall_return_t receiveMessage(int64_t __dst_x, int64_t __dst_y, int64_t __src_x, int64_t __src_y, void* __addr, int64_t __nbyte);
```

`__dst_x` and `__dst_y` specify the destination address. `__src_x` and `__src_y` specify the source address. `__addr` specifies the pointer to the data array. `__nbyte` determines the number of bytes in the data array.

**Lock and unlock**

```cpp
syscall_return_t lock(int64_t __uid, int64_t __src_x, int64_t __src_y);
syscall_return_t unlock(int64_t __uid, int64_t __src_x, int64_t __src_y);
```

`__uid` specifies one unique ID of the mutex. `__uid` should not be the same as any one address in the system. `__src_x` and `__src_y` specify the source address.

**Barrier**

```cpp
syscall_return_t barrier(int64_t __uid, int64_t __src_x, int64_t __src_y, int64_t __count = 0);
```

`__uid` specifies one unique ID of the barrier. `__uid` should not be the same as any one address in the system. `__src_x` and `__src_y` specify the source address. `__count` specifies the number of threads for the barrier.

**Launch**

```cpp
syscall_return_t launch(int64_t __dst_x, int64_t __dst_y, int64_t __src_x, int64_t __src_y);
syscall_return_t waitLaunch(int64_t __dst_x, int64_t __dst_y, int64_t* __src_x, int64_t* __src_y);
```

`__dst_x` and `__dst_y` specify the destination address. `__src_x` and `__src_y` specify the source address. `waitLaunch` returns the source of the launch command through `__src_x` and `__src_y`.

The declaration of these APIs is provided in [`$SIMULATOR_ROOT/interchiplet/includes/apis_c.h`](apiProject1/apis__c_8h.md). The implementation of these APIs is compiled into a static library `$SIMULATOR_ROOT/interchiplet/lib/libinterchiplet_c.a`, which should be linked to the benchmark.

> TODO: A more flexible way to specify the source and the destination address.

## API List on CUDA

APIs on CUDA platforms are implemented by built-in CUDA APIs. These functions return one value of `cudaError_t`.

**Communication**

```cpp
cudaError_t sendMessage(int __dst_x, int __dst_y, int __src_x, int __srx_y, void* __addr, int __nbyte);
cudaError_t receiveMessage(int __dst_x, int __dst_y, int __src_x, int __srx_y, void* __addr, int __nbyte);
```

`__dst_x` and `__dst_y` specify the destination address. `__src_x` and `__src_y` specify the source address. `__addr` specifies the pointer to the data array. `__nbyte` determines the number of bytes in the data array.

**Lock and unlock**

```cpp
cudaError_t lock(int __uid, int __src_x, int __src_y);
cudaError_t unlock(int __uid, int __src_x, int __src_y);
```

`__uid` specifies one unique ID of the mutex. `__uid` should not be the same as any one address in the system. `__src_x` and `__src_y` specify the source address.

**Barrier**

```cpp
cudaError_t barrier(int __uid, int __src_x, int __src_y, int __count = 0);
```

`__uid` specifies one unique ID of the barrier. `__uid` should not be the same as any one address in the system. `__src_x` and `__src_y` specify the source address. `__count` specifies the number of threads for the barrier.

**Launch**

```cpp
cudaError_t launch(int __dst_x, int __dst_y, int __src_x, int __src_y);
cudaError_t waitLaunch(int __dst_x, int __dst_y, int* __src_x, int* __src_y);
```

`__dst_x` and `__dst_y` specify the destination address. `__src_x` and `__src_y` specify the source address. `waitLaunch` returns the source of the launch command through `__src_x` and `__src_y`.

The declaration of these APIs is provided in [`$SIMULATOR_ROOT/interchiplet/includes/apis_cu.h`](apiProject1/apis__cu_8h.md). The implementation of these APIs is provided by the CUDA simulator, like GPGPU-Sim. Hence, when compiling executable files for CUDA platforms, the specified CUDA library should be provided as below:

```makefile
# CUDA language target
CUDA_target: $(CUDA_OBJS)
	$(NVCC) -L$(SIMULATOR_ROOT)/gpgpu-sim/lib/$(GPGPUSIM_CONFIG) --cudart shared $(CUDA_OBJS) -o $(CUDA_TARGET)
```

> TODO: A more flexible way to specify the source and the destination address.

## YAML Configuration File Format

The execution process is controlled by a YAML configuration file. One benchmark must have at least one YAML configuration file. More configuration files can be created to describe different configurations of one benchmark.

The example structure of the YAML file is as follows:

```yaml
# Phase 1 configuration.
phase1:
  # Process 0
  - cmd: "$BENCHMARK_ROOT/bin/matmul_cu"
    args: ["0", "1"]
    log: "gpgpusim.0.1.log"
    is_to_stdout: false
    pre_copy: "$SIMULATOR_ROOT/gpgpu-sim/configs/tested-cfgs/SM2_GTX480/*"
    clock_rate: 1
  # Process 1
  - cmd: "$BENCHMARK_ROOT/bin/matmul_cu"
    args: ["1", "0"]
    log: "gpgpusim.1.0.log"
    is_to_stdout: false
    pre_copy: "$SIMULATOR_ROOT/gpgpu-sim/configs/tested-cfgs/SM2_GTX480/*"
    clock_rate: 1
  ......

# Phase 2 configuration.
phase2:
  # Process 0
  - cmd: "$SIMULATOR_ROOT/popnet_chiplet/build/popnet"
    args: ["-A", "2", "-c", "2", "-V", "3", "-B", "12", "-O", "12", "-F", "4", "-L", "1000", "-T", "10000000", "-r", "1", "-I", "../bench.txt", "-R", "0", "-D", "../delayInfo.txt", "-P"]
    log: "popnet_0.log"
    is_to_stdout: false
    clock_rate: 1
```

In the above configuration files, the first-level tags are

* `phase1` provides the configuration for processes in Phase 1.
* `phase2` provides the configuration for processes in Phase 2.

Both `phase1` and `phase2` accept a list of process configuration structures. Each structure corresponds to one parallel simulator process.

Configuration structures provide the following tags:

- `cmd` ppresents the command of the simulator. A string is accepted. The environment variables `$BENCHMARK_ROOT` and `$SIMULATOR_ROOT` are supported to describe the path of the simulator.
- `args` presents the arguments of the simulator. A list of strings is accepted. The environment variables `$BENCHMARK_ROOT` and `$SIMULATOR_ROOT` are also supported to specify the path of related files. `cmd` and `args` combine the SHELL command to execute one simulator.
- `log` presents the name of the logger. A string is accepted. Neither the absolute path nor the related path is supported. The log file is stored in the sub-directory of each simulation process.
- `is_to_stdout` presents whether the standard output and standard error output of this simulator process are redirected to the standard output of the intercoupled.
- `pre_copy` provides a list of files that should be copied to the sub-directory of this simulation process before calling the simulator. A string is accepted. If there are multiple files to copy, files are separated by space.
- `clock_rate` provides a floating-pointing number as the ratio between chiplet clocks (clocks of simulators) and the system clock (clock of the *interchiplet*).

> TODO: Change pre_copy to pre_cmd.

The following commands are supported when writing one benchmark configuration file.

- `$BENCHMARK_ROOT` presents the root path of the benchmark, specified by the location of the YAML configuration file.
- `$SIMULATOR_ROOT` presents the root path of the LegoSim, set by *setup_env.sh*.
