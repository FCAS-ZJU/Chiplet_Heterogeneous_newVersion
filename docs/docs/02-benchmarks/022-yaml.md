
# YAML Configuration File

You can learn the following topics from this page:

- Create the configuration file (YAML format) of a novel benchmark.

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
