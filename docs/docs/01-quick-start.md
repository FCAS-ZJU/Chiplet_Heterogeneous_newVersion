
# Quick Start

You can learn the following topics from this page:

1. Install the LegoSim.
2. Execute benchmarks provided in the LegoSim.

## Installation

**1. Clone the repository from GitHub and enter the downloaded repository.**

```shell
git clone --single-branch --branch master_v2 https://github.com/FCAS-SCUT/Chiplet_Heterogeneous_newVersion.git
cd Chiplet_Heterogeneous_newVersion
```

The following commands are executed under the root folder. The root folder is defined as `$SIMULATOR_ROOT` in the environment variables.

**2. Initialize and update submodules.** Third-party repositories such as SniperSim and GPGPUSim will be downloaded and updated.

```shell
git submodule init
git submodule update
```

**3. Setup environment.**

```shell
source setup_env.sh
```

If the operation succeeds, `setup_environment succeeded` is printed to the terminal.

**4. Apply modifications to Imported Simulators.** The simulator can still execute benchmarks if Step 4 is skipped, but no interactions between chiplets are performed.

```shell
./apply_path.sh
```

**5. Compile Imported Simulators.**

The installation procedure can be skipped for unnecessary simulators.

**5.1. Compile SniperSim.**

```shell
cd snipersim
make -j4
```

**5.2. Gem5.** The detailed installation guide for Gem5 can be found at https://www.gem5.org/documentation/learning_gem5/part1/building/.

If requirements have been addressed, X86 and ARM simulators can be used in LegoSim.

```shell
cd gem5
scons build/X86/gem5.opt -j4
```

or

```shell
cd gem5
scons build/ARM/gem5.opt -j4
```

**5.3. Compile GPGPUSim.** Installation of GPGPUSim requires the following conditions:

- CUDA environment is correctly installed and configured. Imported GPGPUSim supports CUDA versions from 4.0 to 11.0. See README within *gpgpu-sim* for details.
- GPGPUSim has a limitation on the compiler version. According to our experiment, GCC7 is recommended.

```shell
cd gpgpu-sim
make -j4
```

**6. Compile Popnet.**

```shell
cd popnet
mkdir build
cd build
cmake ..
make -j4
```

**7. Compile the LegoSim.** The compile process is managed by CMake.

```shell
cd interchiplet
mkdir build
cd build
cmake ..
make
```

If the operation succeeds, *interchiplet* can be found under *interchiplet/bin*, while *libinterchiplet_c.a* and *libinterchiplet_cu.a* can be found under *interchiplet/lib*.

## Execute Benchmark

To check whether the LegoSim is installed and configured correctly, you can execute the demo benchmark *matmul*.

**1. Setup environment.**

```shell
source setup_env.sh
```

If the operation succeeds, `setup_environment succeeded` is printed to the terminal.

**2. Compile execution files for benchmark.**

```shell
cd benchmark/matmul
make
```

If operation successes, *matmul_c* and *matmul_cu* can be found under *benchmark/matmul/bin*.

**3. Execute simulation.**

```shell
$SIMULATOR_ROOT/interchiplet/bin/interchiplet ./matmul.yml
```

The *matmul* benchmark launches four processes, which are 1 CPU process and 3 GPGPU processes.

If the benchmark executes correctly, you will find it experiences two rounds of simulation. Also, the following information can be found on your terminal.

```log
[2024-06-23 14:48:59.546] [info] ==== LegoSim Chiplet Simulator ====
[2024-06-23 14:48:59.547] [info] Load benchmark configuration from matmul.yml.
[2024-06-23 14:48:59.547] [info] **** Round 1 Phase 1 ****
[2024-06-23 14:48:59.547] [info] Load 0 delay records.
[2024-06-23 14:48:59.548] [info] Start simulation process 1617365. Command: /data_sda/junwan02/Chiplet_Heterogeneous_newVersion/benchmark/matmul/bin/matmul_cu
[2024-06-23 14:48:59.548] [info] Start simulation process 1617366. Command: /data_sda/junwan02/Chiplet_Heterogeneous_newVersion/benchmark/matmul/bin/matmul_cu
[2024-06-23 14:48:59.549] [info] Start simulation process 1617368. Command: /data_sda/junwan02/Chiplet_Heterogeneous_newVersion/benchmark/matmul/bin/matmul_cu
[2024-06-23 14:48:59.549] [info] Start simulation process 1617370. Command: /data_sda/junwan02/Chiplet_Heterogeneous_newVersion/snipersim/run-sniper
[2024-06-23 14:49:14.522] [info] Simulation process 1617366 terminate with status = 0.
[2024-06-23 14:49:14.522] [info] Simulation process 1617365 terminate with status = 0.
[2024-06-23 14:49:15.520] [info] Simulation process 1617368 terminate with status = 0.
[2024-06-23 14:49:16.807] [info] Simulation process 1617370 terminate with status = 0.
[2024-06-23 14:49:16.807] [info] All process has exit.
[2024-06-23 14:49:16.807] [info] Dump 9 bench records.
[2024-06-23 14:49:16.807] [info] Benchmark elapses 2910097 cycle.
[2024-06-23 14:49:16.807] [info] **** Round 1 Phase 2 ***
[2024-06-23 14:49:16.808] [info] Start simulation process 1618141. Command: /data_sda/junwan02/Chiplet_Heterogeneous_newVersion/popnet_chiplet/build/popnet
[2024-06-23 14:49:16.961] [info] Simulation process 1618141 terminate with status = 0.
[2024-06-23 14:49:16.961] [info] All process has exit.
[2024-06-23 14:49:16.961] [info] Round 1 elapses 0d 0h 0m 17s.
[2024-06-23 14:49:16.961] [info] **** Round 2 Phase 1 ****
[2024-06-23 14:49:16.962] [info] Load 9 delay records.
[2024-06-23 14:49:16.962] [info] Start simulation process 1618146. Command: /data_sda/junwan02/Chiplet_Heterogeneous_newVersion/benchmark/matmul/bin/matmul_cu
[2024-06-23 14:49:16.963] [info] Start simulation process 1618147. Command: /data_sda/junwan02/Chiplet_Heterogeneous_newVersion/benchmark/matmul/bin/matmul_cu
[2024-06-23 14:49:16.963] [info] Start simulation process 1618149. Command: /data_sda/junwan02/Chiplet_Heterogeneous_newVersion/benchmark/matmul/bin/matmul_cu
[2024-06-23 14:49:16.964] [info] Start simulation process 1618151. Command: /data_sda/junwan02/Chiplet_Heterogeneous_newVersion/snipersim/run-sniper
[2024-06-23 14:49:30.816] [info] Simulation process 1618146 terminate with status = 0.
[2024-06-23 14:49:30.816] [info] Simulation process 1618147 terminate with status = 0.
[2024-06-23 14:49:31.811] [info] Simulation process 1618149 terminate with status = 0.
[2024-06-23 14:49:32.219] [info] Simulation process 1618151 terminate with status = 0.
[2024-06-23 14:49:32.219] [info] All process has exit.
[2024-06-23 14:49:32.219] [info] Dump 9 bench records.
[2024-06-23 14:49:32.219] [info] Benchmark elapses 2909345 cycle.
[2024-06-23 14:49:32.219] [info] Difference related to pervious round is 0.025847742361253135%.
[2024-06-23 14:49:32.219] [info] Quit simulation because simulation cycle has converged.
[2024-06-23 14:49:32.219] [info] **** End of Simulation ****
[2024-06-23 14:49:32.219] [info] Benchmark elapses 2909345 cycle.
[2024-06-23 14:49:32.219] [info] Simulation elapseds 0d 0h 0m 33s.
```

The cycle elapsed by benchmark is about 5.77M cycles with a minor error on different platforms. The time elapsed by the simulator depends on the performance of the host machines, so it varies across different platforms.
 
**4. Clear executable files and output files.**

```shell
make clean
```

## Command Line Interface of *interchiplet*

The user interface of the LegoSim is **interchiplet** that is provided in *$SIMULATOR_ROOT/interchiplet/bin*. The command line format of *interchiplet* is shown below:

```shell
interchiplet <bench>.yml [--cwd <string>] [-t|--timeout <int>] [-e|--error <float>] [--debug] [-h]
```

Command line options:

- `<bench>.yml` specifies the configuration file of the benchmark. This option is mandatory.
- `--cwd <string>` specifies the execution directory of the simulation. This option is optional. If not provided, the simulation is executed in the current working directory.
- `-t <int>` or `--timeout <int>` specifies the time-out threshold of simulation in units of simulation rounds. No matter whether the cycle elapsed by the benchmark has converged, the simulation will quit after a certain account of simulation rounds specified by this option. This option is optional. A default value of 5 is used if not provided.
- `e <float>` or `--error <float>` specifies the convergence condition of the cycle elapsed by the benchmark. If the error of cycle elapsed by the benchmark between simulation rounds is lower than the specified value, the simulation will stop. This option is optional. A default value of 0.05 is used if not provided.
- `--debug` defines the verbosity of the output log. If `--debug` appears, the details of the synchronization protocol are printed on the stand output.

For example,

```shell
$SIMULATOR_ROOT/interchiplet/bin/interchiplet $SIMULATOR_ROOT/benchmark/matmul/matmul.yml
```

The above command starts the simulation of *matmul* in the current directory. The simulation will last at most five iterations and quit if the error of cycle elapsed by benchmark is lower than 0.5%.
