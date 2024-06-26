CC=g++
CFLAGS=-Wall -Werror -g -I$(SIMULATOR_ROOT)/interchiplet/includes -I$(SIMULATOR_ROOT)/benchmark/MLP/json/include
INTERCHIPLETLIB=$(SIMULATOR_ROOT)/interchiplet/lib/libinterchiplet_c.a
SNIPER_EXEC=$(SIMULATOR_ROOT)/snipersim/run-sniper

NVCC=nvcc
CUFLAGS=--compiler-options -Wall -I$(SIMULATOR_ROOT)/interchiplet/includes

BENCHMARK_ROOT=$(SIMULATOR_ROOT)/benchmark/MLP

SNIPER_SRCS=mlp.cpp
SNIPER_OBJS=mlp.o
SNIPER_TARGET=bin/mlp_cpu

GPGPUSIM_SRCS=mlp.cu
GPGPUSIM_TARGET=bin/mlp_cu
REPO_DIR=$(BENCHMARK_ROOT)/json
GIT_REPO=https://github.com/nlohmann/json.git

all: sniper_target gpgpusim_target

sniper_target: $(SNIPER_OBJS)
	$(CC) -g $(SNIPER_OBJS) $(INTERCHIPLETLIB) -o $(SNIPER_TARGET) -lpthread

%.o: %.cpp
	if [ ! -d $(REPO_DIR) ]; then \
        git clone $(GIT_REPO) $(REPO_DIR); \
    fi
	
	$(CC) $(CFLAGS) -c $< -o $@

debug: CFLAGS += -DDEBUG -g
debug: all

gpgpusim_target:
	$(NVCC) -L$(SIMULATOR_ROOT)/gpgpu-sim/lib/$(GPGPUSIM_CONFIG) --cudart shared $(GPGPUSIM_SRCS) $(CUFLAGS) -o $(GPGPUSIM_TARGET) -g

run:
	touch "start running" &
	./$(GPGPUSIM_TARGET) 0 1 > gpgpusim.0.1.log 2>&1 &
	./$(GPGPUSIM_TARGET) 0 2 > gpgpusim.0.2.log 2>&1 &
	./$(GPGPUSIM_TARGET) 0 3 > gpgpusim.0.3.log 2>&1 &
	./$(GPGPUSIM_TARGET) 0 4 > gpgpusim.0.4.log 2>&1 &
	./$(GPGPUSIM_TARGET) 1 1 > gpgpusim.1.1.log 2>&1 &
	./$(GPGPUSIM_TARGET) 1 2 > gpgpusim.1.2.log 2>&1 &
	./$(GPGPUSIM_TARGET) 1 3 > gpgpusim.1.3.log 2>&1 &
	./$(GPGPUSIM_TARGET) 1 4 > gpgpusim.1.4.log 2>&1 &
	$(SNIPER_EXEC) --curdir $(BENCHMARK_ROOT) -- $(BENCHMARK_ROOT)/$(SNIPER_TARGET) 0 0 > sniper.0.0.log 2>&1

run_cpu:
	$(SNIPER_EXEC) --curdir $(BENCHMARK_ROOT) -- $(BENCHMARK_ROOT)/$(SNIPER_TARGET) 0 0

run_gpu:
	./$(GPGPUSIM_TARGET) 0 1 > gpgpusim.0.1.log 2>&1 


gdb: $(SNIPER_TARGET)
	cd $(BENCHMARK_ROOT) && gdb ./$(SNIPER_TARGET) -- $(BENCHMARK_ROOT)/$(SNIPER_TARGET) 0 0

valgrind: $(SNIPER_TARGET)
	cd $(BENCHMARK_ROOT) && valgrind --leak-check=full ./$(SNIPER_TARGET) -- $(BENCHMARK_ROOT)/$(SNIPER_TARGET) 0 0

clean:
	rm -rf bench* buffer* message_record.txt
	rm -rf _app_cuda_version_* _cuobjdump_list_ptx_* gpgpusim_power_report_* *.ptx *.ptxas gpgpusim.*.log gpgpu_inst_stats.txt gpuRead* cpuRead*  
	rm -rf sim.cfg sim.info sim.out sim.stats.sqlite3
	rm -rf checkpoint_files

cleanall:
	make clean 
	rm -rf $(SNIPER_OBJS) $(GPGPUSIM_TARGET) $(SNIPER_TARGET)
	rm -f start* sniper*

kill:
	pkill -f mlp_cu
	pkill -f mlp_cpu
