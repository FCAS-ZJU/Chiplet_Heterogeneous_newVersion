
CC=g++
CFLAGS=-Wall -Werror -g -I$(SIMULATOR_ROOT)/interchiplet/includes
INTERCHIPLETLIB=$(SIMULATOR_ROOT)/interchiplet/lib/libinterchiplet_app.a
SNIPER_EXEC=$(SIMULATOR_ROOT)/snipersim/run-sniper

NVCC=nvcc

BENCHMARK_ROOT=$(SIMULATOR_ROOT)/benchmark/matmul

SNIPER_SRCS=matmul.cpp
SNIPER_OBJS=obj/matmul.o
SNIPER_TARGET=bin/matmul_cpu

GPGPUSIM_SRCS=matmul.cu
GPGPUSIM_TARGET=bin/matmul_cu

all: bin_dir obj_dir sniper_target gpgpusim_target

sniper_target: $(SNIPER_OBJS)
	$(CC) $(SNIPER_OBJS) $(INTERCHIPLETLIB) -o $(SNIPER_TARGET)

obj/%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

gpgpusim_target:
	$(NVCC) --cudart shared $(GPGPUSIM_SRCS) -o $(GPGPUSIM_TARGET)

bin_dir:
	mkdir -p bin

obj_dir:
	mkdir -p obj

run:
	mkfifo buffer0_0_0_1 buffer0_0_1_0 buffer0_0_1_1 buffer0_1_0_0 buffer1_0_0_0 buffer1_1_0_0
	./$(GPGPUSIM_TARGET) 0 1 > gpgpusim.0.1.log 2>&1 &
	./$(GPGPUSIM_TARGET) 1 0 > gpgpusim.1.0.log 2>&1 &
	./$(GPGPUSIM_TARGET) 1 1 > gpgpusim.1.1.log 2>&1 &
	$(SNIPER_EXEC) --curdir $(BENCHMARK_ROOT) -- $(BENCHMARK_ROOT)/$(SNIPER_TARGET) 0 0

clean:
	rm -rf obj bin
	rm -rf bench* buffer* message_record.txt
	rm -rf _app_cuda_version_* _cuobjdump_list_ptx_* gpgpusim_power_report_* *.ptx *.ptxas gpgpusim.*.log gpgpu_inst_stats.txt
	rm -rf sim.cfg sim.info sim.out sim.stats.sqlite3
	rm -rf checkpoint_files

cleanall:
	make clean
	rm -rf $(SNIPER_OBJS) $(GPGPUSIM_TARGET) $(SNIPER_TARGET)
