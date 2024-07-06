# Project environment
# SIMULATOR_ROOT, defined by setup_env.sh
BENCHMARK_ROOT=$(SIMULATOR_ROOT)/benchmark/test/gem5/pipe

# Compiler environment of C/C++
CC=aarch64-linux-gnu-g++
CFLAGS=-Wall -Werror -g -I$(SIMULATOR_ROOT)/interchiplet/includes

# C/C++ Source file
C_SRCS=test.cpp
C_OBJS=obj/test.o
C_TARGET=bin/test_c

all: bin_dir obj_dir C_target

# C language target
C_target: $(C_OBJS) obj/interchiplet.o
	$(CC) $(C_OBJS) obj/interchiplet.o -o $(C_TARGET)

# Interchiplet library
obj/interchiplet.o: ../../../../interchiplet/srcs/apis_c.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Rule for C object
obj/%.o: ../../snipersim/pipe/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Directory for binary files.
bin_dir:
	mkdir -p bin

# Directory for object files for C.
obj_dir:
	mkdir -p obj

# Clean generated files.
clean:
	rm -rf bench.txt delayInfo.txt buffer* message_record.txt
	rm -rf proc_r*_t* *.log
	rm -rf obj bin
