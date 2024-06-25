
# File interchiplet.cpp



[**FileList**](files.md) **>** [**interchiplet**](dir_a2025b34133129e5724d121abe9a4a4a.md) **>** [**srcs**](dir_b94c70d771af9f161858c2c4e7b3d1c5.md) **>** [**interchiplet.cpp**](interchiplet_8cpp.md)

[Go to the source code of this file.](interchiplet_8cpp_source.md)



* `#include <fcntl.h>`
* `#include <poll.h>`
* `#include <sys/stat.h>`
* `#include <sys/time.h>`
* `#include <sys/types.h>`
* `#include <sys/wait.h>`
* `#include <ctime>`
* `#include <vector>`
* `#include "benchmark_yaml.h"`
* `#include "cmd_handler.h"`
* `#include "cmdline_options.h"`
* `#include "spdlog/spdlog.h"`










## Classes

| Type | Name |
| ---: | :--- |
| class | [**ProcessStruct**](classProcessStruct.md) <br>_Data structure of process configuration._  |





## Public Functions

| Type | Name |
| ---: | :--- |
|  InterChiplet::InnerTimeType | [**\_\_loop\_phase\_one**](#function-__loop_phase_one) (int \_\_round, const std::vector&lt; [**InterChiplet::ProcessConfig**](classInterChiplet_1_1ProcessConfig.md) &gt; & \_\_proc\_phase1\_cfg\_list, const std::vector&lt; [**InterChiplet::ProcessConfig**](classInterChiplet_1_1ProcessConfig.md) &gt; & \_\_proc\_phase2\_cfg\_list) <br> |
|  void | [**\_\_loop\_phase\_two**](#function-__loop_phase_two) (int \_\_round, const std::vector&lt; [**InterChiplet::ProcessConfig**](classInterChiplet_1_1ProcessConfig.md) &gt; & \_\_proc\_cfg\_list) <br> |
|  void \* | [**bridge\_thread**](#function-bridge_thread) (void \* \_\_args\_ptr) <br> |
|  int | [**main**](#function-main) (int argc, const char \* argv) <br> |
|  void | [**parse\_command**](#function-parse_command) (char \* \_\_pipe\_buf, [**ProcessStruct**](classProcessStruct.md) \* \_\_proc\_struct, int \_\_stdin\_fd) <br> |







## Macros

| Type | Name |
| ---: | :--- |
| define  | [**PIPE\_BUF\_SIZE**](interchiplet_8cpp.md#define-pipe_buf_size)  1024<br> |

## Public Functions Documentation


### function \_\_loop\_phase\_one 

```C++
InterChiplet::InnerTimeType __loop_phase_one (
    int __round,
    const std::vector< InterChiplet::ProcessConfig > & __proc_phase1_cfg_list,
    const std::vector< InterChiplet::ProcessConfig > & __proc_phase2_cfg_list
) 
```




### function \_\_loop\_phase\_two 

```C++
void __loop_phase_two (
    int __round,
    const std::vector< InterChiplet::ProcessConfig > & __proc_cfg_list
) 
```




### function bridge\_thread 

```C++
void * bridge_thread (
    void * __args_ptr
) 
```




### function main 

```C++
int main (
    int argc,
    const char * argv
) 
```




### function parse\_command 

```C++
void parse_command (
    char * __pipe_buf,
    ProcessStruct * __proc_struct,
    int __stdin_fd
) 
```



## Macro Definition Documentation



### define PIPE\_BUF\_SIZE 

```C++
#define PIPE_BUF_SIZE 1024
```




------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/srcs/interchiplet.cpp`