
# File cmd\_handler.cpp



[**FileList**](files.md) **>** [**interchiplet**](dir_a2025b34133129e5724d121abe9a4a4a.md) **>** [**srcs**](dir_b94c70d771af9f161858c2c4e7b3d1c5.md) **>** [**cmd\_handler.cpp**](cmd__handler_8cpp.md)

[Go to the source code of this file.](cmd__handler_8cpp_source.md)



* `#include "cmd_handler.h"`
* `#include <sstream>`
* `#include <string>`
* `#include "spdlog/spdlog.h"`















## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**handle\_barrier\_cmd**](#function-handle_barrier_cmd) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd, [**SyncStruct**](classSyncStruct.md) \* \_\_sync\_struct) <br>_Handle BARRIER command._  |
|  void | [**handle\_barrier\_write\_cmd**](#function-handle_barrier_write_cmd) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd, [**SyncStruct**](classSyncStruct.md) \* \_\_sync\_struct) <br>_Handle WRITE command with barrier flag._  |
|  void | [**handle\_cycle\_cmd**](#function-handle_cycle_cmd) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd, [**SyncStruct**](classSyncStruct.md) \* \_\_sync\_struct) <br>_Handle CYCLE command._  |
|  void | [**handle\_launch\_cmd**](#function-handle_launch_cmd) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd, [**SyncStruct**](classSyncStruct.md) \* \_\_sync\_struct) <br>_Handle LAUNCH command._  |
|  void | [**handle\_lock\_cmd**](#function-handle_lock_cmd) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd, [**SyncStruct**](classSyncStruct.md) \* \_\_sync\_struct) <br>_Handle LOCK command._  |
|  void | [**handle\_lock\_write\_cmd**](#function-handle_lock_write_cmd) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd, [**SyncStruct**](classSyncStruct.md) \* \_\_sync\_struct) <br>_Handle WRITE command with LOCK flag._  |
|  void | [**handle\_pipe\_cmd**](#function-handle_pipe_cmd) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd, [**SyncStruct**](classSyncStruct.md) \* \_\_sync\_struct) <br>_Handle PIPE command._  |
|  void | [**handle\_read\_cmd**](#function-handle_read_cmd) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd, [**SyncStruct**](classSyncStruct.md) \* \_\_sync\_struct) <br>_Handle READ command._  |
|  void | [**handle\_unlock\_cmd**](#function-handle_unlock_cmd) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd, [**SyncStruct**](classSyncStruct.md) \* \_\_sync\_struct) <br>_Handle UNLOCK command._  |
|  void | [**handle\_unlock\_write\_cmd**](#function-handle_unlock_write_cmd) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd, [**SyncStruct**](classSyncStruct.md) \* \_\_sync\_struct) <br>_Handle WRITE command with UNLOCK flag._  |
|  void | [**handle\_waitlaunch\_cmd**](#function-handle_waitlaunch_cmd) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd, [**SyncStruct**](classSyncStruct.md) \* \_\_sync\_struct) <br>_Handle WAITLAUNCH command._  |
|  void | [**handle\_write\_cmd**](#function-handle_write_cmd) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd, [**SyncStruct**](classSyncStruct.md) \* \_\_sync\_struct) <br>_Handle WRITE command._  |








## Public Functions Documentation


### function handle\_barrier\_cmd 

_Handle BARRIER command._ 
```C++
void handle_barrier_cmd (
    const InterChiplet::SyncCommand & __cmd,
    SyncStruct * __sync_struct
) 
```





**Parameters:**


* `__cmd` Command to handle. 
* `__sync_struct` Pointer to global synchronize structure. 




        

### function handle\_barrier\_write\_cmd 

_Handle WRITE command with barrier flag._ 
```C++
void handle_barrier_write_cmd (
    const InterChiplet::SyncCommand & __cmd,
    SyncStruct * __sync_struct
) 
```





**Parameters:**


* `__cmd` Command to handle. 
* `__sync_struct` Pointer to global synchronize structure. 




        

### function handle\_cycle\_cmd 

_Handle CYCLE command._ 
```C++
void handle_cycle_cmd (
    const InterChiplet::SyncCommand & __cmd,
    SyncStruct * __sync_struct
) 
```





**Parameters:**


* `__cmd` Command to handle. 
* `__sync_struct` Pointer to global synchronize structure. 




        

### function handle\_launch\_cmd 

_Handle LAUNCH command._ 
```C++
void handle_launch_cmd (
    const InterChiplet::SyncCommand & __cmd,
    SyncStruct * __sync_struct
) 
```





**Parameters:**


* `__cmd` Command to handle. 
* `__sync_struct` Pointer to global synchronize structure. 




        

### function handle\_lock\_cmd 

_Handle LOCK command._ 
```C++
void handle_lock_cmd (
    const InterChiplet::SyncCommand & __cmd,
    SyncStruct * __sync_struct
) 
```





**Parameters:**


* `__cmd` Command to handle. 
* `__sync_struct` Pointer to global synchronize structure. 




        

### function handle\_lock\_write\_cmd 

_Handle WRITE command with LOCK flag._ 
```C++
void handle_lock_write_cmd (
    const InterChiplet::SyncCommand & __cmd,
    SyncStruct * __sync_struct
) 
```





**Parameters:**


* `__cmd` Command to handle. 
* `__sync_struct` Pointer to global synchronize structure. 




        

### function handle\_pipe\_cmd 

_Handle PIPE command._ 
```C++
void handle_pipe_cmd (
    const InterChiplet::SyncCommand & __cmd,
    SyncStruct * __sync_struct
) 
```





**Parameters:**


* `__cmd` Command to handle. 
* `__sync_struct` Pointer to global synchronize structure. 




        

### function handle\_read\_cmd 

_Handle READ command._ 
```C++
void handle_read_cmd (
    const InterChiplet::SyncCommand & __cmd,
    SyncStruct * __sync_struct
) 
```





**Parameters:**


* `__cmd` Command to handle. 
* `__sync_struct` Pointer to global synchronize structure. 




        

### function handle\_unlock\_cmd 

_Handle UNLOCK command._ 
```C++
void handle_unlock_cmd (
    const InterChiplet::SyncCommand & __cmd,
    SyncStruct * __sync_struct
) 
```





**Parameters:**


* `__cmd` Command to handle. 
* `__sync_struct` Pointer to global synchronize structure. 




        

### function handle\_unlock\_write\_cmd 

_Handle WRITE command with UNLOCK flag._ 
```C++
void handle_unlock_write_cmd (
    const InterChiplet::SyncCommand & __cmd,
    SyncStruct * __sync_struct
) 
```





**Parameters:**


* `__cmd` Command to handle. 
* `__sync_struct` Pointer to global synchronize structure. 




        

### function handle\_waitlaunch\_cmd 

_Handle WAITLAUNCH command._ 
```C++
void handle_waitlaunch_cmd (
    const InterChiplet::SyncCommand & __cmd,
    SyncStruct * __sync_struct
) 
```





**Parameters:**


* `__cmd` Command to handle. 
* `__sync_struct` Pointer to global synchronize structure. 




        

### function handle\_write\_cmd 

_Handle WRITE command._ 
```C++
void handle_write_cmd (
    const InterChiplet::SyncCommand & __cmd,
    SyncStruct * __sync_struct
) 
```





**Parameters:**


* `__cmd` Command to handle. 
* `__sync_struct` Pointer to global synchronize structure. 




        

------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/srcs/cmd_handler.cpp`