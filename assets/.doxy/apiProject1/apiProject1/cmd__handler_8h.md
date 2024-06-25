
# File cmd\_handler.h



[**FileList**](files.md) **>** [**includes**](dir_943fa6db2bfb09b7dcf1f02346dde40e.md) **>** [**cmd\_handler.h**](cmd__handler_8h.md)

[Go to the source code of this file.](cmd__handler_8h_source.md)



* `#include <list>`
* `#include <map>`
* `#include <set>`
* `#include <string>`
* `#include <vector>`
* `#include "net_bench.h"`
* `#include "net_delay.h"`
* `#include "sync_protocol.h"`









## Namespaces

| Type | Name |
| ---: | :--- |
| namespace | [**InterChiplet**](namespaceInterChiplet.md) <br> |

## Classes

| Type | Name |
| ---: | :--- |
| class | [**SyncBarrierStruct**](classInterChiplet_1_1SyncBarrierStruct.md) <br>_Structure for Barrier synchronization._  |
| class | [**SyncClockStruct**](classInterChiplet_1_1SyncClockStruct.md) <br>_Structure for Clock synchronization._  |
| class | [**SyncCommStruct**](classInterChiplet_1_1SyncCommStruct.md) <br>_Structure for Communication synchronization._  |
| class | [**SyncLaunchStruct**](classInterChiplet_1_1SyncLaunchStruct.md) <br>_Structure for Launch and Wait-launch synchronization._  |
| class | [**SyncLockStruct**](classInterChiplet_1_1SyncLockStruct.md) <br>_Structure for Lock and Unlock synchronization._  |
| class | [**SyncPipeStruct**](classInterChiplet_1_1SyncPipeStruct.md) <br>_Structure for Pipe synchronization._  |
| class | [**SyncStruct**](classSyncStruct.md) <br>_Data structure of synchronize operation._  |





## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**handle\_barrier\_cmd**](#function-handle_barrier_cmd) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd, [**SyncStruct**](classSyncStruct.md) \* \_\_sync\_struct) <br>_Handle BARRIER command._  |
|  void | [**handle\_cycle\_cmd**](#function-handle_cycle_cmd) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd, [**SyncStruct**](classSyncStruct.md) \* \_\_sync\_struct) <br>_Handle CYCLE command._  |
|  void | [**handle\_launch\_cmd**](#function-handle_launch_cmd) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd, [**SyncStruct**](classSyncStruct.md) \* \_\_sync\_struct) <br>_Handle LAUNCH command._  |
|  void | [**handle\_lock\_cmd**](#function-handle_lock_cmd) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd, [**SyncStruct**](classSyncStruct.md) \* \_\_sync\_struct) <br>_Handle LOCK command._  |
|  void | [**handle\_pipe\_cmd**](#function-handle_pipe_cmd) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd, [**SyncStruct**](classSyncStruct.md) \* \_\_sync\_struct) <br>_Handle PIPE command._  |
|  void | [**handle\_read\_cmd**](#function-handle_read_cmd) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd, [**SyncStruct**](classSyncStruct.md) \* \_\_sync\_struct) <br>_Handle READ command._  |
|  void | [**handle\_unlock\_cmd**](#function-handle_unlock_cmd) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd, [**SyncStruct**](classSyncStruct.md) \* \_\_sync\_struct) <br>_Handle UNLOCK command._  |
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
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/cmd_handler.h`