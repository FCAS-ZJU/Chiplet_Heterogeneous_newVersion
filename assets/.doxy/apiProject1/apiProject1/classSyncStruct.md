
# Class SyncStruct



[**ClassList**](annotated.md) **>** [**SyncStruct**](classSyncStruct.md)



_Data structure of synchronize operation._ 

* `#include <cmd_handler.h>`













## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**SyncBarrierStruct**](classSyncBarrierStruct.md) | [**m\_barrier\_struct**](#variable-m_barrier_struct)  <br>_Barrier behavior._  |
|  [**SyncBarrierStruct**](classSyncBarrierStruct.md) | [**m\_barrier\_timing\_struct**](#variable-m_barrier_timing_struct)  <br>_Barrier timing behavior._  |
|  [**NetworkBenchList**](classNetworkBenchList.md) | [**m\_bench\_list**](#variable-m_bench_list)  <br>_Benchmark list, recording the communication transactions have sent out._  |
|  [**SyncCommStruct**](classSyncCommStruct.md) | [**m\_comm\_struct**](#variable-m_comm_struct)  <br>_Communication behavior._  |
|  [**SyncClockStruct**](classSyncClockStruct.md) | [**m\_cycle\_struct**](#variable-m_cycle_struct)  <br>_Global simulation cycle, which is the largest notified cycle count._  |
|  [**NetworkDelayStruct**](classNetworkDelayStruct.md) | [**m\_delay\_list**](#variable-m_delay_list)  <br>_Delay list, recording the delay of each communication transactions._  |
|  [**SyncLaunchStruct**](classSyncLaunchStruct.md) | [**m\_launch\_struct**](#variable-m_launch_struct)  <br>_Launch behavior._  |
|  [**SyncLockStruct**](classSyncLockStruct.md) | [**m\_lock\_struct**](#variable-m_lock_struct)  <br>_Lock behavior._  |
|  [**SyncLockStruct**](classSyncLockStruct.md) | [**m\_lock\_timing\_struct**](#variable-m_lock_timing_struct)  <br>_Lock behavior._  |
|  pthread\_mutex\_t | [**m\_mutex**](#variable-m_mutex)  <br>_Mutex to access this structure._  |
|  [**SyncPipeStruct**](classSyncPipeStruct.md) | [**m\_pipe\_struct**](#variable-m_pipe_struct)  <br>_Pipe behavior._  |


## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**SyncStruct**](#function-syncstruct) () <br>_Construct synchronize stucture._  |
|   | [**~SyncStruct**](#function-syncstruct) () <br>_Destory synchronize structure._  |








## Public Attributes Documentation


### variable m\_barrier\_struct 

```C++
SyncBarrierStruct SyncStruct::m_barrier_struct;
```




### variable m\_barrier\_timing\_struct 

```C++
SyncBarrierStruct SyncStruct::m_barrier_timing_struct;
```




### variable m\_bench\_list 

```C++
NetworkBenchList SyncStruct::m_bench_list;
```




### variable m\_comm\_struct 

```C++
SyncCommStruct SyncStruct::m_comm_struct;
```




### variable m\_cycle\_struct 

```C++
SyncClockStruct SyncStruct::m_cycle_struct;
```




### variable m\_delay\_list 

```C++
NetworkDelayStruct SyncStruct::m_delay_list;
```




### variable m\_launch\_struct 

```C++
SyncLaunchStruct SyncStruct::m_launch_struct;
```




### variable m\_lock\_struct 

```C++
SyncLockStruct SyncStruct::m_lock_struct;
```




### variable m\_lock\_timing\_struct 

```C++
SyncLockStruct SyncStruct::m_lock_timing_struct;
```




### variable m\_mutex 

```C++
pthread_mutex_t SyncStruct::m_mutex;
```




### variable m\_pipe\_struct 

```C++
SyncPipeStruct SyncStruct::m_pipe_struct;
```



## Public Functions Documentation


### function SyncStruct 

_Construct synchronize stucture._ 
```C++
inline SyncStruct::SyncStruct () 
```



Initializete mutex. 


        

### function ~SyncStruct 

_Destory synchronize structure._ 
```C++
inline SyncStruct::~SyncStruct () 
```



Destory mutex. 


        

------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/cmd_handler.h`