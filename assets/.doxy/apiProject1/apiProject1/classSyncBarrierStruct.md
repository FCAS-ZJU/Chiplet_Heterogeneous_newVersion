
# Class SyncBarrierStruct



[**ClassList**](annotated.md) **>** [**SyncBarrierStruct**](classSyncBarrierStruct.md)



_Structure for Barrier synchronization._ 

* `#include <cmd_handler.h>`















## Public Functions

| Type | Name |
| ---: | :--- |
|  [**SyncCmdList**](cmd__handler_8h.md#typedef-synccmdlist) & | [**barrierCmd**](#function-barriercmd) (int \_\_uid) <br> |
|  void | [**insertBarrier**](#function-insertbarrier-12) (int \_\_uid, int \_\_count) <br> |
|  void | [**insertBarrier**](#function-insertbarrier-22) (int \_\_uid, int \_\_count, const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br> |
|  bool | [**overflow**](#function-overflow) (int \_\_uid) <br> |
|  void | [**reset**](#function-reset) (int \_\_uid) <br> |








## Public Functions Documentation


### function barrierCmd 

```C++
inline SyncCmdList & SyncBarrierStruct::barrierCmd (
    int __uid
) 
```




### function insertBarrier [1/2]

```C++
inline void SyncBarrierStruct::insertBarrier (
    int __uid,
    int __count
) 
```




### function insertBarrier [2/2]

```C++
inline void SyncBarrierStruct::insertBarrier (
    int __uid,
    int __count,
    const InterChiplet::SyncCommand & __cmd
) 
```




### function overflow 

```C++
inline bool SyncBarrierStruct::overflow (
    int __uid
) 
```




### function reset 

```C++
inline void SyncBarrierStruct::reset (
    int __uid
) 
```




------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/cmd_handler.h`