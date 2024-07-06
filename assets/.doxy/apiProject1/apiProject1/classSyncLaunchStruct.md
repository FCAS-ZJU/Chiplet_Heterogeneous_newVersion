
# Class SyncLaunchStruct



[**ClassList**](annotated.md) **>** [**SyncLaunchStruct**](classSyncLaunchStruct.md)



_Structure for Launch and Wait-launch synchronization._ 

* `#include <cmd_handler.h>`















## Public Functions

| Type | Name |
| ---: | :--- |
|  bool | [**hasMatchLaunch**](#function-hasmatchlaunch) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br> |
|  bool | [**hasMatchWaitlaunch**](#function-hasmatchwaitlaunch) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br> |
|  void | [**insertLaunch**](#function-insertlaunch-12) ([**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br> |
|  void | [**insertLaunch**](#function-insertlaunch-22) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br> |
|  void | [**insertWaitlaunch**](#function-insertwaitlaunch-12) ([**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br> |
|  void | [**insertWaitlaunch**](#function-insertwaitlaunch-22) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br> |
|  [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) | [**popMatchLaunch**](#function-popmatchlaunch) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br> |
|  [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) | [**popMatchWaitlaunch**](#function-popmatchwaitlaunch) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br> |








## Public Functions Documentation


### function hasMatchLaunch 

```C++
inline bool SyncLaunchStruct::hasMatchLaunch (
    const InterChiplet::SyncCommand & __cmd
) 
```




### function hasMatchWaitlaunch 

```C++
inline bool SyncLaunchStruct::hasMatchWaitlaunch (
    const InterChiplet::SyncCommand & __cmd
) 
```




### function insertLaunch [1/2]

```C++
inline void SyncLaunchStruct::insertLaunch (
    InterChiplet::SyncCommand & __cmd
) 
```




### function insertLaunch [2/2]

```C++
inline void SyncLaunchStruct::insertLaunch (
    const InterChiplet::SyncCommand & __cmd
) 
```




### function insertWaitlaunch [1/2]

```C++
inline void SyncLaunchStruct::insertWaitlaunch (
    InterChiplet::SyncCommand & __cmd
) 
```




### function insertWaitlaunch [2/2]

```C++
inline void SyncLaunchStruct::insertWaitlaunch (
    const InterChiplet::SyncCommand & __cmd
) 
```




### function popMatchLaunch 

```C++
inline InterChiplet::SyncCommand SyncLaunchStruct::popMatchLaunch (
    const InterChiplet::SyncCommand & __cmd
) 
```




### function popMatchWaitlaunch 

```C++
inline InterChiplet::SyncCommand SyncLaunchStruct::popMatchWaitlaunch (
    const InterChiplet::SyncCommand & __cmd
) 
```




------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/cmd_handler.h`