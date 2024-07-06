
# Class SyncCommStruct



[**ClassList**](annotated.md) **>** [**SyncCommStruct**](classSyncCommStruct.md)



_Structure for Communication synchronization._ 

* `#include <cmd_handler.h>`















## Public Functions

| Type | Name |
| ---: | :--- |
|  bool | [**hasMatchRead**](#function-hasmatchread) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br> |
|  bool | [**hasMatchWrite**](#function-hasmatchwrite) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br> |
|  void | [**insertRead**](#function-insertread-12) ([**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br> |
|  void | [**insertRead**](#function-insertread-22) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br> |
|  void | [**insertWrite**](#function-insertwrite-12) ([**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br> |
|  void | [**insertWrite**](#function-insertwrite-22) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br> |
|  [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) | [**popMatchRead**](#function-popmatchread) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br> |
|  [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) | [**popMatchWrite**](#function-popmatchwrite) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br> |








## Public Functions Documentation


### function hasMatchRead 

```C++
inline bool SyncCommStruct::hasMatchRead (
    const InterChiplet::SyncCommand & __cmd
) 
```




### function hasMatchWrite 

```C++
inline bool SyncCommStruct::hasMatchWrite (
    const InterChiplet::SyncCommand & __cmd
) 
```




### function insertRead [1/2]

```C++
inline void SyncCommStruct::insertRead (
    InterChiplet::SyncCommand & __cmd
) 
```




### function insertRead [2/2]

```C++
inline void SyncCommStruct::insertRead (
    const InterChiplet::SyncCommand & __cmd
) 
```




### function insertWrite [1/2]

```C++
inline void SyncCommStruct::insertWrite (
    InterChiplet::SyncCommand & __cmd
) 
```




### function insertWrite [2/2]

```C++
inline void SyncCommStruct::insertWrite (
    const InterChiplet::SyncCommand & __cmd
) 
```




### function popMatchRead 

```C++
inline InterChiplet::SyncCommand SyncCommStruct::popMatchRead (
    const InterChiplet::SyncCommand & __cmd
) 
```




### function popMatchWrite 

```C++
inline InterChiplet::SyncCommand SyncCommStruct::popMatchWrite (
    const InterChiplet::SyncCommand & __cmd
) 
```




------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/cmd_handler.h`