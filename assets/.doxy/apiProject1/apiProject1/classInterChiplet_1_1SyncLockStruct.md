
# Class InterChiplet::SyncLockStruct



[**ClassList**](annotated.md) **>** [**InterChiplet**](namespaceInterChiplet.md) **>** [**SyncLockStruct**](classInterChiplet_1_1SyncLockStruct.md)



_Structure for Lock and Unlock synchronization._ 

* `#include <cmd_handler.h>`















## Public Functions

| Type | Name |
| ---: | :--- |
|  [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) | [**getLastCmd**](#function-getlastcmd) (int \_\_uid) <br> |
|  bool | [**hasLastCmd**](#function-haslastcmd) (int \_\_uid) <br> |
|  bool | [**hasLockCmd**](#function-haslockcmd-12) (int \_\_uid) <br> |
|  bool | [**hasLockCmd**](#function-haslockcmd-22) (int \_\_uid, const AddrType & \_\_src) <br> |
|  void | [**insertLockCmd**](#function-insertlockcmd) (int \_\_uid, const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br> |
|  bool | [**isLocked**](#function-islocked) (int \_\_uid) <br> |
|  void | [**lock**](#function-lock) (int \_\_uid, const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br> |
|  [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) | [**popLockCmd**](#function-poplockcmd-12) (int \_\_uid) <br> |
|  [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) | [**popLockCmd**](#function-poplockcmd-22) (int \_\_uid, const AddrType & \_\_src) <br> |
|  void | [**unlock**](#function-unlock) (int \_\_uid, const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br> |








## Public Functions Documentation


### function getLastCmd 

```C++
inline InterChiplet::SyncCommand InterChiplet::SyncLockStruct::getLastCmd (
    int __uid
) 
```




### function hasLastCmd 

```C++
inline bool InterChiplet::SyncLockStruct::hasLastCmd (
    int __uid
) 
```




### function hasLockCmd [1/2]

```C++
inline bool InterChiplet::SyncLockStruct::hasLockCmd (
    int __uid
) 
```




### function hasLockCmd [2/2]

```C++
inline bool InterChiplet::SyncLockStruct::hasLockCmd (
    int __uid,
    const AddrType & __src
) 
```




### function insertLockCmd 

```C++
inline void InterChiplet::SyncLockStruct::insertLockCmd (
    int __uid,
    const InterChiplet::SyncCommand & __cmd
) 
```




### function isLocked 

```C++
inline bool InterChiplet::SyncLockStruct::isLocked (
    int __uid
) 
```




### function lock 

```C++
inline void InterChiplet::SyncLockStruct::lock (
    int __uid,
    const InterChiplet::SyncCommand & __cmd
) 
```




### function popLockCmd [1/2]

```C++
inline InterChiplet::SyncCommand InterChiplet::SyncLockStruct::popLockCmd (
    int __uid
) 
```




### function popLockCmd [2/2]

```C++
inline InterChiplet::SyncCommand InterChiplet::SyncLockStruct::popLockCmd (
    int __uid,
    const AddrType & __src
) 
```




### function unlock 

```C++
inline void InterChiplet::SyncLockStruct::unlock (
    int __uid,
    const InterChiplet::SyncCommand & __cmd
) 
```




------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/cmd_handler.h`