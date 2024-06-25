
# Class InterChiplet::NetworkDelayStruct



[**ClassList**](annotated.md) **>** [**InterChiplet**](namespaceInterChiplet.md) **>** [**NetworkDelayStruct**](classInterChiplet_1_1NetworkDelayStruct.md)



_List of network delay item._ 

* `#include <net_delay.h>`















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**NetworkDelayStruct**](#function-networkdelaystruct) () <br> |
|  bool | [**checkOrderOfCommand**](#function-checkorderofcommand) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br>_Check the order of write/read commands._  |
|  void | [**clearDelayInfo**](#function-cleardelayinfo) () <br>_Clear delay information._  |
|  AddrType | [**frontLaunchSrc**](#function-frontlaunchsrc) (const AddrType & \_\_dst) <br>_Return the source address of the first LAUNCH command to the specified destination address._  |
|  AddrType | [**frontLockSrc**](#function-frontlocksrc) (const AddrType & \_\_dst) <br>_Return the source address of the first LOCK command to the specified destination address._  |
|  InnerTimeType | [**getBarrierCycle**](#function-getbarriercycle) (const std::vector&lt; [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) &gt; & barrier\_items) <br> |
|  InnerTimeType | [**getDefaultEndCycle**](#function-getdefaultendcycle-12) (const [**SyncCommand**](classInterChiplet_1_1SyncCommand.md) & write\_cmd) <br>_Get end cycle of one communication if cannot find this communication from delay list._  |
|  CmdDelayPair | [**getDefaultEndCycle**](#function-getdefaultendcycle-22) (const [**SyncCommand**](classInterChiplet_1_1SyncCommand.md) & write\_cmd, const [**SyncCommand**](classInterChiplet_1_1SyncCommand.md) & read\_cmd) <br>_Get end cycle of one communication if cannot find this communication from delay list._  |
|  CmdDelayPair | [**getEndCycle**](#function-getendcycle) (const [**SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_write\_cmd, const [**SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_read\_cmd) <br>_Get end cycle of one communication._  |
|  bool | [**hasLaunch**](#function-haslaunch) (const AddrType & \_\_dst) <br>_Check whether there is LAUNCH command in order._  |
|  bool | [**hasLock**](#function-haslock) (const AddrType & \_\_dst) <br>_Check whether there is LOCK command in order._  |
|  void | [**loadDelayInfo**](#function-loaddelayinfo) (const std::string & \_\_file\_name, double \_\_clock\_rate) <br>_Load package delay list from specified file._  |
|  void | [**popLaunch**](#function-poplaunch) (const AddrType & \_\_dst) <br>_Pop the first LAUNCH command to the specified destination address._  |
|  void | [**popLock**](#function-poplock) (const AddrType & \_\_dst) <br>_Pop the first LOCK command to the specified destination address._  |
|  int | [**size**](#function-size) () const<br> |








## Public Functions Documentation


### function NetworkDelayStruct 

```C++
inline InterChiplet::NetworkDelayStruct::NetworkDelayStruct () 
```




### function checkOrderOfCommand 

_Check the order of write/read commands._ 
```C++
inline bool InterChiplet::NetworkDelayStruct::checkOrderOfCommand (
    const InterChiplet::SyncCommand & __cmd
) 
```





**Parameters:**


* `__cmd` Command to check. 



**Returns:**

If the order of command matches the order of delay infomation, return True. Otherwise return False. 





        

### function clearDelayInfo 

```C++
inline void InterChiplet::NetworkDelayStruct::clearDelayInfo () 
```




### function frontLaunchSrc 

_Return the source address of the first LAUNCH command to the specified destination address._ 
```C++
inline AddrType InterChiplet::NetworkDelayStruct::frontLaunchSrc (
    const AddrType & __dst
) 
```





**Parameters:**


* `__dst` Destination address. 



**Returns:**

The source address of the first LAUNCH command. 





        

### function frontLockSrc 

_Return the source address of the first LOCK command to the specified destination address._ 
```C++
inline AddrType InterChiplet::NetworkDelayStruct::frontLockSrc (
    const AddrType & __dst
) 
```





**Parameters:**


* `__dst` Destination address. 



**Returns:**

The source address of the first LOCK command. 





        

### function getBarrierCycle 

```C++
inline InnerTimeType InterChiplet::NetworkDelayStruct::getBarrierCycle (
    const std::vector< InterChiplet::SyncCommand > & barrier_items
) 
```




### function getDefaultEndCycle [1/2]

_Get end cycle of one communication if cannot find this communication from delay list._ 
```C++
inline InnerTimeType InterChiplet::NetworkDelayStruct::getDefaultEndCycle (
    const SyncCommand & write_cmd
) 
```





**Parameters:**


* `__write_cmd` Write command 



**Returns:**

End cycle of this communication, used to acknowledge SYNC command. 





        

### function getDefaultEndCycle [2/2]

_Get end cycle of one communication if cannot find this communication from delay list._ 
```C++
inline CmdDelayPair InterChiplet::NetworkDelayStruct::getDefaultEndCycle (
    const SyncCommand & write_cmd,
    const SyncCommand & read_cmd
) 
```





**Parameters:**


* `__write_cmd` Write command 
* `__read_cmd` Read command. 



**Returns:**

End cycle of this communication, used to acknowledge SYNC command. 





        

### function getEndCycle 

_Get end cycle of one communication._ 
```C++
inline CmdDelayPair InterChiplet::NetworkDelayStruct::getEndCycle (
    const SyncCommand & __write_cmd,
    const SyncCommand & __read_cmd
) 
```





**Parameters:**


* `__write_cmd` Write command 
* `__read_cmd` Read command. 



**Returns:**

End cycle of this communication, used to acknowledge SYNC command. 





        

### function hasLaunch 

_Check whether there is LAUNCH command in order._ 
```C++
inline bool InterChiplet::NetworkDelayStruct::hasLaunch (
    const AddrType & __dst
) 
```





**Parameters:**


* `__dst` Destination address. 



**Returns:**

If there is LAUNCH command to the specified destination address, return True. 





        

### function hasLock 

_Check whether there is LOCK command in order._ 
```C++
inline bool InterChiplet::NetworkDelayStruct::hasLock (
    const AddrType & __dst
) 
```





**Parameters:**


* `__dst` Destination address. 



**Returns:**

If there is LOCK command to the specified destination address, return True. 





        

### function loadDelayInfo 

_Load package delay list from specified file._ 
```C++
inline void InterChiplet::NetworkDelayStruct::loadDelayInfo (
    const std::string & __file_name,
    double __clock_rate
) 
```





**Parameters:**


* `__file_name` Path to benchmark file. 
* `__clock_rate` Clock ratio (Simulator clock/Interchiplet clock). 




        

### function popLaunch 

_Pop the first LAUNCH command to the specified destination address._ 
```C++
inline void InterChiplet::NetworkDelayStruct::popLaunch (
    const AddrType & __dst
) 
```





**Parameters:**


* `__dst` Destination address. 




        

### function popLock 

_Pop the first LOCK command to the specified destination address._ 
```C++
inline void InterChiplet::NetworkDelayStruct::popLock (
    const AddrType & __dst
) 
```





**Parameters:**


* `__dst` Destination address. 




        

### function size 

```C++
inline int InterChiplet::NetworkDelayStruct::size () const
```




------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/net_delay.h`