
# Class InterChiplet::NetworkDelayMap



[**ClassList**](annotated.md) **>** [**InterChiplet**](namespaceInterChiplet.md) **>** [**NetworkDelayMap**](classInterChiplet_1_1NetworkDelayMap.md)



_Map for network delay information._ 

* `#include <net_delay.h>`



Inherits the following classes: std::map< AddrType, NetworkDelayOrder >












## Public Functions

| Type | Name |
| ---: | :--- |
|  bool | [**checkOrderOfCommand**](#function-checkorderofcommand) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br>_Check the order of write/read commands._  |
|  [**NetworkDelayItem**](classInterChiplet_1_1NetworkDelayItem.md) | [**front**](#function-front-12) (const AddrType & \_\_addr) <br>_Return the first delay information for the specified address._  |
|  [**NetworkDelayItem**](classInterChiplet_1_1NetworkDelayItem.md) | [**front**](#function-front-22) (const AddrType & \_\_addr, const AddrType & \_\_src, const AddrType & \_\_dst) <br>_Return the first delay information for the specified address._  |
|  bool | [**hasAddr**](#function-hasaddr-12) (const AddrType & \_\_addr) <br>_Check whether there is delay information for the specified address._  |
|  bool | [**hasAddr**](#function-hasaddr-22) (const AddrType & \_\_addr, const AddrType & \_\_src, const AddrType & \_\_dst) <br>_Check whether there is delay information for the specified address._  |
|  void | [**insert**](#function-insert) (const AddrType & \_\_addr, InnerTimeType \_\_cycle, const [**NetworkDelayItem**](classInterChiplet_1_1NetworkDelayItem.md) & \_\_item) <br>_Insert delay information._  |
|  void | [**pop**](#function-pop-12) (const AddrType & \_\_addr) <br>_Pop the first delay information for the specified address._  |
|  void | [**pop**](#function-pop-22) (const AddrType & \_\_addr, const AddrType & \_\_src, const AddrType & \_\_dst) <br>_Pop the first delay information for the specified address._  |








## Public Functions Documentation


### function checkOrderOfCommand 

_Check the order of write/read commands._ 
```C++
inline bool InterChiplet::NetworkDelayMap::checkOrderOfCommand (
    const InterChiplet::SyncCommand & __cmd
) 
```





**Parameters:**


* `__cmd` Command to check. 



**Returns:**

If the order of command matches the order of delay infomation, return True. Otherwise return False. 





        

### function front [1/2]

_Return the first delay information for the specified address._ 
```C++
inline NetworkDelayItem InterChiplet::NetworkDelayMap::front (
    const AddrType & __addr
) 
```





**Parameters:**


* `__addr` Address. 



**Returns:**

The first delay information to the specified address. 





        

### function front [2/2]

_Return the first delay information for the specified address._ 
```C++
inline NetworkDelayItem InterChiplet::NetworkDelayMap::front (
    const AddrType & __addr,
    const AddrType & __src,
    const AddrType & __dst
) 
```





**Parameters:**


* `__addr` Address as key. 
* `__src` Source address. 
* `__dst` Destination address. 



**Returns:**

The first delay information to the specified address. 





        

### function hasAddr [1/2]

_Check whether there is delay information for the specified address._ 
```C++
inline bool InterChiplet::NetworkDelayMap::hasAddr (
    const AddrType & __addr
) 
```





**Parameters:**


* `__addr` Address. 



**Returns:**

If there is delay information for the specified address, return True. 





        

### function hasAddr [2/2]

_Check whether there is delay information for the specified address._ 
```C++
inline bool InterChiplet::NetworkDelayMap::hasAddr (
    const AddrType & __addr,
    const AddrType & __src,
    const AddrType & __dst
) 
```





**Parameters:**


* `__addr` Address as key. 
* `__src` Source address. 
* `__dst` Destination address. 



**Returns:**

If there is delay information for the specified address, return True. 





        

### function insert 

_Insert delay information._ 
```C++
inline void InterChiplet::NetworkDelayMap::insert (
    const AddrType & __addr,
    InnerTimeType __cycle,
    const NetworkDelayItem & __item
) 
```





**Parameters:**


* `__addr` Address as key. 
* `__cycle` Event cycle. 
* `__item` Delay information structure. 




        

### function pop [1/2]

_Pop the first delay information for the specified address._ 
```C++
inline void InterChiplet::NetworkDelayMap::pop (
    const AddrType & __addr
) 
```





**Parameters:**


* `__addr` Asdress. 




        

### function pop [2/2]

_Pop the first delay information for the specified address._ 
```C++
inline void InterChiplet::NetworkDelayMap::pop (
    const AddrType & __addr,
    const AddrType & __src,
    const AddrType & __dst
) 
```





**Parameters:**


* `__addr` Address as key. 
* `__src` Source address. 
* `__dst` Destination address. 




        

------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/net_delay.h`