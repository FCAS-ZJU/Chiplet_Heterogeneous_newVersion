
# Class NetworkDelayMap



[**ClassList**](annotated.md) **>** [**NetworkDelayMap**](classNetworkDelayMap.md)



_Map for network delay information._ 

* `#include <net_delay.h>`



Inherits the following classes: std::map< InterChiplet::AddrType, NetworkDelayOrder >












## Public Functions

| Type | Name |
| ---: | :--- |
|  bool | [**checkOrderOfCommand**](#function-checkorderofcommand) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br>_Check the order of write/read commands._  |
|  [**NetworkDelayItem**](classNetworkDelayItem.md) | [**front**](#function-front-12) (const InterChiplet::AddrType & \_\_addr) <br>_Return the first delay information for the specified address._  |
|  [**NetworkDelayItem**](classNetworkDelayItem.md) | [**front**](#function-front-22) (const InterChiplet::AddrType & \_\_addr, const InterChiplet::AddrType & \_\_src, const InterChiplet::AddrType & \_\_dst) <br>_Return the first delay information for the specified address._  |
|  bool | [**hasAddr**](#function-hasaddr-12) (const InterChiplet::AddrType & \_\_addr) <br>_Check whether there is delay information for the specified address._  |
|  bool | [**hasAddr**](#function-hasaddr-22) (const InterChiplet::AddrType & \_\_addr, const InterChiplet::AddrType & \_\_src, const InterChiplet::AddrType & \_\_dst) <br>_Check whether there is delay information for the specified address._  |
|  void | [**insert**](#function-insert) (const InterChiplet::AddrType & \_\_addr, InterChiplet::InnerTimeType \_\_cycle, const [**NetworkDelayItem**](classNetworkDelayItem.md) & \_\_item) <br>_Insert delay information._  |
|  void | [**pop**](#function-pop-12) (const InterChiplet::AddrType & \_\_addr) <br>_Pop the first delay information for the specified address._  |
|  void | [**pop**](#function-pop-22) (const InterChiplet::AddrType & \_\_addr, const InterChiplet::AddrType & \_\_src, const InterChiplet::AddrType & \_\_dst) <br>_Pop the first delay information for the specified address._  |








## Public Functions Documentation


### function checkOrderOfCommand 

_Check the order of write/read commands._ 
```C++
inline bool NetworkDelayMap::checkOrderOfCommand (
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
inline NetworkDelayItem NetworkDelayMap::front (
    const InterChiplet::AddrType & __addr
) 
```





**Parameters:**


* `__addr` Address. 



**Returns:**

The first delay information to the specified address. 





        

### function front [2/2]

_Return the first delay information for the specified address._ 
```C++
inline NetworkDelayItem NetworkDelayMap::front (
    const InterChiplet::AddrType & __addr,
    const InterChiplet::AddrType & __src,
    const InterChiplet::AddrType & __dst
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
inline bool NetworkDelayMap::hasAddr (
    const InterChiplet::AddrType & __addr
) 
```





**Parameters:**


* `__addr` Address. 



**Returns:**

If there is delay information for the specified address, return True. 





        

### function hasAddr [2/2]

_Check whether there is delay information for the specified address._ 
```C++
inline bool NetworkDelayMap::hasAddr (
    const InterChiplet::AddrType & __addr,
    const InterChiplet::AddrType & __src,
    const InterChiplet::AddrType & __dst
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
inline void NetworkDelayMap::insert (
    const InterChiplet::AddrType & __addr,
    InterChiplet::InnerTimeType __cycle,
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
inline void NetworkDelayMap::pop (
    const InterChiplet::AddrType & __addr
) 
```





**Parameters:**


* `__addr` Asdress. 




        

### function pop [2/2]

_Pop the first delay information for the specified address._ 
```C++
inline void NetworkDelayMap::pop (
    const InterChiplet::AddrType & __addr,
    const InterChiplet::AddrType & __src,
    const InterChiplet::AddrType & __dst
) 
```





**Parameters:**


* `__addr` Address as key. 
* `__src` Source address. 
* `__dst` Destination address. 




        

------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/net_delay.h`