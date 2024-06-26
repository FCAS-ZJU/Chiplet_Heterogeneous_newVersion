
# Class NetworkBenchList



[**ClassList**](annotated.md) **>** [**NetworkBenchList**](classNetworkBenchList.md)



_List of network benchmark item._ 

* `#include <net_bench.h>`



Inherits the following classes: std::multimap< InterChiplet::InnerTimeType, NetworkBenchItem >












## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**NetworkBenchList**](#function-networkbenchlist) () <br>_Construct_ [_**NetworkBenchList**_](classNetworkBenchList.md) _._ |
|  void | [**dumpBench**](#function-dumpbench) (const std::string & \_\_file\_name, double \_\_clock\_rate) <br>_Dump benchmark list to specified file._  |
|  void | [**insert**](#function-insert) (const [**NetworkBenchItem**](classNetworkBenchItem.md) & \_\_item) <br>_Insert item into list._  |








## Public Functions Documentation


### function NetworkBenchList 

```C++
inline NetworkBenchList::NetworkBenchList () 
```




### function dumpBench 

_Dump benchmark list to specified file._ 
```C++
inline void NetworkBenchList::dumpBench (
    const std::string & __file_name,
    double __clock_rate
) 
```





**Parameters:**


* `__file_name` Path to benchmark file. 
* `__clock_rate` Clock ratio (Simulator clock/Interchiplet clock). 




        

### function insert 

_Insert item into list._ 
```C++
inline void NetworkBenchList::insert (
    const NetworkBenchItem & __item
) 
```



Take the start cycle on source side as ordering key. 


        

------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/net_bench.h`