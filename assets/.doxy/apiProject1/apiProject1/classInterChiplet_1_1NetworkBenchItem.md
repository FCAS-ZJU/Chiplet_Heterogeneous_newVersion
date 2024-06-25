
# Class InterChiplet::NetworkBenchItem



[**ClassList**](annotated.md) **>** [**InterChiplet**](namespaceInterChiplet.md) **>** [**NetworkBenchItem**](classInterChiplet_1_1NetworkBenchItem.md)



_Structure of one package in network._ 

* `#include <net_bench.h>`













## Public Attributes

| Type | Name |
| ---: | :--- |
|  long | [**m\_desc**](#variable-m_desc)  <br>_Synchronization protocol descriptor._  |
|  AddrType | [**m\_dst**](#variable-m_dst)  <br>_Destination address._  |
|  InnerTimeType | [**m\_dst\_cycle**](#variable-m_dst_cycle)  <br>_Package injection cycle from the destination side._  |
|  uint64\_t | [**m\_id**](#variable-m_id)  <br>_Packate id. (Not used yet.)_  |
|  int | [**m\_pac\_size**](#variable-m_pac_size)  <br>_Size of package in bytes._  |
|  AddrType | [**m\_src**](#variable-m_src)  <br>_Source address._  |
|  InnerTimeType | [**m\_src\_cycle**](#variable-m_src_cycle)  <br>_Package injection cycle from the source side._  |


## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**NetworkBenchItem**](#function-networkbenchitem-13) () <br>_Construct Empty_ [_**NetworkBenchItem**_](classInterChiplet_1_1NetworkBenchItem.md) _._ |
|   | [**NetworkBenchItem**](#function-networkbenchitem-23) (const [**SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_src\_cmd, const [**SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_dst\_cmd) <br>_Construct_ [_**NetworkBenchItem**_](classInterChiplet_1_1NetworkBenchItem.md) _from_[_**SyncCommand**_](classInterChiplet_1_1SyncCommand.md) _._ |
|   | [**NetworkBenchItem**](#function-networkbenchitem-33) (const [**SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_src\_cmd) <br>_Construct_ [_**NetworkBenchItem**_](classInterChiplet_1_1NetworkBenchItem.md) _from_[_**SyncCommand**_](classInterChiplet_1_1SyncCommand.md) _._ |








## Public Attributes Documentation


### variable m\_desc 

```C++
long InterChiplet::NetworkBenchItem::m_desc;
```




### variable m\_dst 

```C++
AddrType InterChiplet::NetworkBenchItem::m_dst;
```




### variable m\_dst\_cycle 

```C++
InnerTimeType InterChiplet::NetworkBenchItem::m_dst_cycle;
```




### variable m\_id 

```C++
uint64_t InterChiplet::NetworkBenchItem::m_id;
```




### variable m\_pac\_size 

```C++
int InterChiplet::NetworkBenchItem::m_pac_size;
```




### variable m\_src 

```C++
AddrType InterChiplet::NetworkBenchItem::m_src;
```




### variable m\_src\_cycle 

```C++
InnerTimeType InterChiplet::NetworkBenchItem::m_src_cycle;
```



## Public Functions Documentation


### function NetworkBenchItem [1/3]

```C++
inline InterChiplet::NetworkBenchItem::NetworkBenchItem () 
```




### function NetworkBenchItem [2/3]

_Construct_ [_**NetworkBenchItem**_](classInterChiplet_1_1NetworkBenchItem.md) _from_[_**SyncCommand**_](classInterChiplet_1_1SyncCommand.md) _._
```C++
inline InterChiplet::NetworkBenchItem::NetworkBenchItem (
    const SyncCommand & __src_cmd,
    const SyncCommand & __dst_cmd
) 
```





**Parameters:**


* `__src_cmd` Structure of source command. 
* `__dst_cmd` Structure of destination command. 




        

### function NetworkBenchItem [3/3]

_Construct_ [_**NetworkBenchItem**_](classInterChiplet_1_1NetworkBenchItem.md) _from_[_**SyncCommand**_](classInterChiplet_1_1SyncCommand.md) _._
```C++
inline InterChiplet::NetworkBenchItem::NetworkBenchItem (
    const SyncCommand & __src_cmd
) 
```





**Parameters:**


* `__src_cmd` Structure of source command. 




        ## Friends Documentation



### friend operator&lt;&lt; 

_Overloading operator &lt;&lt;._ 
```C++
inline friend std::ostream & InterChiplet::NetworkBenchItem::operator<< (
    std::ostream & os,
    const NetworkBenchItem & __item
) 
```



Write [**NetworkBenchItem**](classInterChiplet_1_1NetworkBenchItem.md) to output stream. 


        

### friend operator&gt;&gt; 

_Overloading operator &gt;&gt;._ 
```C++
inline friend std::istream & InterChiplet::NetworkBenchItem::operator>> (
    std::istream & os,
    NetworkBenchItem & __item
) 
```



Read [**NetworkBenchItem**](classInterChiplet_1_1NetworkBenchItem.md) from input stream. 


        

------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/net_bench.h`