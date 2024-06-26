
# Class NetworkBenchItem



[**ClassList**](annotated.md) **>** [**NetworkBenchItem**](classNetworkBenchItem.md)



_Structure of one package in network._ 

* `#include <net_bench.h>`













## Public Attributes

| Type | Name |
| ---: | :--- |
|  long | [**m\_desc**](#variable-m_desc)  <br>_Synchronization protocol descriptor._  |
|  InterChiplet::AddrType | [**m\_dst**](#variable-m_dst)  <br>_Destination address._  |
|  InterChiplet::InnerTimeType | [**m\_dst\_cycle**](#variable-m_dst_cycle)  <br>_Package injection cycle from the destination side._  |
|  uint64\_t | [**m\_id**](#variable-m_id)  <br>_Packate id. (Not used yet.)_  |
|  int | [**m\_pac\_size**](#variable-m_pac_size)  <br>_Size of package in bytes._  |
|  InterChiplet::AddrType | [**m\_src**](#variable-m_src)  <br>_Source address._  |
|  InterChiplet::InnerTimeType | [**m\_src\_cycle**](#variable-m_src_cycle)  <br>_Package injection cycle from the source side._  |


## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**NetworkBenchItem**](#function-networkbenchitem-13) () <br>_Construct Empty_ [_**NetworkBenchItem**_](classNetworkBenchItem.md) _._ |
|   | [**NetworkBenchItem**](#function-networkbenchitem-23) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_src\_cmd, const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_dst\_cmd) <br>_Construct_ [_**NetworkBenchItem**_](classNetworkBenchItem.md) _from SyncCommand._ |
|   | [**NetworkBenchItem**](#function-networkbenchitem-33) (const [**InterChiplet::SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_src\_cmd) <br>_Construct_ [_**NetworkBenchItem**_](classNetworkBenchItem.md) _from SyncCommand._ |








## Public Attributes Documentation


### variable m\_desc 

```C++
long NetworkBenchItem::m_desc;
```




### variable m\_dst 

```C++
InterChiplet::AddrType NetworkBenchItem::m_dst;
```




### variable m\_dst\_cycle 

```C++
InterChiplet::InnerTimeType NetworkBenchItem::m_dst_cycle;
```




### variable m\_id 

```C++
uint64_t NetworkBenchItem::m_id;
```




### variable m\_pac\_size 

```C++
int NetworkBenchItem::m_pac_size;
```




### variable m\_src 

```C++
InterChiplet::AddrType NetworkBenchItem::m_src;
```




### variable m\_src\_cycle 

```C++
InterChiplet::InnerTimeType NetworkBenchItem::m_src_cycle;
```



## Public Functions Documentation


### function NetworkBenchItem [1/3]

```C++
inline NetworkBenchItem::NetworkBenchItem () 
```




### function NetworkBenchItem [2/3]

_Construct_ [_**NetworkBenchItem**_](classNetworkBenchItem.md) _from SyncCommand._
```C++
inline NetworkBenchItem::NetworkBenchItem (
    const InterChiplet::SyncCommand & __src_cmd,
    const InterChiplet::SyncCommand & __dst_cmd
) 
```





**Parameters:**


* `__src_cmd` Structure of source command. 
* `__dst_cmd` Structure of destination command. 




        

### function NetworkBenchItem [3/3]

_Construct_ [_**NetworkBenchItem**_](classNetworkBenchItem.md) _from SyncCommand._
```C++
inline NetworkBenchItem::NetworkBenchItem (
    const InterChiplet::SyncCommand & __src_cmd
) 
```





**Parameters:**


* `__src_cmd` Structure of source command. 




        ## Friends Documentation



### friend operator&lt;&lt; 

_Overloading operator &lt;&lt;._ 
```C++
inline friend std::ostream & NetworkBenchItem::operator<< (
    std::ostream & os,
    const NetworkBenchItem & __item
) 
```



Write [**NetworkBenchItem**](classNetworkBenchItem.md) to output stream. 


        

### friend operator&gt;&gt; 

_Overloading operator &gt;&gt;._ 
```C++
inline friend std::istream & NetworkBenchItem::operator>> (
    std::istream & os,
    NetworkBenchItem & __item
) 
```



Read [**NetworkBenchItem**](classNetworkBenchItem.md) from input stream. 


        

------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/net_bench.h`