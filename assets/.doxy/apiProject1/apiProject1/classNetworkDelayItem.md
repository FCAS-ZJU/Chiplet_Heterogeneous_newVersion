
# Class NetworkDelayItem



[**ClassList**](annotated.md) **>** [**NetworkDelayItem**](classNetworkDelayItem.md)



_Structure presents delay of one package in network._ 

* `#include <net_delay.h>`













## Public Attributes

| Type | Name |
| ---: | :--- |
|  InterChiplet::InnerTimeType | [**m\_cycle**](#variable-m_cycle)  <br>_Package injection cycle. Used to order packages._  |
|  std::vector&lt; InterChiplet::InnerTimeType &gt; | [**m\_delay\_list**](#variable-m_delay_list)  <br>_Delay of packages._  |
|  long | [**m\_desc**](#variable-m_desc)  <br>_Synchronization protocol descriptor._  |
|  InterChiplet::AddrType | [**m\_dst**](#variable-m_dst)  <br>_Destination address._  |
|  uint64\_t | [**m\_id**](#variable-m_id)  <br>_Packate id. (Not used yet.)_  |
|  InterChiplet::AddrType | [**m\_src**](#variable-m_src)  <br>_Source address._  |


## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**NetworkDelayItem**](#function-networkdelayitem-12) () <br>_Construct Empty_ [_**NetworkDelayItem**_](classNetworkDelayItem.md) _._ |
|   | [**NetworkDelayItem**](#function-networkdelayitem-22) (InterChiplet::InnerTimeType \_\_cycle, const InterChiplet::AddrType & \_\_src, const InterChiplet::AddrType & \_\_dst, long \_\_desc, const std::vector&lt; InterChiplet::InnerTimeType &gt; & \_\_delay\_list) <br>_Construct_ [_**NetworkDelayItem**_](classNetworkDelayItem.md) _._ |








## Public Attributes Documentation


### variable m\_cycle 

```C++
InterChiplet::InnerTimeType NetworkDelayItem::m_cycle;
```




### variable m\_delay\_list 

_Delay of packages._ 
```C++
std::vector<InterChiplet::InnerTimeType> NetworkDelayItem::m_delay_list;
```



Each package has two delay values. The first value is the delay from the write side, and the second value is the delay from the read side. 


        

### variable m\_desc 

```C++
long NetworkDelayItem::m_desc;
```




### variable m\_dst 

```C++
InterChiplet::AddrType NetworkDelayItem::m_dst;
```




### variable m\_id 

```C++
uint64_t NetworkDelayItem::m_id;
```




### variable m\_src 

```C++
InterChiplet::AddrType NetworkDelayItem::m_src;
```



## Public Functions Documentation


### function NetworkDelayItem [1/2]

```C++
inline NetworkDelayItem::NetworkDelayItem () 
```




### function NetworkDelayItem [2/2]

_Construct_ [_**NetworkDelayItem**_](classNetworkDelayItem.md) _._
```C++
inline NetworkDelayItem::NetworkDelayItem (
    InterChiplet::InnerTimeType __cycle,
    const InterChiplet::AddrType & __src,
    const InterChiplet::AddrType & __dst,
    long __desc,
    const std::vector< InterChiplet::InnerTimeType > & __delay_list
) 
```





**Parameters:**


* `__cycle` Package injection cycle. 
* `__src` Source address. 
* `__dst` Destination address. 
* `__desc` Synchronization protocol descriptor. 
* `__delay_list` List of package delays. 




        ## Friends Documentation



### friend operator&lt;&lt; 

_Overloading operator &lt;&lt;._ 
```C++
inline friend std::ostream & NetworkDelayItem::operator<< (
    std::ostream & os,
    const NetworkDelayItem & __item
) 
```



Write [**NetworkDelayItem**](classNetworkDelayItem.md) to output stream. 


        

### friend operator&gt;&gt; 

_Overloading operator &gt;&gt;._ 
```C++
inline friend std::istream & NetworkDelayItem::operator>> (
    std::istream & os,
    NetworkDelayItem & __item
) 
```



Read [**NetworkDelayItem**](classNetworkDelayItem.md) from input stream. 


        

------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/net_delay.h`