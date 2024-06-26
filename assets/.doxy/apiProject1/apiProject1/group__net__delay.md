
# Group net\_delay



[**Modules**](modules.md) **>** [**net\_delay**](group__net__delay.md)



_Network latency information interface._ 











## Classes

| Type | Name |
| ---: | :--- |
| class | [**NetworkDelayItem**](classNetworkDelayItem.md) <br>_Structure presents delay of one package in network._  |
| class | [**NetworkDelayMap**](classNetworkDelayMap.md) <br>_Map for network delay information._  |
| class | [**NetworkDelayStruct**](classNetworkDelayStruct.md) <br>_List of network delay item._  |

## Public Types

| Type | Name |
| ---: | :--- |
| typedef std::tuple&lt; InterChiplet::InnerTimeType, InterChiplet::InnerTimeType &gt; | [**CmdDelayPair**](#typedef-cmddelaypair)  <br> |
| typedef std::multimap&lt; InterChiplet::InnerTimeType, [**NetworkDelayItem**](classNetworkDelayItem.md) &gt; | [**NetworkDelayOrder**](#typedef-networkdelayorder)  <br> |











## Macros

| Type | Name |
| ---: | :--- |
| define  | [**DST\_DELAY**](net__delay_8h.md#define-dst_delay) (pair) std::get&lt;1&gt;(pair)<br> |
| define  | [**SRC\_DELAY**](net__delay_8h.md#define-src_delay) (pair) std::get&lt;0&gt;(pair)<br> |

## Public Types Documentation


### typedef CmdDelayPair 

```C++
typedef std::tuple<InterChiplet::InnerTimeType, InterChiplet::InnerTimeType> CmdDelayPair;
```




### typedef NetworkDelayOrder 

```C++
typedef std::multimap<InterChiplet::InnerTimeType, NetworkDelayItem> NetworkDelayOrder;
```



## Macro Definition Documentation



### define DST\_DELAY 

```C++
#define DST_DELAY (
    pair
) std::get<1>(pair)
```




### define SRC\_DELAY 

```C++
#define SRC_DELAY (
    pair
) std::get<0>(pair)
```




------------------------------
