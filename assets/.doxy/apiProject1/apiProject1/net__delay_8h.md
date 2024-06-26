
# File net\_delay.h



[**FileList**](files.md) **>** [**includes**](dir_943fa6db2bfb09b7dcf1f02346dde40e.md) **>** [**net\_delay.h**](net__delay_8h.md)

[Go to the source code of this file.](net__delay_8h_source.md)



* `#include <fstream>`
* `#include <map>`
* `#include "global_define.h"`
* `#include "spdlog/spdlog.h"`










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
| define  | [**PAC\_PAYLOAD\_BIT**](net__delay_8h.md#define-pac_payload_bit)  512<br> |
| define  | [**PAC\_PAYLOAD\_BYTE**](net__delay_8h.md#define-pac_payload_byte)  (PAC\_PAYLOAD\_BIT / 8)<br> |
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




### define PAC\_PAYLOAD\_BIT 

```C++
#define PAC_PAYLOAD_BIT 512
```




### define PAC\_PAYLOAD\_BYTE 

```C++
#define PAC_PAYLOAD_BYTE (PAC_PAYLOAD_BIT / 8)
```




### define SRC\_DELAY 

```C++
#define SRC_DELAY (
    pair
) std::get<0>(pair)
```




------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/net_delay.h`