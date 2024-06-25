
# File global\_define.h



[**FileList**](files.md) **>** [**includes**](dir_943fa6db2bfb09b7dcf1f02346dde40e.md) **>** [**global\_define.h**](global__define_8h.md)

[Go to the source code of this file.](global__define_8h_source.md)



* `#include <cstdint>`
* `#include <string>`
* `#include <vector>`









## Namespaces

| Type | Name |
| ---: | :--- |
| namespace | [**InterChiplet**](namespaceInterChiplet.md) <br> |

## Classes

| Type | Name |
| ---: | :--- |
| class | [**SyncCommand**](classInterChiplet_1_1SyncCommand.md) <br>_Structure of synchronization command._  |












## Macros

| Type | Name |
| ---: | :--- |
| define  | [**DIM\_X**](global__define_8h.md#define-dim_x) (addr) (addr[0])<br> |
| define  | [**DIM\_Y**](global__define_8h.md#define-dim_y) (addr) (addr[1])<br> |
| define  | [**UNSPECIFIED\_ADDR**](global__define_8h.md#define-unspecified_addr) (addr) ((addr[0]) &lt; 0 && (addr[1]) &lt; 0)<br> |

## Macro Definition Documentation



### define DIM\_X 

```C++
#define DIM_X (
    addr
) (addr[0])
```




### define DIM\_Y 

```C++
#define DIM_Y (
    addr
) (addr[1])
```




### define UNSPECIFIED\_ADDR 

```C++
#define UNSPECIFIED_ADDR (
    addr
) ((addr[0]) < 0 && (addr[1]) < 0)
```




------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/global_define.h`