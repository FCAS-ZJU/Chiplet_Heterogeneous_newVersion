
# Class InterChiplet::PipeCommUnit



[**ClassList**](annotated.md) **>** [**InterChiplet**](namespaceInterChiplet.md) **>** [**PipeCommUnit**](classInterChiplet_1_1PipeCommUnit.md)



_Structure for Single Pipe communication._ 

* `#include <pipe_comm.h>`















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**PipeCommUnit**](#function-pipecommunit) (const char \* file\_name, bool read) <br> |
|  int | [**read\_data**](#function-read_data) (void \* dst\_buf, int nbyte) <br> |
|  bool | [**valid**](#function-valid) () const<br> |
|  int | [**write\_data**](#function-write_data) (void \* src\_buf, int nbyte) <br> |








## Public Functions Documentation


### function PipeCommUnit 

```C++
inline InterChiplet::PipeCommUnit::PipeCommUnit (
    const char * file_name,
    bool read
) 
```




### function read\_data 

```C++
inline int InterChiplet::PipeCommUnit::read_data (
    void * dst_buf,
    int nbyte
) 
```




### function valid 

```C++
inline bool InterChiplet::PipeCommUnit::valid () const
```




### function write\_data 

```C++
inline int InterChiplet::PipeCommUnit::write_data (
    void * src_buf,
    int nbyte
) 
```




------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/pipe_comm.h`