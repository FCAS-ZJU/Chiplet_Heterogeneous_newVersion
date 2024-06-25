
# Class InterChiplet::SyncCommand



[**ClassList**](annotated.md) **>** [**InterChiplet**](namespaceInterChiplet.md) **>** [**SyncCommand**](classInterChiplet_1_1SyncCommand.md)



_Structure of synchronization command._ 

* `#include <global_define.h>`













## Public Attributes

| Type | Name |
| ---: | :--- |
|  double | [**m\_clock\_rate**](#variable-m_clock_rate)  <br>_Cycle convert rate._  |
|  InnerTimeType | [**m\_cycle**](#variable-m_cycle)  <br>_Cycle to send/receive command._  |
|  long | [**m\_desc**](#variable-m_desc)  <br>_Descriptor of synchronization behavior._  |
|  AddrType | [**m\_dst**](#variable-m_dst)  <br>_Destiantion address in X-axis._  |
|  int | [**m\_nbytes**](#variable-m_nbytes)  <br>_Number of bytes to write._  |
|  std::vector&lt; std::string &gt; | [**m\_res\_list**](#variable-m_res_list)  <br>_List of result strings._  |
|  AddrType | [**m\_src**](#variable-m_src)  <br>_Source address._  |
|  int | [**m\_stdin\_fd**](#variable-m_stdin_fd)  <br>_File descriptor to write response of this command._  |
|  SyncCommType | [**m\_type**](#variable-m_type)  <br>_Type of synchronization command._  |










## Public Attributes Documentation


### variable m\_clock\_rate 

```C++
double InterChiplet::SyncCommand::m_clock_rate;
```




### variable m\_cycle 

```C++
InnerTimeType InterChiplet::SyncCommand::m_cycle;
```




### variable m\_desc 

```C++
long InterChiplet::SyncCommand::m_desc;
```




### variable m\_dst 

```C++
AddrType InterChiplet::SyncCommand::m_dst;
```




### variable m\_nbytes 

```C++
int InterChiplet::SyncCommand::m_nbytes;
```




### variable m\_res\_list 

```C++
std::vector<std::string> InterChiplet::SyncCommand::m_res_list;
```




### variable m\_src 

```C++
AddrType InterChiplet::SyncCommand::m_src;
```




### variable m\_stdin\_fd 

_File descriptor to write response of this command._ 
```C++
int InterChiplet::SyncCommand::m_stdin_fd;
```



For example, if one entity presents READ command, the SYNC command to response this READ command should to send to this file descriptor. 


        

### variable m\_type 

```C++
SyncCommType InterChiplet::SyncCommand::m_type;
```




------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/global_define.h`