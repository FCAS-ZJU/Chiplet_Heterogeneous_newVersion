
# Class ProcessStruct



[**ClassList**](annotated.md) **>** [**ProcessStruct**](classProcessStruct.md)



_Data structure of process configuration._ 














## Public Attributes

| Type | Name |
| ---: | :--- |
|  std::vector&lt; std::string &gt; | [**m\_args**](#variable-m_args)  <br> |
|  double | [**m\_clock\_rate**](#variable-m_clock_rate)  <br> |
|  std::string | [**m\_command**](#variable-m_command)  <br> |
|  std::string | [**m\_log\_file**](#variable-m_log_file)  <br> |
|  int | [**m\_phase**](#variable-m_phase)  <br> |
|  int | [**m\_pid**](#variable-m_pid)  <br> |
|  int | [**m\_pid2**](#variable-m_pid2)  <br> |
|  std::string | [**m\_pre\_copy**](#variable-m_pre_copy)  <br> |
|  int | [**m\_round**](#variable-m_round)  <br> |
|  [**SyncStruct**](classSyncStruct.md) \* | [**m\_sync\_struct**](#variable-m_sync_struct)  <br>_Pointer to synchronize structure._  |
|  int | [**m\_thread**](#variable-m_thread)  <br> |
|  pthread\_t | [**m\_thread\_id**](#variable-m_thread_id)  <br> |
|  bool | [**m\_to\_stdout**](#variable-m_to_stdout)  <br> |
|  std::string | [**m\_unfinished\_line**](#variable-m_unfinished_line)  <br> |


## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**ProcessStruct**](#function-processstruct) (const [**InterChiplet::ProcessConfig**](classInterChiplet_1_1ProcessConfig.md) & \_\_config) <br> |








## Public Attributes Documentation


### variable m\_args 

```C++
std::vector<std::string> ProcessStruct::m_args;
```




### variable m\_clock\_rate 

```C++
double ProcessStruct::m_clock_rate;
```




### variable m\_command 

```C++
std::string ProcessStruct::m_command;
```




### variable m\_log\_file 

```C++
std::string ProcessStruct::m_log_file;
```




### variable m\_phase 

```C++
int ProcessStruct::m_phase;
```




### variable m\_pid 

```C++
int ProcessStruct::m_pid;
```




### variable m\_pid2 

```C++
int ProcessStruct::m_pid2;
```




### variable m\_pre\_copy 

```C++
std::string ProcessStruct::m_pre_copy;
```




### variable m\_round 

```C++
int ProcessStruct::m_round;
```




### variable m\_sync\_struct 

```C++
SyncStruct* ProcessStruct::m_sync_struct;
```




### variable m\_thread 

```C++
int ProcessStruct::m_thread;
```




### variable m\_thread\_id 

```C++
pthread_t ProcessStruct::m_thread_id;
```




### variable m\_to\_stdout 

```C++
bool ProcessStruct::m_to_stdout;
```




### variable m\_unfinished\_line 

```C++
std::string ProcessStruct::m_unfinished_line;
```



## Public Functions Documentation


### function ProcessStruct 

```C++
inline ProcessStruct::ProcessStruct (
    const InterChiplet::ProcessConfig & __config
) 
```




------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/srcs/interchiplet.cpp`