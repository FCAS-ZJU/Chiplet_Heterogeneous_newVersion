
# Class InterChiplet::ProcessConfig



[**ClassList**](annotated.md) **>** [**InterChiplet**](namespaceInterChiplet.md) **>** [**ProcessConfig**](classInterChiplet_1_1ProcessConfig.md)



_Data structure to configure one simulation process._ 

* `#include <benchmark_yaml.h>`













## Public Attributes

| Type | Name |
| ---: | :--- |
|  std::vector&lt; std::string &gt; | [**m\_args**](#variable-m_args)  <br>_Arguments of simulation process._  |
|  double | [**m\_clock\_rate**](#variable-m_clock_rate)  <br>_the rate of inter-simulator cycle convert._  |
|  std::string | [**m\_command**](#variable-m_command)  <br>_Command of simulation process._  |
|  std::string | [**m\_log\_file**](#variable-m_log_file)  <br>_Path of logging name._  |
|  std::string | [**m\_pre\_copy**](#variable-m_pre_copy)  <br>_Files copy to sub-directory of simulator before executing._  |
|  bool | [**m\_to\_stdout**](#variable-m_to_stdout)  <br>_True means redirect output of this process to standard output._  |


## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**ProcessConfig**](#function-processconfig) (const std::string & \_\_cmd, const std::vector&lt; std::string &gt; & \_\_args, const std::string & \_\_log, bool \_\_to\_stdout, double \_\_clock\_rate, const std::string & \_\_pre\_copy) <br>_Construct_ [_**ProcessConfig**_](classInterChiplet_1_1ProcessConfig.md) _._ |








## Public Attributes Documentation


### variable m\_args 

```C++
std::vector<std::string> InterChiplet::ProcessConfig::m_args;
```




### variable m\_clock\_rate 

```C++
double InterChiplet::ProcessConfig::m_clock_rate;
```




### variable m\_command 

```C++
std::string InterChiplet::ProcessConfig::m_command;
```




### variable m\_log\_file 

```C++
std::string InterChiplet::ProcessConfig::m_log_file;
```




### variable m\_pre\_copy 

```C++
std::string InterChiplet::ProcessConfig::m_pre_copy;
```




### variable m\_to\_stdout 

```C++
bool InterChiplet::ProcessConfig::m_to_stdout;
```



## Public Functions Documentation


### function ProcessConfig 

_Construct_ [_**ProcessConfig**_](classInterChiplet_1_1ProcessConfig.md) _._
```C++
inline InterChiplet::ProcessConfig::ProcessConfig (
    const std::string & __cmd,
    const std::vector< std::string > & __args,
    const std::string & __log,
    bool __to_stdout,
    double __clock_rate,
    const std::string & __pre_copy
) 
```





**Parameters:**


* `__cmd` Command of simulation process. 
* `__args` Arguments of simulation process. 
* `__log` Path of logging name. 
* `__to_stdout` True means redirect output of this process to standard output. 
* `__clock_rate` the rate of inter-simulator cycle convert. 
* `__pre_copy` Files copy to sub-directory of simulator before executing. 




        

------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/benchmark_yaml.h`