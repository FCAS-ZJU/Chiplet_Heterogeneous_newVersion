
# Class InterChiplet::CmdLineOptions



[**ClassList**](annotated.md) **>** [**InterChiplet**](namespaceInterChiplet.md) **>** [**CmdLineOptions**](classInterChiplet_1_1CmdLineOptions.md)



_Options from command line._ 

* `#include <cmdline_options.h>`













## Public Attributes

| Type | Name |
| ---: | :--- |
|  std::string | [**m\_bench**](#variable-m_bench)  <br>_Path of benchmark configuration yaml._  |
|  std::string | [**m\_cwd**](#variable-m_cwd)  <br>_New working directory._  |
|  bool | [**m\_debug**](#variable-m_debug)  <br>_Print debug information._  |
|  double | [**m\_err\_rate\_threshold**](#variable-m_err_rate_threshold)  <br>_Error rate threshold, used to quit iteration._  |
|  long | [**m\_timeout\_threshold**](#variable-m_timeout_threshold)  <br>_Timeout threshold, in term of round._  |


## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**CmdLineOptions**](#function-cmdlineoptions) () <br>_Constructor._  |
|  int | [**parse**](#function-parse) (int argc, const char \* argv) <br>_Read options from command line._  |








## Public Attributes Documentation


### variable m\_bench 

```C++
std::string InterChiplet::CmdLineOptions::m_bench;
```




### variable m\_cwd 

```C++
std::string InterChiplet::CmdLineOptions::m_cwd;
```




### variable m\_debug 

```C++
bool InterChiplet::CmdLineOptions::m_debug;
```




### variable m\_err\_rate\_threshold 

```C++
double InterChiplet::CmdLineOptions::m_err_rate_threshold;
```




### variable m\_timeout\_threshold 

```C++
long InterChiplet::CmdLineOptions::m_timeout_threshold;
```



## Public Functions Documentation


### function CmdLineOptions 

```C++
inline InterChiplet::CmdLineOptions::CmdLineOptions () 
```




### function parse 

_Read options from command line._ 
```C++
inline int InterChiplet::CmdLineOptions::parse (
    int argc,
    const char * argv
) 
```





**Parameters:**


* `argc` Number of argument. 
* `argv` String of argument. 




        

------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/cmdline_options.h`