
# Class InterChiplet::BenchmarkConfig



[**ClassList**](annotated.md) **>** [**InterChiplet**](namespaceInterChiplet.md) **>** [**BenchmarkConfig**](classInterChiplet_1_1BenchmarkConfig.md)



_Benchmark configuration structure._ 

* `#include <benchmark_yaml.h>`













## Public Attributes

| Type | Name |
| ---: | :--- |
|  std::string | [**m\_benchmark\_root**](#variable-m_benchmark_root)  <br>_Environments._  |
|  std::vector&lt; [**ProcessConfig**](classInterChiplet_1_1ProcessConfig.md) &gt; | [**m\_phase1\_proc\_cfg\_list**](#variable-m_phase1_proc_cfg_list)  <br>_List of configuration structures of phase 1._  |
|  std::vector&lt; [**ProcessConfig**](classInterChiplet_1_1ProcessConfig.md) &gt; | [**m\_phase2\_proc\_cfg\_list**](#variable-m_phase2_proc_cfg_list)  <br>_List of configuration structures of phase 2._  |
|  std::string | [**m\_simulator\_root**](#variable-m_simulator_root)  <br> |


## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**BenchmarkConfig**](#function-benchmarkconfig) (const std::string & file\_name) <br>_Parse YAML configuration file to get benchmark configuration._  |
|  void | [**yaml\_parse**](#function-yaml_parse) (const YAML::Node & config) <br>_Parse YAML configuration tree._  |








## Public Attributes Documentation


### variable m\_benchmark\_root 

```C++
std::string InterChiplet::BenchmarkConfig::m_benchmark_root;
```




### variable m\_phase1\_proc\_cfg\_list 

```C++
std::vector<ProcessConfig> InterChiplet::BenchmarkConfig::m_phase1_proc_cfg_list;
```




### variable m\_phase2\_proc\_cfg\_list 

```C++
std::vector<ProcessConfig> InterChiplet::BenchmarkConfig::m_phase2_proc_cfg_list;
```




### variable m\_simulator\_root 

```C++
std::string InterChiplet::BenchmarkConfig::m_simulator_root;
```



## Public Functions Documentation


### function BenchmarkConfig 

_Parse YAML configuration file to get benchmark configuration._ 
```C++
inline InterChiplet::BenchmarkConfig::BenchmarkConfig (
    const std::string & file_name
) 
```





**Parameters:**


* `file_name` Path of YAML configuration file. 




        

### function yaml\_parse 

_Parse YAML configuration tree._ 
```C++
inline void InterChiplet::BenchmarkConfig::yaml_parse (
    const YAML::Node & config
) 
```





**Parameters:**


* `config` Top node of YAML Tree. 




        

------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/benchmark_yaml.h`