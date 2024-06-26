
# Group apis\_for\_cuda



[**Modules**](modules.md) **>** [**apis\_for\_cuda**](group__apis__for__cuda.md)



_APIs for CUDA._ 
















## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ cudaError\_t CUDARTAPI | [**barrier**](#function-barrier) (int \_\_uid, int \_\_src\_x, int \_\_src\_y, int \_\_count=0) <br>_Barrier._  |
|  \_\_host\_\_ cudaError\_t CUDARTAPI | [**launch**](#function-launch) (int \_\_dst\_x, int \_\_dst\_y, int \_\_src\_x, int \_\_src\_y) <br>_Launch application to remote chiplet._  |
|  \_\_host\_\_ cudaError\_t CUDARTAPI | [**lock**](#function-lock) (int \_\_uid, int \_\_src\_x, int \_\_src\_y) <br>_Lock mutex._  |
|  \_\_host\_\_ cudaError\_t CUDARTAPI | [**receiveMessage**](#function-receivemessage) (int \_\_dst\_x, int \_\_dst\_y, int \_\_src\_x, int \_\_srx\_y, void \* \_\_addr, int \_\_nbyte) <br>_Read data from remote chiplet._  |
|  \_\_host\_\_ cudaError\_t CUDARTAPI | [**sendMessage**](#function-sendmessage) (int \_\_dst\_x, int \_\_dst\_y, int \_\_src\_x, int \_\_srx\_y, void \* \_\_addr, int \_\_nbyte) <br>_Send data to remote chiplet._  |
|  \_\_host\_\_ cudaError\_t CUDARTAPI | [**unlock**](#function-unlock) (int \_\_uid, int \_\_src\_x, int \_\_src\_y) <br>_Unlock mutex._  |
|  \_\_host\_\_ cudaError\_t CUDARTAPI | [**waitLaunch**](#function-waitlaunch) (int \_\_dst\_x, int \_\_dst\_y, int \* \_\_src\_x, int \* \_\_src\_y) <br>_Wait launch from remote chiplet._  |








## Public Functions Documentation


### function barrier 

_Barrier._ 
```C++
__host__ cudaError_t CUDARTAPI barrier (
    int __uid,
    int __src_x,
    int __src_y,
    int __count=0
) 
```





**Parameters:**


* `__uid` Barrier ID. 
* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__count` Number of threads to barrier. 




        

### function launch 

_Launch application to remote chiplet._ 
```C++
__host__ cudaError_t CUDARTAPI launch (
    int __dst_x,
    int __dst_y,
    int __src_x,
    int __src_y
) 
```





**Parameters:**


* `__dst_x` Destination address in X-axis. 
* `__dst_y` Destination address in Y-axis. 
* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 




        

### function lock 

_Lock mutex._ 
```C++
__host__ cudaError_t CUDARTAPI lock (
    int __uid,
    int __src_x,
    int __src_y
) 
```





**Parameters:**


* `__uid` Mutex ID. 
* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 




        

### function receiveMessage 

_Read data from remote chiplet._ 
```C++
__host__ cudaError_t CUDARTAPI receiveMessage (
    int __dst_x,
    int __dst_y,
    int __src_x,
    int __srx_y,
    void * __addr,
    int __nbyte
) 
```





**Parameters:**


* `__dst_x` Destination address in X-axis. 
* `__dst_y` Destination address in Y-axis. 
* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__addr` Data address. 
* `__nbyte` Number of bytes. 




        

### function sendMessage 

_Send data to remote chiplet._ 
```C++
__host__ cudaError_t CUDARTAPI sendMessage (
    int __dst_x,
    int __dst_y,
    int __src_x,
    int __srx_y,
    void * __addr,
    int __nbyte
) 
```





**Parameters:**


* `__dst_x` Destination address in X-axis. 
* `__dst_y` Destination address in Y-axis. 
* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__addr` Data address. 
* `__nbyte` Number of bytes. 




        

### function unlock 

_Unlock mutex._ 
```C++
__host__ cudaError_t CUDARTAPI unlock (
    int __uid,
    int __src_x,
    int __src_y
) 
```





**Parameters:**


* `__uid` Mutex ID. 
* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 




        

### function waitLaunch 

_Wait launch from remote chiplet._ 
```C++
__host__ cudaError_t CUDARTAPI waitLaunch (
    int __dst_x,
    int __dst_y,
    int * __src_x,
    int * __src_y
) 
```





**Parameters:**


* `__dst_x` Destination address in X-axis. 
* `__dst_y` Destination address in Y-axis. 
* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 




        

------------------------------
