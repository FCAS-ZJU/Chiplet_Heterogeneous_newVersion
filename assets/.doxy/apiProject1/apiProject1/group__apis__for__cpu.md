
# Group apis\_for\_cpu



[**Modules**](modules.md) **>** [**apis\_for\_cpu**](group__apis__for__cpu.md)



_APIs for CPU._ 
















## Public Functions

| Type | Name |
| ---: | :--- |
|  syscall\_return\_t | [**InterChiplet::barrier**](#function-barrier) (int64\_t \_\_uid, int64\_t \_\_src\_x, int64\_t \_\_src\_y, int64\_t \_\_count=0) <br>_Barrier._  |
|  syscall\_return\_t | [**InterChiplet::launch**](#function-launch) (int64\_t \_\_dst\_x, int64\_t \_\_dst\_y, int64\_t \_\_src\_x, int64\_t \_\_src\_y) <br>_Launch application to remote chiplet._  |
|  syscall\_return\_t | [**InterChiplet::lock**](#function-lock) (int64\_t \_\_uid, int64\_t \_\_src\_x, int64\_t \_\_src\_y) <br>_Lock mutex._  |
|  syscall\_return\_t | [**InterChiplet::receiveMessage**](#function-receivemessage) (int64\_t \_\_dst\_x, int64\_t \_\_dst\_y, int64\_t \_\_src\_x, int64\_t \_\_src\_y, void \* \_\_addr, int64\_t \_\_nbyte) <br>_Read data from remote chiplet._  |
|  syscall\_return\_t | [**InterChiplet::sendMessage**](#function-sendmessage) (int64\_t \_\_dst\_x, int64\_t \_\_dst\_y, int64\_t \_\_src\_x, int64\_t \_\_src\_y, void \* \_\_addr, int64\_t \_\_nbyte) <br>_Send data to remote chiplet._  |
|  syscall\_return\_t | [**InterChiplet::unlock**](#function-unlock) (int64\_t \_\_uid, int64\_t \_\_src\_x, int64\_t \_\_src\_y) <br>_Unlock mutex._  |
|  syscall\_return\_t | [**InterChiplet::waitLaunch**](#function-waitlaunch) (int64\_t \_\_dst\_x, int64\_t \_\_dst\_y, int64\_t \* \_\_src\_x, int64\_t \* \_\_src\_y) <br>_Wait launch from remote chiplet._  |








## Public Functions Documentation


### function barrier 

_Barrier._ 
```C++
syscall_return_t InterChiplet::barrier (
    int64_t __uid,
    int64_t __src_x,
    int64_t __src_y,
    int64_t __count=0
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
syscall_return_t InterChiplet::launch (
    int64_t __dst_x,
    int64_t __dst_y,
    int64_t __src_x,
    int64_t __src_y
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
syscall_return_t InterChiplet::lock (
    int64_t __uid,
    int64_t __src_x,
    int64_t __src_y
) 
```





**Parameters:**


* `__uid` Mutex ID. 
* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 




        

### function receiveMessage 

_Read data from remote chiplet._ 
```C++
syscall_return_t InterChiplet::receiveMessage (
    int64_t __dst_x,
    int64_t __dst_y,
    int64_t __src_x,
    int64_t __src_y,
    void * __addr,
    int64_t __nbyte
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
syscall_return_t InterChiplet::sendMessage (
    int64_t __dst_x,
    int64_t __dst_y,
    int64_t __src_x,
    int64_t __src_y,
    void * __addr,
    int64_t __nbyte
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
syscall_return_t InterChiplet::unlock (
    int64_t __uid,
    int64_t __src_x,
    int64_t __src_y
) 
```





**Parameters:**


* `__uid` Mutex ID. 
* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 




        

### function waitLaunch 

_Wait launch from remote chiplet._ 
```C++
syscall_return_t InterChiplet::waitLaunch (
    int64_t __dst_x,
    int64_t __dst_y,
    int64_t * __src_x,
    int64_t * __src_y
) 
```





**Parameters:**


* `__dst_x` Destination address in X-axis. 
* `__dst_y` Destination address in Y-axis. 
* `__src_x` Source address in X-axis. Return value. 
* `__src_y` Source address in Y-axis. Return value. 




        

------------------------------
