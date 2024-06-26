
# Namespace InterChiplet



[**Namespace List**](namespaces.md) **>** [**InterChiplet**](namespaceInterChiplet.md)















## Classes

| Type | Name |
| ---: | :--- |
| class | [**PipeComm**](classInterChiplet_1_1PipeComm.md) <br>_Pipe communication structure._  |
| class | [**PipeCommUnit**](classInterChiplet_1_1PipeCommUnit.md) <br>_Structure for Single Pipe communication._  |
| class | [**SyncCommand**](classInterChiplet_1_1SyncCommand.md) <br>_Structure of synchronization command._  |

## Public Types

| Type | Name |
| ---: | :--- |
| typedef std::vector&lt; long &gt; | [**AddrType**](#typedef-addrtype)  <br>_Address type;._  |
| typedef double | [**InnerTimeType**](#typedef-innertimetype)  <br>_Time type used by interchiplet module._  |
| enum  | [**SyncCommType**](#enum-synccommtype)  <br>_Type of synchronization command between simulators._  |
| enum  | [**SyncProtocolDesc**](#enum-syncprotocoldesc)  <br>_Behavior descriptor of synchronization protocol._  |
| enum  | [**SysCallID**](#enum-syscallid)  <br>_Syscall ID used in CPU/GPU._  |
| typedef unsigned long long | [**TimeType**](#typedef-timetype)  <br>_Time type used between simulators._  |


## Public Attributes

| Type | Name |
| ---: | :--- |
|  decltype(syscall(0)) typedef | [**syscall\_return\_t**](#variable-syscall_return_t)  <br> |


## Public Functions

| Type | Name |
| ---: | :--- |
|  syscall\_return\_t | [**barrier**](#function-barrier) (int64\_t \_\_uid, int64\_t \_\_src\_x, int64\_t \_\_src\_y, int64\_t \_\_count=0) <br>_Barrier._  |
|  void | [**barrierSync**](#function-barriersync) (int \_\_src\_x, int \_\_src\_y, int \_\_uid, int \_\_count) <br>_Send BARRIER command and wait for SYNC command._  |
|  TimeType | [**cycleSync**](#function-cyclesync) (TimeType \_\_cycle) <br>_Send CYCLE command and wait for SYNC command._  |
|  std::string | [**dumpCmd**](#function-dumpcmd) (const [**SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br>_Dump command to a string for debugging._  |
|  syscall\_return\_t | [**launch**](#function-launch) (int64\_t \_\_dst\_x, int64\_t \_\_dst\_y, int64\_t \_\_src\_x, int64\_t \_\_src\_y) <br>_Launch application to remote chiplet._  |
|  void | [**launchSync**](#function-launchsync) (int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y) <br>_Send LAUNCH command and wait for SYNC command._  |
|  syscall\_return\_t | [**lock**](#function-lock) (int64\_t \_\_uid, int64\_t \_\_src\_x, int64\_t \_\_src\_y) <br>_Lock mutex._  |
|  void | [**lockSync**](#function-locksync) (int \_\_src\_x, int \_\_src\_y, int \_\_uid) <br>_Send UNLOCK command and wait for SYNC command._  |
|  [**SyncCommand**](classInterChiplet_1_1SyncCommand.md) | [**parseCmd**](#function-parsecmd) (const std::string & \_\_message) <br>_Parse command from string._  |
|  [**SyncCommand**](classInterChiplet_1_1SyncCommand.md) | [**parseCmd**](#function-parsecmd) (int \_\_fd\_in=STDIN\_FILENO) <br>_Receive command from stdin and parse the message. Used by simulator only._  |
|  std::string | [**pipeName**](#function-pipename) (const AddrType & \_\_src, const AddrType & \_\_dst) <br>_Return name of file name in a std::string. Return directory related to the directory of main process._  |
|  TimeType | [**readSync**](#function-readsync) (TimeType \_\_cycle, int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y, int \_\_nbyte, long \_\_desc) <br>_Send READ command and wait for SYNC command._  |
|  syscall\_return\_t | [**receiveMessage**](#function-receivemessage) (int64\_t \_\_dst\_x, int64\_t \_\_dst\_y, int64\_t \_\_src\_x, int64\_t \_\_src\_y, void \* \_\_addr, int64\_t \_\_nbyte) <br>_Read data from remote chiplet._  |
|  std::string | [**receiveSync**](#function-receivesync) (int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y) <br>_Send RECEIVE command and wait for SYNC command._  |
|  void | [**sendBarrierCmd**](#function-sendbarriercmd) (int \_\_src\_x, int \_\_src\_y, int \_\_uid, int \_\_count) <br>_Send BARRIER command._  |
|  void | [**sendCycleCmd**](#function-sendcyclecmd) (TimeType \_\_cycle) <br>_Send CYCLE command._  |
|  void | [**sendLaunchCmd**](#function-sendlaunchcmd) (int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y) <br>_Send LAUNCH command._  |
|  void | [**sendLockCmd**](#function-sendlockcmd) (int \_\_src\_x, int \_\_src\_y, int \_\_uid) <br>_Send LOCK command._  |
|  syscall\_return\_t | [**sendMessage**](#function-sendmessage) (int64\_t \_\_dst\_x, int64\_t \_\_dst\_y, int64\_t \_\_src\_x, int64\_t \_\_src\_y, void \* \_\_addr, int64\_t \_\_nbyte) <br>_Send data to remote chiplet._  |
|  void | [**sendReadCmd**](#function-sendreadcmd) (TimeType \_\_cycle, int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y, int \_\_nbyte, long \_\_desc) <br>_Send READ command._  |
|  void | [**sendReceiveCmd**](#function-sendreceivecmd) (int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y) <br>_Send RECEIVE command._  |
|  void | [**sendResultCmd**](#function-sendresultcmd) () <br>_Send RESULT command._  |
|  void | [**sendResultCmd**](#function-sendresultcmd) (const std::vector&lt; std::string &gt; & \_\_res\_list) <br>_Send RESULT command._  |
|  void | [**sendResultCmd**](#function-sendresultcmd) (const std::vector&lt; long &gt; & \_\_res\_list) <br>_Send RESULT command._  |
|  void | [**sendResultCmd**](#function-sendresultcmd) (int \_\_fd) <br>_Send RESULT command._  |
|  void | [**sendResultCmd**](#function-sendresultcmd) (int \_\_fd, const std::vector&lt; std::string &gt; & \_\_res\_list) <br>_Send RESULT command._  |
|  void | [**sendResultCmd**](#function-sendresultcmd) (int \_\_fd, const std::vector&lt; long &gt; & \_\_res\_list) <br>_Send RESULT command._  |
|  void | [**sendSendCmd**](#function-sendsendcmd) (int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y) <br>_Send SEND command._  |
|  std::string | [**sendSync**](#function-sendsync) (int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y) <br>_Send SEND command and wait for SYNC command._  |
|  void | [**sendSyncCmd**](#function-sendsynccmd) (TimeType \_\_cycle) <br>_Send SYNC command._  |
|  void | [**sendSyncCmd**](#function-sendsynccmd) (int \_\_fd, TimeType \_\_cycle) <br>_Send SYNC command to specified file descriptor._  |
|  void | [**sendUnlockCmd**](#function-sendunlockcmd) (int \_\_src\_x, int \_\_src\_y, int \_\_uid) <br>_Send UNLOCK command._  |
|  void | [**sendWaitlaunchCmd**](#function-sendwaitlaunchcmd) (int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y) <br>_Send WAITLAUNCH command._  |
|  void | [**sendWriteCmd**](#function-sendwritecmd) (TimeType \_\_cycle, int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y, int \_\_nbyte, long \_\_desc) <br>_Send WRITE command._  |
|  syscall\_return\_t | [**unlock**](#function-unlock) (int64\_t \_\_uid, int64\_t \_\_src\_x, int64\_t \_\_src\_y) <br>_Unlock mutex._  |
|  void | [**unlockSync**](#function-unlocksync) (int \_\_src\_x, int \_\_src\_y, int \_\_uid) <br>_Send UNLOCK command and wait for SYNC command._  |
|  syscall\_return\_t | [**waitLaunch**](#function-waitlaunch) (int64\_t \_\_dst\_x, int64\_t \_\_dst\_y, int64\_t \* \_\_src\_x, int64\_t \* \_\_src\_y) <br>_Wait launch from remote chiplet._  |
|  void | [**waitlaunchSync**](#function-waitlaunchsync) (int \* \_\_src\_x, int \* \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y) <br>_Send WAITLAUNCH command and wait for LAUNCH command._  |
|  TimeType | [**writeSync**](#function-writesync) (TimeType \_\_cycle, int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y, int \_\_nbyte, long \_\_desc) <br>_Send WRITE command and wait for SYNC command._  |








## Public Types Documentation


### typedef AddrType 

```C++
typedef std::vector<long> InterChiplet::AddrType;
```




### typedef InnerTimeType 

```C++
typedef double InterChiplet::InnerTimeType;
```




### enum SyncCommType 

```C++
enum InterChiplet::SyncCommType {
    SC_CYCLE,
    SC_SEND,
    SC_RECEIVE,
    SC_BARRIER,
    SC_LOCK,
    SC_UNLOCK,
    SC_LAUNCH,
    SC_WAITLAUNCH,
    SC_READ,
    SC_WRITE,
    SC_SYNC,
    SC_RESULT
};
```




### enum SyncProtocolDesc 

```C++
enum InterChiplet::SyncProtocolDesc {
    SPD_ACK = 0x01,
    SPD_PRE_SYNC = 0x02,
    SPD_POST_SYNC = 0x04,
    SPD_LAUNCH = 0x10000,
    SPD_BARRIER = 0x20000,
    SPD_LOCK = 0x40000,
    SPD_UNLOCK = 0x80000
};
```




### enum SysCallID 

```C++
enum InterChiplet::SysCallID {
    SYSCALL_LAUNCH = 501,
    SYSCALL_WAITLAUNCH = 502,
    SYSCALL_BARRIER = 503,
    SYSCALL_LOCK = 504,
    SYSCALL_UNLOCK = 505,
    SYSCALL_REMOTE_READ = 506,
    SYSCALL_REMOTE_WRITE = 507
};
```




### typedef TimeType 

```C++
typedef unsigned long long InterChiplet::TimeType;
```



## Public Attributes Documentation


### variable syscall\_return\_t 

```C++
decltype(syscall(0)) typedef InterChiplet::syscall_return_t;
```



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




        

### function barrierSync 

_Send BARRIER command and wait for SYNC command._ 
```C++
inline void InterChiplet::barrierSync (
    int __src_x,
    int __src_y,
    int __uid,
    int __count
) 
```





**Parameters:**


* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__uid` Barrier ID. 
* `__count` Number of items in barrier. 




        

### function cycleSync 

_Send CYCLE command and wait for SYNC command._ 
```C++
inline TimeType InterChiplet::cycleSync (
    TimeType __cycle
) 
```





**Parameters:**


* `__cycle` Cycle to send CYCLE command. 



**Returns:**

Cycle to receive SYNC command. 





        

### function dumpCmd 

_Dump command to a string for debugging._ 
```C++
inline std::string InterChiplet::dumpCmd (
    const SyncCommand & __cmd
) 
```





**Parameters:**


* `__cmd` Structure of synchronization command. 



**Returns:**

String of message. 





        

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




        

### function launchSync 

_Send LAUNCH command and wait for SYNC command._ 
```C++
inline void InterChiplet::launchSync (
    int __src_x,
    int __src_y,
    int __dst_x,
    int __dst_y
) 
```





**Parameters:**


* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__dst_x` Destiantion address in X-axis. 
* `__dst_y` Destination address in Y-axis. 




        

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




        

### function lockSync 

_Send UNLOCK command and wait for SYNC command._ 
```C++
inline void InterChiplet::lockSync (
    int __src_x,
    int __src_y,
    int __uid
) 
```





**Parameters:**


* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__uid` Mutex ID. 




        

### function parseCmd 

_Parse command from string._ 
```C++
inline SyncCommand InterChiplet::parseCmd (
    const std::string & __message
) 
```





**Parameters:**


* `__message` String of message. 



**Returns:**

Structure of synchronization command. 





        

### function parseCmd 

_Receive command from stdin and parse the message. Used by simulator only._ 
```C++
inline SyncCommand InterChiplet::parseCmd (
    int __fd_in=STDIN_FILENO
) 
```





**Parameters:**


* `__fd_in` Input file descriptor. 



**Returns:**

Structure of synchronization command. 





        

### function pipeName 

_Return name of file name in a std::string. Return directory related to the directory of main process._ 
```C++
inline std::string InterChiplet::pipeName (
    const AddrType & __src,
    const AddrType & __dst
) 
```





**Parameters:**


* `__src` Source address. 
* `__dst` Destiantion address. 




        

### function readSync 

_Send READ command and wait for SYNC command._ 
```C++
inline TimeType InterChiplet::readSync (
    TimeType __cycle,
    int __src_x,
    int __src_y,
    int __dst_x,
    int __dst_y,
    int __nbyte,
    long __desc
) 
```





**Parameters:**


* `__cycle` Cycle to send READ command. 
* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__dst_x` Destiantion address in X-axis. 
* `__dst_y` Destination address in Y-axis. 
* `__nbyte` Number of bytes to read. 
* `__desc` Synchronization protocol descriptor. 



**Returns:**

Cycle to receive SYNC command. 





        

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




        

### function receiveSync 

_Send RECEIVE command and wait for SYNC command._ 
```C++
inline std::string InterChiplet::receiveSync (
    int __src_x,
    int __src_y,
    int __dst_x,
    int __dst_y
) 
```





**Parameters:**


* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__dst_x` Destiantion address in X-axis. 
* `__dst_y` Destination address in Y-axis. 



**Returns:**

Pipe name. 





        

### function sendBarrierCmd 

_Send BARRIER command._ 
```C++
inline void InterChiplet::sendBarrierCmd (
    int __src_x,
    int __src_y,
    int __uid,
    int __count
) 
```





**Parameters:**


* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__uid` Barrier ID. 
* `__count` Number of items in barrier. 




        

### function sendCycleCmd 

_Send CYCLE command._ 
```C++
inline void InterChiplet::sendCycleCmd (
    TimeType __cycle
) 
```





**Parameters:**


* `__cycle` Cycle to send CYCLE command. 




        

### function sendLaunchCmd 

_Send LAUNCH command._ 
```C++
inline void InterChiplet::sendLaunchCmd (
    int __src_x,
    int __src_y,
    int __dst_x,
    int __dst_y
) 
```





**Parameters:**


* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__dst_x` Destiantion address in X-axis. 
* `__dst_y` Destination address in Y-axis. 




        

### function sendLockCmd 

_Send LOCK command._ 
```C++
inline void InterChiplet::sendLockCmd (
    int __src_x,
    int __src_y,
    int __uid
) 
```





**Parameters:**


* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__uid` Barrier ID. 




        

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




        

### function sendReadCmd 

_Send READ command._ 
```C++
inline void InterChiplet::sendReadCmd (
    TimeType __cycle,
    int __src_x,
    int __src_y,
    int __dst_x,
    int __dst_y,
    int __nbyte,
    long __desc
) 
```





**Parameters:**


* `__cycle` Cycle to send READ command. 
* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__dst_x` Destiantion address in X-axis. 
* `__dst_y` Destination address in Y-axis. 
* `__nbyte` Number of bytes to read. 
* `__desc` Synchronization protocol descriptor. 




        

### function sendReceiveCmd 

_Send RECEIVE command._ 
```C++
inline void InterChiplet::sendReceiveCmd (
    int __src_x,
    int __src_y,
    int __dst_x,
    int __dst_y
) 
```





**Parameters:**


* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__dst_x` Destiantion address in X-axis. 
* `__dst_y` Destination address in Y-axis. 




        

### function sendResultCmd 

```C++
inline void InterChiplet::sendResultCmd () 
```




### function sendResultCmd 

_Send RESULT command._ 
```C++
inline void InterChiplet::sendResultCmd (
    const std::vector< std::string > & __res_list
) 
```





**Parameters:**


* `__res_list` Result list. 




        

### function sendResultCmd 

_Send RESULT command._ 
```C++
inline void InterChiplet::sendResultCmd (
    const std::vector< long > & __res_list
) 
```





**Parameters:**


* `__res_list` Result list. 




        

### function sendResultCmd 

_Send RESULT command._ 
```C++
inline void InterChiplet::sendResultCmd (
    int __fd
) 
```





**Parameters:**


* `__fd` File descriptor. 




        

### function sendResultCmd 

_Send RESULT command._ 
```C++
inline void InterChiplet::sendResultCmd (
    int __fd,
    const std::vector< std::string > & __res_list
) 
```





**Parameters:**


* `__fd` File descriptor. 
* `__res_list` Result list. 




        

### function sendResultCmd 

_Send RESULT command._ 
```C++
inline void InterChiplet::sendResultCmd (
    int __fd,
    const std::vector< long > & __res_list
) 
```





**Parameters:**


* `__fd` File descriptor. 
* `__res_list` Result list. 




        

### function sendSendCmd 

_Send SEND command._ 
```C++
inline void InterChiplet::sendSendCmd (
    int __src_x,
    int __src_y,
    int __dst_x,
    int __dst_y
) 
```





**Parameters:**


* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__dst_x` Destiantion address in X-axis. 
* `__dst_y` Destination address in Y-axis. 




        

### function sendSync 

_Send SEND command and wait for SYNC command._ 
```C++
inline std::string InterChiplet::sendSync (
    int __src_x,
    int __src_y,
    int __dst_x,
    int __dst_y
) 
```





**Parameters:**


* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__dst_x` Destiantion address in X-axis. 
* `__dst_y` Destination address in Y-axis. 



**Returns:**

Pipe name. 





        

### function sendSyncCmd 

_Send SYNC command._ 
```C++
inline void InterChiplet::sendSyncCmd (
    TimeType __cycle
) 
```





**Parameters:**


* `__cycle` Cycle to receive SYNC command. 




        

### function sendSyncCmd 

_Send SYNC command to specified file descriptor._ 
```C++
inline void InterChiplet::sendSyncCmd (
    int __fd,
    TimeType __cycle
) 
```





**Parameters:**


* `__fd` File descriptor. 
* `__cycle` Cycle to receive SYNC command. 




        

### function sendUnlockCmd 

_Send UNLOCK command._ 
```C++
inline void InterChiplet::sendUnlockCmd (
    int __src_x,
    int __src_y,
    int __uid
) 
```





**Parameters:**


* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__uid` Barrier ID. 




        

### function sendWaitlaunchCmd 

_Send WAITLAUNCH command._ 
```C++
inline void InterChiplet::sendWaitlaunchCmd (
    int __src_x,
    int __src_y,
    int __dst_x,
    int __dst_y
) 
```





**Parameters:**


* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__dst_x` Destiantion address in X-axis. 
* `__dst_y` Destination address in Y-axis. 




        

### function sendWriteCmd 

_Send WRITE command._ 
```C++
inline void InterChiplet::sendWriteCmd (
    TimeType __cycle,
    int __src_x,
    int __src_y,
    int __dst_x,
    int __dst_y,
    int __nbyte,
    long __desc
) 
```





**Parameters:**


* `__cycle` Cycle to send WRITE command. 
* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__dst_x` Destiantion address in X-axis. 
* `__dst_y` Destination address in Y-axis. 
* `__nbyte` Number of bytes to write. 
* `__desc` Synchronization protocol descriptor. 




        

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




        

### function unlockSync 

_Send UNLOCK command and wait for SYNC command._ 
```C++
inline void InterChiplet::unlockSync (
    int __src_x,
    int __src_y,
    int __uid
) 
```





**Parameters:**


* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__uid` Mutex ID. 




        

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




        

### function waitlaunchSync 

_Send WAITLAUNCH command and wait for LAUNCH command._ 
```C++
inline void InterChiplet::waitlaunchSync (
    int * __src_x,
    int * __src_y,
    int __dst_x,
    int __dst_y
) 
```





**Parameters:**


* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__dst_x` Destiantion address in X-axis. 
* `__dst_y` Destination address in Y-axis. 




        

### function writeSync 

_Send WRITE command and wait for SYNC command._ 
```C++
inline TimeType InterChiplet::writeSync (
    TimeType __cycle,
    int __src_x,
    int __src_y,
    int __dst_x,
    int __dst_y,
    int __nbyte,
    long __desc
) 
```





**Parameters:**


* `__cycle` Cycle to send WRITE command. 
* `__src_x` Source address in X-axis. 
* `__src_y` Source address in Y-axis. 
* `__dst_x` Destiantion address in X-axis. 
* `__dst_y` Destination address in Y-axis. 
* `__nbyte` Number of bytes to write. 
* `__desc` Synchronization protocol descriptor. 



**Returns:**

Cycle to receive SYNC command. 





        

------------------------------
The documentation for this class was generated from the following file `/data_sda/junwan02/legosim/Chiplet_Heterogeneous_newVersion/interchiplet/includes/apis_c.h`