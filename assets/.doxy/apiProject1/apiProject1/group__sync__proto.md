
# Group sync\_proto



[**Modules**](modules.md) **>** [**sync\_proto**](group__sync__proto.md)



_Synchronization protocol interface._ 
















## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**InterChiplet::barrierSync**](#function-barriersync) (int \_\_src\_x, int \_\_src\_y, int \_\_uid, int \_\_count) <br>_Send BARRIER command and wait for SYNC command._  |
|  TimeType | [**InterChiplet::cycleSync**](#function-cyclesync) (TimeType \_\_cycle) <br>_Send CYCLE command and wait for SYNC command._  |
|  std::string | [**InterChiplet::dumpCmd**](#function-dumpcmd) (const [**SyncCommand**](classInterChiplet_1_1SyncCommand.md) & \_\_cmd) <br>_Dump command to a string for debugging._  |
|  void | [**InterChiplet::launchSync**](#function-launchsync) (int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y) <br>_Send LAUNCH command and wait for SYNC command._  |
|  void | [**InterChiplet::lockSync**](#function-locksync) (int \_\_src\_x, int \_\_src\_y, int \_\_uid) <br>_Send UNLOCK command and wait for SYNC command._  |
|  [**SyncCommand**](classInterChiplet_1_1SyncCommand.md) | [**InterChiplet::parseCmd**](#function-parsecmd) (const std::string & \_\_message) <br>_Parse command from string._  |
|  [**SyncCommand**](classInterChiplet_1_1SyncCommand.md) | [**InterChiplet::parseCmd**](#function-parsecmd) (int \_\_fd\_in=STDIN\_FILENO) <br>_Receive command from stdin and parse the message. Used by simulator only._  |
|  std::string | [**InterChiplet::pipeName**](#function-pipename) (const AddrType & \_\_src, const AddrType & \_\_dst) <br>_Return name of file name in a std::string. Return directory related to the directory of main process._  |
|  TimeType | [**InterChiplet::readSync**](#function-readsync) (TimeType \_\_cycle, int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y, int \_\_nbyte, long \_\_desc) <br>_Send READ command and wait for SYNC command._  |
|  std::string | [**InterChiplet::receiveSync**](#function-receivesync) (int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y) <br>_Send RECEIVE command and wait for SYNC command._  |
|  void | [**InterChiplet::sendBarrierCmd**](#function-sendbarriercmd) (int \_\_src\_x, int \_\_src\_y, int \_\_uid, int \_\_count) <br>_Send BARRIER command._  |
|  void | [**InterChiplet::sendCycleCmd**](#function-sendcyclecmd) (TimeType \_\_cycle) <br>_Send CYCLE command._  |
|  void | [**InterChiplet::sendLaunchCmd**](#function-sendlaunchcmd) (int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y) <br>_Send LAUNCH command._  |
|  void | [**InterChiplet::sendLockCmd**](#function-sendlockcmd) (int \_\_src\_x, int \_\_src\_y, int \_\_uid) <br>_Send LOCK command._  |
|  void | [**InterChiplet::sendReadCmd**](#function-sendreadcmd) (TimeType \_\_cycle, int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y, int \_\_nbyte, long \_\_desc) <br>_Send READ command._  |
|  void | [**InterChiplet::sendReceiveCmd**](#function-sendreceivecmd) (int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y) <br>_Send RECEIVE command._  |
|  void | [**InterChiplet::sendResultCmd**](#function-sendresultcmd) () <br>_Send RESULT command._  |
|  void | [**InterChiplet::sendResultCmd**](#function-sendresultcmd) (const std::vector&lt; std::string &gt; & \_\_res\_list) <br>_Send RESULT command._  |
|  void | [**InterChiplet::sendResultCmd**](#function-sendresultcmd) (const std::vector&lt; long &gt; & \_\_res\_list) <br>_Send RESULT command._  |
|  void | [**InterChiplet::sendResultCmd**](#function-sendresultcmd) (int \_\_fd) <br>_Send RESULT command._  |
|  void | [**InterChiplet::sendResultCmd**](#function-sendresultcmd) (int \_\_fd, const std::vector&lt; std::string &gt; & \_\_res\_list) <br>_Send RESULT command._  |
|  void | [**InterChiplet::sendResultCmd**](#function-sendresultcmd) (int \_\_fd, const std::vector&lt; long &gt; & \_\_res\_list) <br>_Send RESULT command._  |
|  void | [**InterChiplet::sendSendCmd**](#function-sendsendcmd) (int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y) <br>_Send SEND command._  |
|  std::string | [**InterChiplet::sendSync**](#function-sendsync) (int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y) <br>_Send SEND command and wait for SYNC command._  |
|  void | [**InterChiplet::sendSyncCmd**](#function-sendsynccmd) (TimeType \_\_cycle) <br>_Send SYNC command._  |
|  void | [**InterChiplet::sendSyncCmd**](#function-sendsynccmd) (int \_\_fd, TimeType \_\_cycle) <br>_Send SYNC command to specified file descriptor._  |
|  void | [**InterChiplet::sendUnlockCmd**](#function-sendunlockcmd) (int \_\_src\_x, int \_\_src\_y, int \_\_uid) <br>_Send UNLOCK command._  |
|  void | [**InterChiplet::sendWaitlaunchCmd**](#function-sendwaitlaunchcmd) (int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y) <br>_Send WAITLAUNCH command._  |
|  void | [**InterChiplet::sendWriteCmd**](#function-sendwritecmd) (TimeType \_\_cycle, int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y, int \_\_nbyte, long \_\_desc) <br>_Send WRITE command._  |
|  void | [**InterChiplet::unlockSync**](#function-unlocksync) (int \_\_src\_x, int \_\_src\_y, int \_\_uid) <br>_Send UNLOCK command and wait for SYNC command._  |
|  void | [**InterChiplet::waitlaunchSync**](#function-waitlaunchsync) (int \* \_\_src\_x, int \* \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y) <br>_Send WAITLAUNCH command and wait for LAUNCH command._  |
|  TimeType | [**InterChiplet::writeSync**](#function-writesync) (TimeType \_\_cycle, int \_\_src\_x, int \_\_src\_y, int \_\_dst\_x, int \_\_dst\_y, int \_\_nbyte, long \_\_desc) <br>_Send WRITE command and wait for SYNC command._  |








## Public Functions Documentation


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
