# Communication

You can find the following topics on this page:

- The sequences of synchronization protocol to handle communication transactions between processes.
- The algorithm to handle synchronization protocol within *interchiplet*.
- The algorithm to calculate the end cycle of one communication transaction.

## Command syntax

```
# sendMessage
SEND <src_x> <src_y> <dst_x> <dst_y>
WRITE <cycle> <src_x> <src_y> <dst_x> <dst_y> <nbytes> <desc>

# receiveMessage
RECEIVE <src_x> <src_y> <dst_x> <dst_y>
READ <cycle> <src_x> <src_y> <dst_x> <dst_y> <nbytes> <desc>
```

`src_x` and `src_y` present the source address and `dst_x` and `dst_y` present the destination address.

The `cycle` field in the WRITE command presents the time when the source component starts. The `cycle` field in the READ command presents the time when the destination component starts to wait for data to be ready for reading.

The `nbytes` field presents the total byte number of communication. `desc` describes the transaction behavior.

The figure below shows the relationship between arguments of APIs and commands.

```mermaid
flowchart TB

subgraph sendMessage
A1[__src_x]
A2[__src_y]
A3[__dst_x]
A4[__dst_y]
A5[__addr]
A6[__nbyte]
end

subgraph SEND command
B1[src_x]
B2[src_y]
B3[dst_x]
B4[dst_y]
end

subgraph WRITE command
C0[cycle]
C1[src_x]
C2[src_y]
C3[dst_x]
C4[dst_y]
C6[nbytes]
C7[desc]
end

A1 -.-> B1 -.-> C1
A2 -.-> B2 -.-> C2
A3 -.-> B3 -.-> C3
A4 -.-> B4 -.-> C4
A6 -.-> C6

```

```mermaid
flowchart TB

subgraph receiveMessage
D1[__src_x]
D2[__src_y]
D3[__dst_x]
D4[__dst_y]
D5[__addr]
D6[__nbyte]
end

subgraph RECEIVE command
E1[src_x]
E2[src_y]
E3[dst_x]
E4[dst_y]
end

subgraph READ command
F0[cycle]
F1[src_x]
F2[src_y]
F3[dst_x]
F4[dst_y]
F6[nbytes]
F7[desc]
end

D1 -.-> E1 -.-> F1
D2 -.-> E2 -.-> F2
D3 -.-> E3 -.-> F3
D4 -.-> E4 -.-> F4
D6 -.-> F6

```

## Command Sequence

One example of the command sequence is shown below:

```mermaid
sequenceDiagram
autonumber

participant SP0 as Simulator<br/>Process 0
participant interchiplet
participant SP1 as Simulator<br/>Process 1

activate SP0
activate SP1

Note over SP0,SP1: Example starts

SP0->>interchiplet: SEND 0 0 0 1
deactivate SP0
activate interchiplet
Note over interchiplet: Create Pipe name buffer0_0_0_1.
interchiplet->>SP0: RESULT 1 ../buffer0_0_0_1
deactivate interchiplet

SP1->>interchiplet: RECEIVE 0 0 0 0 1
deactivate SP1
activate interchiplet
Note over interchiplet: Reuse Pipe name buffer0_0_0_1.
interchiplet->>SP1: RESULT 1 ../buffer0_0_0_1
deactivate interchiplet

SP0->>SP1: Send data from (0,0) to (0,1) through Named Pipe ../buffer0_0_0_1

SP0->>interchiplet: WRITE 2578659 0 0 0 1 80000 0
activate interchiplet
Note over interchiplet: Register WRITE command.
deactivate interchiplet

SP1->>interchiplet: READ 2276672 0 0 0 1 80000 0
activate interchiplet
Note over interchiplet: Pair with pending WRITE command.
interchiplet->>SP0: SYNC 2579909
activate SP0
interchiplet->>SP1: SYNC 2579914
activate SP1
deactivate interchiplet

Note over SP0,SP1: Example ends

deactivate SP0
deactivate SP1
```

*interchiplet* does not pair SEND and RECEIVE commands because the communication is handled by Named Pipes. Named Pipes already provide the functionality to synchronize the source and destination. As shown in the above example, the actual data transfer (5) operates after the RESULT command to Simulator Process 1 (4).



## Handle SEND and RECEIVE command

The SEND and RECEIVE commands are used to create a Named Pipe for each pair of communication. The SEND command is used at the source, while the RECEIVE command is used at the destination. The name of the Named Pipe is specified by the source and destination address, which is `buffer{src_x}_{src_y}_{dst_x}_{dst_y}`. For example, `buffer0_0_0_1` means the Pipe to send data from node (0,0) to node (0,1). If the request pipe does not exist, *interchiplet* creates one.

In the above example, Simulator Process 0 wants to send data to Simulator Process 1. *interchiplet* receives the SEND command (1) from Simulator Process 0 and creates the Pipe file. Then, *interchiplet* sends a RESULT command (2) to Simulator Process 0 with the name of the Pipe file. *interchiplet* directly issues the RESULT command (4) to Simulator Process 1 after receiving the RECEIVE command (3) from Simulation Process 1 because the required Pipe file already exists.

## Handle READ and WRITE command

The `cycle` field in the WRITE command presents the time when the source component starts to send data, referenced as `src_cycle`. The `cycle` field in the READ command presents the time when the destination component starts to wait for data to be ready for reading, referenced as `dst_cycle`.

The SYNC command after one WRITE command means the source has finished sending data. The SYNC command after one READ command means the destination has finished reading data. The task or flow in the source and destination can continue after receiving the SYNC command. The execution cycle of the source and destination should be adjusted to the value specified in the cycle field of SYNC commands.

In the above example, Simulator Process 0 sends the WRITE command (6) to *interchiplet* with `src_cycle`, and Simulator Process 1 sends the READ command (7) to *interchiplet* with `dst_cycle`. After pairing the WRITE and READ commands with the same source address, the same destination address, and the same number of bytes, *interchiplet* sends SYNC commands to Simulator Process 0 (8) and Simulator Process 1 (9) with the end cycle of the transaction.

Latency information provides two latency values (`lat_0` and `lat_1`) for one communication transaction:

- `lat_0` means the package latency from the source's view, including propagate latency.
- `lat_1` means the package latency from the destination's view, including the propagate latency and the transmission latency.

The package is injected at `src_cycle`. Hence, the package arrives at the destination at `src_cycle + lat_1`.

The timing sequence is shown below:

```mermaid
sequenceDiagram 
autonumber

participant SP0 as Simulator<br/>Process 0
participant SP1 as Simulator<br/>Process 1

note left of SP0: src_cycle
SP0->>SP1: 
note right of SP1: src_cycle + lat_1
```

The destination does not need further blocking if `dst_cycle` is later than `src_cycle + lat_1`. Otherwise, the destination needs to block till `src_cycle + lat_1`.

In summary,

- The `cycle` of the SYNC command to the WRITE command is `src_cycle + lat_0`.
- The `cycle` of the SYNC command to the READ command is `max(src_cycle + lat_1, dst_cycle)`.

> TODO: Complex descriptor of communication transactions.
