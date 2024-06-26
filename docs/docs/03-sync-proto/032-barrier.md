# Barrier

You can find the following topics on this page:

- The sequences of synchronization protocol to handle barrier transactions between processes.
- The algorithm to handle synchronization protocol within *interchiplet*.
- The algorithm to calculate the end cycle of one barrier transaction.

## Command syntax

```
# barrier
BARRIER <src_x> <src_y> <uid> <count>
WRITE <cycle> <src_x> <src_y> <dst_x> <dst_y> <nbytes=1> <desc=0x20000+count>
```

`src_x` and `src_y` present the source address of the process that enters the barrier. `uid` specifies the unique ID of the barrier. `count` specifies the number of processes that enter the barrier when the barrier overflows. Non-zero `count` always overrides the number of the barrier.

The `cycle` field in the WRITE command presents the time when the process enters the barrier.

The figure below shows the relationship between arguments of APIs and commands.

```mermaid
flowchart TB

subgraph barrier
A1[__src_x]
A2[__src_y]
A3[__uid]
A4[__count]
end

subgraph BARRIER command
B1[src_x]
B2[src_y]
B3[uid]
B4[count]
end

subgraph WRITE command
C0[cycle]
C1[src_x]
C2[src_y]
C3[dst_x]
C4[dst_y=0]
C5[nbytes=1]
C6[desc=0x20000+count]
end

A1 -.-> B1 -.-> C1
A2 -.-> B2 -.-> C2
A3 -.-> B3 -.-> C3
A4 -.-> B4 -.-> C6

```

## Command Sequence

One example of the command sequence is shown below:

```mermaid
sequenceDiagram
autonumber

participant interchiplet
participant SP0 as Simulator<br/>Process 0
participant SP1 as Simulator<br/>Process 1
participant SP2 as Simulator<br/>Process 2
participant SP3 as Simulator<br/>Process 3

activate SP0
activate SP1
activate SP2
activate SP3

Note over SP0,SP3: Example starts

SP1->>interchiplet: BARRIER 0 1 255 4
deactivate SP1
activate interchiplet
Note over interchiplet: Register BARRIER command.
deactivate interchiplet

SP0->>interchiplet: BARRIER 0 0 255 4
deactivate SP0
activate interchiplet
Note over interchiplet: Register BARRIER command.
deactivate interchiplet

SP3->>interchiplet: BARRIER 1 1 255 4
deactivate SP3
activate interchiplet
Note over interchiplet: Register BARRIER command.
deactivate interchiplet

SP2->>interchiplet: BARRIER 1 0 255 4
deactivate SP2
activate interchiplet
Note over interchiplet: 1. Register BARRIER command.<br/>2. Barrier overflows.<br/>3. Send RESULT command to<br/>each Simulator Process.
interchiplet->>SP1: RESULT 0
interchiplet->>SP0: RESULT 0
interchiplet->>SP3: RESULT 0
interchiplet->>SP2: RESULT 0
deactivate interchiplet

SP1->>interchiplet: WRITE 2305339 0 1 255 0 1 131076
activate interchiplet
Note over interchiplet: Register WRITE command<br>with the barrier flag.
deactivate interchiplet

SP0->>interchiplet: WRITE 2410745 0 0 255 0 1 131076
activate interchiplet
Note over interchiplet: Register WRITE command<br>with the barrier flag.
deactivate interchiplet

SP3->>interchiplet: WRITE 2330513 1 1 255 0 1 131076
activate interchiplet
Note over interchiplet: Register WRITE command<br>with the barrier flag.
deactivate interchiplet

SP2->>interchiplet: WRITE 2331564 1 0 255 0 1 131076
activate interchiplet
Note over interchiplet: 1. Register BARRIER command<br/>with the barrier flag.<br/>2. Barrier overflows and<br>calculate barrier overflow time.<br/>3. Send SYNC command to<br/>each Simulator Process.
interchiplet->>SP1: SYNC 2411664
activate SP1
interchiplet->>SP0: SYNC 2411659
activate SP0
interchiplet->>SP3: SYNC 2411669
activate SP3
interchiplet->>SP2: SYNC 2411664
activate SP2
deactivate interchiplet

Note over SP0,SP3: Example ends

deactivate SP0
deactivate SP1
deactivate SP2
deactivate SP3
```

## Handle BARRIER Command

*interchiplet* emulates the function of the barrier. The following diagram shows the flow to handle one BARRIER command.

```mermaid
flowchart TB

A(Start)
B[Register BARRIER command]
C{Check whether<br/>barrier overflows}
E[Send RESULT commands<br/>to each pending<br/>BARRIER command]
Z(End)

A-->B-->C--"Yes"-->E-->Z
C--"No"-->Z
```

*interchiplet* response one RESULT command without any result for each process that enters the barrier when the barrier overflows.

> The order of BARRIER does not change by the timing information.

## Handle WRITE Command with the Barrier Flag

In a realistic system, when a process enters a barrier, the process sends one request to a controller, like a mailbox. Then, the process blocks till it receives the acknowledgment from the controller. The location of the controller is configured in Popnet.

The `cycle` field in the WRITE command with the barrier flag presents the time when the source component sends the barrier requirement to the controller in the system, referenced as `src_cycle`. WRITE commands with the barrier flag do not need to pair with READ commands.

The SYNC command after one WRITE command with the barrier flag means the source has received acknowledgment. The task or flow in the source can continue after receiving the SYNC command. The execution cycle of the source should be adjusted to the value specified in the cycle field of SYNC commands.

Latency information provides four latency values (`lat_0`, `lat_1`, `lat_2`, and `lat_3`) for one barrier transaction:

| | From the source's view | From the destination's view |
| ---- | :----: | :----: |
| **Request package** | `lat_0` | `lat_1` |
| **Acknowledgement package** | `lat_2` | `lat_3` |

The request package is injected at `src_cycle`. Hence, the request package arrives at the controller at `src_cycle + lat_1`. Then, when the barrier overflows, the controller sends one acknowledgment package to each source component.

The timing sequence is shown below:

```mermaid
sequenceDiagram 
autonumber

participant SP0 as Simulator<br/>Process 0
participant SP1 as Simulator<br/>Process 1
participant SP2 as Simulator<br/>Process 2

note right of SP2: src_cycle[2]
SP2->>SP0: 
note left of SP0: src_cycle[2] + lat_1[2]

note right of SP1: src_cycle[1]
SP1->>SP0: 
note left of SP0: src_cycle[1] + lat_1[1]

SP0->>SP2: 
note right of SP2: src_cycle[1] + lat_1[1] + lat_3[2]

SP0->>SP1: 
note right of SP1: src_cycle[1] + lat_1[1] + lat_3[1]
```

In summary,

- The barrier overflow time is  `max(src_cycle[i] + lat_1[i])`.
- The `cycle` of the SYNC command to the WRITE command with the barrier flag is `barrier overflow time + lat_3`.
