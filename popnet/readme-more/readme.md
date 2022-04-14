
# 1 gdb
- emacs
  - M-x gdb
  - M-x gdb-many-windows
- `source cmd`
- `start`


```shell
file ./popnet
set args -A 9 -c 2 -V 1 -B 12 -O 12 -F 4 -L 1000 -T 20000 -r 1 -I ./random_trace/bench -R 0

b sim_router.cc:425
```

# 2 debug 
- 1 main.cc:23 	`network_mess_queue.simulator();`
- 2 mess_queue.cc:35  `mess_queue::simulator()`
  - 2.1 35 while //when mess queue is not empty and simulation deadline has not reach
  - 2.2 58 case ROUTER_: sim_foundation::wsf().receive_ROUTER_message(current_message);
- 3 mess_event.cc:59  `mess_event::mess_event(mess_event& me)`
- 4 sim_foundation.cc:158	`sim_foundation.receive_EVG_message(mess_event mesg)` 
  - //router pipline stage
  - 4.1 163 //循环81个节点
    - 4.1.1 条件断点： b 163 if i==80
  - 4.2 164 inter_network_[i].router_sim_pwr();
- 5 sim_router.cc:629 `sim_router_template::router_sim_pwr()`
  - //routing pipeline stages
  - 5.1 634 	flit_outbuffer();
  - 5.2 640   routing_decision();
- 6 sim_routing.cc:98 `sim_router_template::routing_decision()`
  - 6.1 102 //for injection physical port 0(没跳进去执行？)
  - 6.2 138 //for other physical ports
    - 6.2.1 141 //send back CREDIT message
    - 6.2.2 167 //for HEADER_ flit
    - 6.2.3 183 //for BODY_ or TAIL_ flits
- 7 回到2.1


删除断点简化版：2.1->4.1->2.1  
or  
2.1->4.1->5.2->6.1

# 3 Test
- 2.1 current_message

{time_ = 0, mess_ = ROUTER_, src_ = std::vector of length 0, capacity 0, des_ = std::vector of length 0, capacity 0, pc_ = 0, vc_ = 0, flit_ = {flit_id_ = 0, type_ = HEADER_, sor_addr_ = std::vector of length 0, capacity 0, des_addr_ = std::vector of length 0, capacity 0, start_time_ = 0, finish_time_ = 0, data_ = std::vector of length 0, capacity 0}}

- 4.1 

先跳回160两次：  
160 mess_queue::wm_pointer().add_message(mess_event(mesg.event_start() + PIPE_DELAY_, ROUTER_));

- 5 routing pipeline stages

//stage 5 flit traversal
flit_traversal();
//stage 4 flit output buffer
flit_outbuffer();
//stage 3 switch arbitration
sw_arbitration();
//stage 2, vc arbitration
vc_arbitration();
//stage 1, routing decision
routing_decision();

- 6.2.1 //send back CREDIT message

第一次 161  
mess_queue::wm_pointer().add_message(mess_event(event_time + CREDIT_DELAY_,CREDIT_, address_, cre_add_t, cre_pc_t, j));

以后 148  
if((i % 2) == 0) {
...
}else {
...
}

- 6.2.2 //for HEADER_ flit

先跳到164两次


- flit_t 格式

{flit_id_ = 0, type_ = HEADER_, sor_addr_ = std::vector of length 0, capacity 0, des_addr_ = std::vector of length 0, capacity 0, start_time_ = 0, finish_time_ = 0, data_ = std::vector of length 0, capacity 0}

