# Cycle

```
CYCLE <cycle>
```

CYCLE command reports the execution cycle of one simulator process to *interchiplet*. This command does not need any response at this time.

After receiving a CYCLE command, one sub-thread updates the execution cycle with the value provided by the CYCLE command if the new execution cycle is greater than the recorded execution cycle. At last, the maximum execution cycles recorded by all CYCLE commands are reported as the total execution cycle.

> TODO: Use the cycle command to build up period synchronization.
