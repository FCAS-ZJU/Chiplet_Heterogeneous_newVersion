#include "syscall_modeling.h"
#include "sift_assert.h"
#include "globals.h"
#include "threads.h"

#include <iostream>
#include <unistd.h>
#include <syscall.h>

#include "../../../interchiplet/includes/pipe_comm.h"

bool handleAccessMemory(void *arg, Sift::MemoryLockType lock_signal, Sift::MemoryOpType mem_op, uint64_t d_addr, uint8_t* data_buffer, uint32_t data_size)
{
   // Lock memory globally if requested
   // This operation does not occur very frequently, so this should not impact performance
   if (lock_signal == Sift::MemLock)
   {
      PIN_GetLock(&access_memory_lock, 0);
   }

   if (mem_op == Sift::MemRead)
   {
      // The simulator is requesting data from us
      PIN_SafeCopy(data_buffer, reinterpret_cast<void*>(d_addr), data_size);
   }
   else if (mem_op == Sift::MemWrite)
   {
      // The simulator is requesting that we write data back to memory
      PIN_SafeCopy(reinterpret_cast<void*>(d_addr), data_buffer, data_size);
   }
   else
   {
      std::cerr << "Error: invalid memory operation type" << std::endl;
      return false;
   }

   if (lock_signal == Sift::MemUnlock)
   {
      PIN_ReleaseLock(&access_memory_lock);
   }

   return true;
}

InterChiplet::PipeComm global_pipe_comm;

// Emulate all system calls
// Do this as a regular callback (versus syscall enter/exit functions) as those hold the global pin lock
VOID emulateSyscallFunc(THREADID threadid, CONTEXT *ctxt)
{
   // If we haven't set our tid yet, do this now
   if (thread_data[threadid].should_send_threadinfo)
   {
      thread_data[threadid].should_send_threadinfo = false;

      Sift::EmuRequest req;
      Sift::EmuReply res;
      req.setthreadinfo.tid = syscall(__NR_gettid);
      thread_data[threadid].output->Emulate(Sift::EmuTypeSetThreadInfo, req, res);
   }

   ADDRINT syscall_number = PIN_GetContextReg(ctxt, REG_GAX);
   sift_assert(syscall_number < MAX_NUM_SYSCALLS);

   syscall_args_t args;
   #if defined(TARGET_IA32)
      args[0] = PIN_GetContextReg(ctxt, LEVEL_BASE::REG_GBX);
      args[1] = PIN_GetContextReg(ctxt, LEVEL_BASE::REG_GCX);
      args[2] = PIN_GetContextReg(ctxt, LEVEL_BASE::REG_GDX);
      args[3] = PIN_GetContextReg(ctxt, LEVEL_BASE::REG_GSI);
      args[4] = PIN_GetContextReg(ctxt, LEVEL_BASE::REG_GDI);
      args[5] = PIN_GetContextReg(ctxt, LEVEL_BASE::REG_GBP);
   #elif defined(TARGET_INTEL64) || defined(TARGET_IA32E)
      args[0] = PIN_GetContextReg(ctxt, LEVEL_BASE::REG_GDI);
      args[1] = PIN_GetContextReg(ctxt, LEVEL_BASE::REG_GSI);
      args[2] = PIN_GetContextReg(ctxt, LEVEL_BASE::REG_GDX);
      args[3] = PIN_GetContextReg(ctxt, LEVEL_BASE::REG_R10);
      args[4] = PIN_GetContextReg(ctxt, LEVEL_BASE::REG_R8);
      args[5] = PIN_GetContextReg(ctxt, LEVEL_BASE::REG_R9);
   #else
      #error "Unknown target architecture, require either TARGET_IA32 or TARGET_INTEL64 | TARGET_IA32E"
   #endif

   if (thread_data[threadid].icount_reported > 0)
   {
      thread_data[threadid].output->InstructionCount(thread_data[threadid].icount_reported);
      thread_data[threadid].icount_reported = 0;
   }

   // Default: not emulated, override later when needed
   thread_data[threadid].last_syscall_emulated = false;

   if (syscall_number == SYS_write && thread_data[threadid].output)
   {
      int fd = (int)args[0];
      const char *buf = (const char*)args[1];
      size_t count = (size_t)args[2];

      if (count > 0 && (fd == 1 || fd == 2))
         thread_data[threadid].output->Output(fd, buf, count);
   }

   if (KnobEmulateSyscalls.Value() && thread_data[threadid].output)
   {
      switch(syscall_number)
      {
         // Handle SYS_clone child tid capture for proper pthread_join emulation.
         // When the CLONE_CHILD_CLEARTID option is enabled, remember its child_tidptr and
         // then when the thread ends, write 0 to the tid mutex and futex_wake it
         case SYS_clone3_sniper:
         {
            if (args[0] && CLONE_THREAD)
            {
               struct clone_args_sniper* clone3_args = (struct clone_args_sniper*)args[0];
               ADDRINT tidptr = clone3_args->parent_tid;
               PIN_GetLock(&new_threadid_lock, threadid);
               tidptrs.push_back(tidptr);
               PIN_ReleaseLock(&new_threadid_lock);
               /* New thread */
               thread_data[threadid].output->NewThread();
            }
            else
            {
               /* New process */
               // Nothing to do there, handled in fork() -> to check SYS_clone3 is new
            }
            break;
         }
         case SYS_clone:
         {
            if (args[0] & CLONE_THREAD)
            {
               // Store the thread's tid ptr for later use

               #if defined(TARGET_IA32)
                  ADDRINT tidptr = args[2];
               #elif defined(TARGET_INTEL64) || defined(TARGET_IA32E)
                  ADDRINT tidptr = args[3];
               #else
                  #error "Unknown target architecture, require either TARGET_IA32 or TARGET_INTEL64 | TARGET_IA32E"
               #endif

               PIN_GetLock(&new_threadid_lock, threadid);
               tidptrs.push_back(tidptr);
               PIN_ReleaseLock(&new_threadid_lock);
               /* New thread */
               thread_data[threadid].output->NewThread();
            }
            else
            {
               /* New process */
               // Nothing to do there, handled in fork()
            }
            break;
         }

         // System calls not emulated (passed through to OS)
         case SYS_read:
         case SYS_write:
         case SYS_wait4:
            thread_data[threadid].last_syscall_number = syscall_number;
            thread_data[threadid].last_syscall_emulated = false;
            thread_data[threadid].output->Syscall(syscall_number, (char*)args, sizeof(args));
            break;

         // System calls emulated (not passed through to OS)
         case SYS_futex:
         case SYS_sched_yield:
         case SYS_sched_setaffinity:
         case SYS_sched_getaffinity:
         case SYS_nanosleep:
            thread_data[threadid].last_syscall_number = syscall_number;
            thread_data[threadid].last_syscall_emulated = true;
            thread_data[threadid].last_syscall_returnval = thread_data[threadid].output->Syscall(syscall_number, (char*)args, sizeof(args));
            break;

         // System calls sent to Sniper, but also passed through to OS
         case SYS_exit_group:
            thread_data[threadid].output->Syscall(syscall_number, (char*)args, sizeof(args));
            break;

         case InterChiplet::SYSCALL_BARRIER:
         case InterChiplet::SYSCALL_LOCK:
         case InterChiplet::SYSCALL_UNLOCK:
         case InterChiplet::SYSCALL_LAUNCH:
         case InterChiplet::SYSCALL_WAITLAUNCH:
         case InterChiplet::SYSCALL_REMOTE_READ:
         case InterChiplet::SYSCALL_REMOTE_WRITE:
         {
            thread_data[threadid].last_syscall_number = syscall_number;
            thread_data[threadid].last_syscall_emulated=true;

            switch (syscall_number)
            {
               case InterChiplet::SYSCALL_BARRIER:
               {
                  int uid = args[0];
                  int srcX = args[1];
                  int srcY = args[2];
                  int count = args[3];

                  printf("Enter Sniper barrier\n");
                  InterChiplet::barrierSync(srcX, srcY, uid, count);
                  break;
               }
               case InterChiplet::SYSCALL_LOCK:
               {
                  int uid = args[0];
                  int srcX = args[1];
                  int srcY = args[2];

                  printf("Enter Sniper lock\n");
                  InterChiplet::lockSync(srcX, srcY, uid);
                  break;
               }
               case InterChiplet::SYSCALL_UNLOCK:
               {
                  int uid = args[0];
                  int srcX = args[1];
                  int srcY = args[2];

                  printf("Enter Sniper unlock\n");
                  InterChiplet::unlockSync(srcX, srcY, uid);
                  break;
               }
               case InterChiplet::SYSCALL_LAUNCH:
               {
                  int dstX = args[0];
                  int dstY = args[1];
                  int srcX = args[2];
                  int srcY = args[3];

                  printf("Enter Sniper launch\n");
                  InterChiplet::launchSync(srcX, srcY, dstX, dstY);
                  break;
               }
               case InterChiplet::SYSCALL_WAITLAUNCH:
               {
                  int dstX = args[0];
                  int dstY = args[1];
                  int* srcX = (int*)args[2];
                  int* srcY = (int*)args[3];

                  printf("Enter Sniper waitLaunch\n");
                  InterChiplet::waitlaunchSync(srcX, srcY, dstX, dstY);

                  args[2] = *srcX;
                  args[3] = *srcY;
                  break;
               }
               case InterChiplet::SYSCALL_REMOTE_WRITE:
               {
                  int dstX = args[0];
                  int dstY = args[1];
                  int srcX = args[2];
                  int srcY = args[3];
                  int* data = (int*)args[4];
                  int nbytes = args[5];

                  printf("Enter Sniper sendMessage\n");
                  std::string fileName = InterChiplet::sendSync(srcX, srcY, dstX, dstY);
                  global_pipe_comm.write_data(fileName.c_str(), data, nbytes);
                  break;
               }
               case InterChiplet::SYSCALL_REMOTE_READ:
               {
                  int dstX = args[0];
                  int dstY = args[1];
                  int srcX = args[2];
                  int srcY = args[3];
                  int* data = (int*)args[4];
                  int nbytes = args[5];

                  printf("Enter Sniper receiveMessage\n");
                  std::string fileName = InterChiplet::receiveSync(srcX, srcY, dstX, dstY);
                  global_pipe_comm.read_data(fileName.c_str(), data, nbytes);
                  break;
               }
            }

            fflush(stdout);

            thread_data[threadid].last_syscall_returnval = 1;
            thread_data[threadid].output->Syscall(syscall_number, (char *)args, sizeof(args));
            break;
         }
      }
   }
}

static VOID syscallEntryCallback(THREADID threadid, CONTEXT *ctxt, SYSCALL_STANDARD syscall_standard, VOID *v)
{
   if (!thread_data[threadid].last_syscall_emulated)
   {
      return;
   }

   PIN_SetSyscallNumber(ctxt, syscall_standard, SYS_getpid);
}

static VOID syscallExitCallback(THREADID threadid, CONTEXT *ctxt, SYSCALL_STANDARD syscall_standard, VOID *v)
{
   if (!thread_data[threadid].last_syscall_emulated)
   {
      return;
   }

   PIN_SetContextReg(ctxt, REG_GAX, thread_data[threadid].last_syscall_returnval);
   thread_data[threadid].last_syscall_emulated = false;
}

void initSyscallModeling()
{
   PIN_AddSyscallEntryFunction(syscallEntryCallback, 0);
   PIN_AddSyscallExitFunction(syscallExitCallback, 0);
}
