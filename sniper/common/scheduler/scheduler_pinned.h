#ifndef __SCHEDULER_PINNED_H
#define __SCHEDULER_PINNED_H

#include "scheduler_pinned_base.h"

//commented at 2020-4-14
//所用的调度器
//schedulers used
class SchedulerPinned : public SchedulerPinnedBase
{
   public:
      SchedulerPinned(ThreadManager *thread_manager);

      virtual void threadSetInitialAffinity(thread_id_t thread_id);

   private:
      core_id_t getNextCore(core_id_t core_first);
      core_id_t getFreeCore(core_id_t core_first);

      const int m_interleaving;
      std::vector<bool> m_core_mask;

      core_id_t m_next_core;
};

#endif // __SCHEDULER_ROAMING_H
