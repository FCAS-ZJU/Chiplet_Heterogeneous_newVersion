# This file is auto-generated, changes made to it will be lost. Please edit makebuildscripts.py instead.

HOOKS_DIR="${BENCHMARKS_ROOT}/tools/hooks"
HOOKS_CC="gcc"
HOOKS_CXX="g++"
HOOKS_FC="f95"
LD="g++"
HOOKS_CFLAGS="-I${HOOKS_DIR} -I${GRAPHITE_ROOT}/include"
HOOKS_CXXFLAGS="${HOOKS_CFLAGS}"
HOOKS_LDFLAGS="-uparmacs_roi_end -uparmacs_roi_start -L${HOOKS_DIR} -lhooks_base -lrt -pthread"
HOOKS_LDFLAGS_NOROI="-uparmacs_roi_end -uparmacs_roi_start -L${HOOKS_DIR} -lhooks_base_noroi -lrt -pthread"
HOOKS_LDFLAGS_DYN="-uparmacs_roi_end -uparmacs_roi_start -L${HOOKS_DIR} -lhooks_base -lrt -pthread"
HOOKS_LD_LIBRARY_PATH=""
