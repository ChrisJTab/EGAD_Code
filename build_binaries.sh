#!/bin/bash

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Always set both flags explicitly (ON or OFF) so that switching between
# build configurations doesn't leave stale cached values. CMake otherwise
# persists cache variables across runs, which means `./build_binaries.sh --hybrid`
# followed by `./build_binaries.sh` would silently still use HYBRID_RECORD_LAYOUT=ON.
HYBRID_FLAG="OFF"
SMALL_FLAG="OFF"
for arg in "$@"; do
    case "$arg" in
        --hybrid)        HYBRID_FLAG="ON" ;;
        --small-records) SMALL_FLAG="ON" ;;
    esac
done
# Distinct build dir per (record layout, record size) so ON and OFF never clobber
# each other. ON = hybrid record layout, used by hybrid_staging and the EGAD modes.
# OFF = stock-Epic layout, REQUIRED for the cpu_only / gpu_only baselines (they are
# value-incorrect on the ON layout). build / build-small are ON; build-off /
# build-small-off are OFF.
if [ "$HYBRID_FLAG" = "ON" ]; then
    BUILD_SUBDIR=$([ "$SMALL_FLAG" = "ON" ] && echo "build-small" || echo "build")
else
    BUILD_SUBDIR=$([ "$SMALL_FLAG" = "ON" ] && echo "build-small-off" || echo "build-off")
fi

ARTIFACT_ROOTDIR=$(pwd)
echo -e "${GREEN}Starting in directory: ${ARTIFACT_ROOTDIR}${NC}"

EPIC_DIR=${ARTIFACT_ROOTDIR}/EGAD
echo -e "${GREEN}Building Epic in ${EPIC_DIR}/${BUILD_SUBDIR}  (HYBRID_RECORD_LAYOUT=${HYBRID_FLAG}, YCSB_SMALL_RECORDS=${SMALL_FLAG})${NC}"
mkdir -p ${EPIC_DIR}/${BUILD_SUBDIR}
cd ${EPIC_DIR}/${BUILD_SUBDIR}
cmake -DCMAKE_BUILD_TYPE=Release \
      -DHYBRID_RECORD_LAYOUT=${HYBRID_FLAG} \
      -DYCSB_SMALL_RECORDS=${SMALL_FLAG} ..
make -j

# STO_DIR=${ARTIFACT_ROOTDIR}/sto
# echo -e "${GREEN}Building STO in ${STO_DIR}${NC}"
# cd ${STO_DIR}
# ./bootstrap.sh
# ./configure
# make -j PROFILE_COUNTERS=1 DEBUG_ABORTS=0 BOUND=7 ADAPTIVE=1 NDEBUG=0 FINE_GRAINED=1 tpcc_bench ycsb_bench

# CARACAL_DIR=${ARTIFACT_ROOTDIR}/caracal
# echo -e "${GREEN}Building Caracal in ${CARACAL_DIR}${NC}"
# mkdir ${CARACAL_DIR}/build
# cd ${CARACAL_DIR}/build
# cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS=-stdlib=libc++ "-DCMAKE_EXE_LINKER_FLAGS=-stdlib=libc++ -lc++abi" ..
# make -j

# Cannot build Aria along side caracal because libc++ conflicts with google-glog
# see: https://github.com/rust-lang/crates-build-env/issues/125
# see also: https://bugs.launchpad.net/ubuntu/+source/google-glog/+bug/1991919

# ARIA_DIR=${ARTIFACT_ROOTDIR}/aria
# echo -e "${GREEN}Building Aria in ${ARIA_DIR}${NC}"
# cd ${ARIA_DIR}
# ./compile.sh
