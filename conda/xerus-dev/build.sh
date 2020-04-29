#!/bin/bash

set -e  # exit when any command fails

cat <<EOF >config.mk
CXX = ${CXX}
COMPATIBILITY = -std=c++17
COMPILE_THREADS = 8                       # Number of threads to use during link time optimization.
HIGH_OPTIMIZATION = TRUE                  # Activates -O3 -march=native and some others
OTHER += -fopenmp

PYTHON3_CONFIG = `python3-config --cflags`
PYTHON3_CONFIG += `python3-config --ldflags`


LOGGING += -D XERUS_LOG_INFO              # Information that is not linked to any unexpected behaviour but might nevertheless be of interest.
LOGGING += -D XERUS_LOGFILE               # Use 'error.log' file instead of cerr
LOGGING += -D XERUS_LOG_ABSOLUTE_TIME     # Print absolute times instead of relative to program time
XERUS_NO_FANCY_CALLSTACK = TRUE           # Show simple callstacks only

INSTALL_LIB_PATH = ${PREFIX}/lib          # Path where to install the libxerus.so shared library.
INSTALL_HEADER_PATH = ${PREFIX}/include   # Path where to install the xerus header files.
INSTALL_PYTHON3_PATH = ${PREFIX}/lib/python${PY_VER}

BLAS_LIBRARIES = -lopenblas -lgfortran    # Openblas, serial
LAPACK_LIBRARIES = -llapacke -llapack     # Standard Lapack + Lapacke libraries
SUITESPARSE = -lcholmod -lspqr
BOOST_LIBS = -lboost_filesystem

BOOST_PYTHON3 = -lboost_python${CONDA_PY}

OTHER+= -L${BUILD_PREFIX}/lib
EOF

test ${CONDA_PY} = ${PY_VER//./}  # this should always be the case
printenv

export CPP_INCLUDE_PATH=${BUILD_PREFIX}/include:${BUILD_PREFIX}/lib/python${PY_VER}/site-packages/numpy/core/include/
export CPLUS_INCLUDE_PATH=${BUILD_PREFIX}/include:${BUILD_PREFIX}/lib/python${PY_VER}/site-packages/numpy/core/include/
export CXX_INCLUDE_PATH=${BUILD_PREFIX}/include:${BUILD_PREFIX}/lib/python${PY_VER}/site-packages/numpy/core/include/
export LIBRARY_PATH=${BUILD_PREFIX}/lib

ln -sfn ${BUILD_PREFIX}/include/ ${BUILD_PREFIX}/include/suitesparse  # overwrite existing symbolic links
mkdir -p ${PREFIX}/lib/python${PY_VER}
mkdir -p ${PREFIX}/include/

make test -j$((${CPU_COUNT}-1))

make install
rm config.mk
