#!/bin/bash

set -x -e

cd xerus

cat <<EOF >config.mk
CXX = ${CXX}
COMPATIBILITY = -std=c++17
COMPILE_THREADS = 8                       # Number of threads to use during link time optimization.
HIGH_OPTIMIZATION = TRUE                  # Activates -O3 -march=native and some others
OTHER += -fopenmp

PYTHON3_CONFIG = \`python3-config --cflags --ldflags\`

LOGGING += -D XERUS_LOG_INFO              # Information that is not linked to any unexpected behaviour but might nevertheless be of interest.
LOGGING += -D XERUS_LOGFILE               # Use 'error.log' file instead of cerr
LOGGING += -D XERUS_LOG_ABSOLUTE_TIME     # Print absolute times instead of relative to program time
XERUS_NO_FANCY_CALLSTACK = TRUE           # Show simple callstacks only

BLAS_LIBRARIES = -lopenblas -lgfortran    # Openblas, serial
LAPACK_LIBRARIES = -llapacke -llapack     # Standard Lapack + Lapacke libraries
SUITESPARSE = -lcholmod -lspqr
BOOST_LIBS = -lboost_filesystem

OTHER += -I${PREFIX}/include -I${PREFIX}/lib/python${PY_VER}/site-packages/numpy/core/include/
OTHER += -L${PREFIX}/lib
EOF

cp include/xerus.h ${PREFIX}/include/
cp -r include/xerus ${PREFIX}/include/
cp build/libxerus.so ${PREFIX}/lib/
cp build/libxerus_misc.so ${PREFIX}/lib/

ln -s ${PREFIX}/include/ ${PREFIX}/include/suitesparse
make python
${PYTHON} -m pip install . --no-deps -vv

