#!/bin/bash
ENVNAME="${1:-${CONDA_DEFAULT_ENV}}"
BRANCHNAME="${2:-conda}"
set -e

if [ -z "${ENVNAME}" ];
then
    echo "Usage: bash install.sh [<environment>] [<branch>]"
    echo "       If <environment> is not provided the current conda environment will be used."
    echo "       If <branch> is not provided the conda branch will be used."
    exit 0
fi

eval "$(conda shell.bash hook)"

if [ -z "$(conda info --envs | grep ${ENVNAME})" ];
then
    echo "Environment ${ENVNAME} does not exist."
    read -p "Create new conda environment '${ENVNAME}' (y/n)? " answer
    case ${answer:0:1} in
        y|Y )
        conda create -n ${ENVNAME} -c conda-forge 'python=3.8' python_abi gxx_linux-64 make 'pip>=18.1' numpy openblas suitesparse lapack liblapacke boost-cpp libgomp scipy matplotlib rich
        ;;
        * )
        exit 0;
        ;;
    esac
fi

echo "Installing XERUS into environment: ${ENVNAME}"

conda activate ${ENVNAME}
conda install -c conda-forge python pip python_abi gxx_linux-64 make numpy openblas suitesparse lapack liblapacke boost-cpp libgomp scipy matplotlib rich

NUMPY=${CONDA_PREFIX}/lib/python3.8/site-packages/numpy
CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++

cd /tmp
TEMPDIR=$(mktemp -d)
cd ${TEMPDIR}
git clone --recurse-submodules https://github.com/libxerus/xerus.git --branch ${BRANCHNAME}
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

OTHER += -I${CONDA_PREFIX}/include -I${NUMPY}/core/include/
OTHER += -L${CONDA_PREFIX}/lib
EOF

ln -s ${CONDA_PREFIX}/include/ ${CONDA_PREFIX}/include/suitesparse
make python
python -m pip install . --no-deps -vv

cd ../..
rm -rf ${TEMPDIR}/xerus
