CXX = g++

PYTHON2_CONFIG = `python-config --cflags --ldflags`
PYTHON3_CONFIG = `python3-config --cflags --ldflags`

COMPATIBILITY = -std=c++17
COMPILE_THREADS = 8                       # Number of threads to use during link time optimization.
HIGH_OPTIMIZATION = TRUE                  # Activates -O3 -march=native and some others
OTHER += -fopenmp

LOGGING += -D XERUS_LOG_INFO              # Information that is not linked to any unexpected behaviour but might nevertheless be of interest.
LOGGING += -D XERUS_LOGFILE               # Use 'error.log' file instead of cerr
LOGGING += -D XERUS_LOG_ABSOLUTE_TIME     # Print absolute times instead of relative to program time

INSTALL_LIB_PATH = /usr/local/lib         # Path where to install the libxerus.so shared library.
INSTALL_HEADER_PATH = /usr/local/include  # Path where to install the xerus header files.
INSTALL_PYTHON2_PATH = /usr/local/lib/python2.7/site-packages
INSTALL_PYTHON3_PATH = /usr/local/lib/python3.6/site-packages

BLAS_LIBRARIES = -lopenblas -lgfortran    # Openblas, serial
LAPACK_LIBRARIES = -llapacke -llapack     # Standard Lapack + Lapacke libraries
SUITESPARSE = -L/usr/local/lib/suitesparse -lcholmod -lspqr
BOOST_LIBS = -lboost_filesystem
