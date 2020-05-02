#=================================================================================================
# Compiler Options
#=================================================================================================
CXX = clang++
COMPATIBILITY = -std=c++14

DEBUG_OPTIMIZATION = TRUE

ACTIVATE_CODE_COVERAGE = TRUE       # Enable coverage tests
BROCKEN_CI = TRUE                   # Enable workaround for brocken CI runner
DEBUG += -D _GLIBCXX_ASSERTIONS     # Activate GLIBCXX assertions

DEBUG += -g                         # Adds debug symbols

#=================================================================================================
# External libraries
#=================================================================================================
BLAS_LIBRARIES = -lopenblas -lgfortran                    # Openblas, serial
LAPACK_LIBRARIES = -llapacke -llapack                     # Standard Lapack + Lapacke libraries
SUITESPARSE = -lcholmod  -lspqr
BOOST_LIBS = -lboost_filesystem -lboost_system
