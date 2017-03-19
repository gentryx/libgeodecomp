#include <libgeodecomp/config.h>

// MSVC issues this warning for standard library headers, too, which
// is not really helpful.
#ifdef _MSC_BUILD
#pragma warning( disable : 4514 4710 )
#endif

// include MPI header first to skirt troubles with Intel MPI and standard C library
#ifdef LIBGEODECOMP_WITH_MPI
#include <mpi.h>
#endif

// nvcc complains about some type traits if we don't define this macro
#ifdef __CUDACC__
#define CXXTEST_NO_COPY_CONST
#endif
