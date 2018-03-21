#ifndef LIBGEODECOMP_MISC_CUDABOOSTWORKAROUND_H
#define LIBGEODECOMP_MISC_CUDABOOSTWORKAROUND_H

// Since CUDA 9.0 __CUDACC_VER__ is no longer supported, but Boost
// prior to 1.65.0 still uses it. We have to redefine the macro before
// Boost gets pulled in, so this header is included in most places
// where HPX is included (which has Boost as a dependency).
#if defined(__CUDACC_VER__) && (__CUDACC_VER_MAJOR__ >= 9)
# undef __CUDACC_VER__
# define __CUDACC_VER__ (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100)
#endif

#endif
