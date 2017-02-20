#ifndef LIBGEODECOMP_MISC_LIKELY_H
#define LIBGEODECOMP_MISC_LIKELY_H


/* Likely/Unlikely macros borrowed from BigMPI via MPICH via ARMCI-MPI */

/* These likely/unlikely macros provide static branch prediction hints to the
 * compiler, if such hints are available.  Simply wrap the relevant expression in
 * the macro, like this:
 *
 * if (unlikely(ptr == NULL)) {
 *     // ... some unlikely code path ...
 * }
 *
 * They should be used sparingly, especially in upper-level code.  It's easy to
 * incorrectly estimate branching likelihood, while the compiler can often do a
 * decent job if left to its own devices.
 *
 * These macros are not namespaced because the namespacing is cumbersome.
 */

/* safety guard for now, add a configure check in the future */
#ifdef __CUDACC__
#  define LGD_UNLIKELY(X_) (X_)
#  define LGD_LIKELY(X_)   (X_)
#else
#  if ( defined(__GNUC__) && (__GNUC__ >= 3) ) || defined(__IBMC__) || defined(__INTEL_COMPILER) || defined(__clang__)
#    define LGD_UNLIKELY(X_) __builtin_expect(!!(X_),0)
#    define LGD_LIKELY(X_)   __builtin_expect(!!(X_),1)
#  else
#    define LGD_UNLIKELY(X_) (X_)
#    define LGD_LIKELY(X_)   (X_)
#  endif
#endif

#endif
