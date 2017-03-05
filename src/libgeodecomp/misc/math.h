#ifndef LIBGEODECOMP_MISC_MATH_H
#define LIBGEODECOMP_MISC_MATH_H

#ifdef _MSC_BUILD
#  ifndef _USE_MATH_DEFINES
#    define _USE_MATH_DEFINES
#  endif
#endif
#include <cmath>

#if defined(_MSC_VER)
inline int round(double d)
{
    return d >= 0.0 ? static_cast<int>(d + 0.5) : static_cast<int>(d - 0.5);
}
#endif

#endif
