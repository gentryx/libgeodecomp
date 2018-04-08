#ifndef LIBGEODECOMP_MISC_LIMITS_H
#define LIBGEODECOMP_MISC_LIMITS_H

// Kill warning 4514 in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <limits>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

/**
 * Portable wrapper for std::numeric_limits as some compilers struggle
 * even with such a basic piece of the STL.
 */
template<typename VALUE>
class Limits
{
public:
    static inline VALUE getMax()
    {
        // Microsoft Visual C++ (MSVC) has max() defined as a macro,
        // so we have to use parentheses to confuse the preprocessor.
        // This form however isn't compatible with g++ 4.6, hence the
        // macro conditional. Finally, defining NOMINMAX is not an
        // option as user code may depend on min()/max() being macros.
#ifdef _MSC_BUILD
        return (std::numeric_limits<VALUE>::max)();
#else
        return std::numeric_limits<VALUE>::max();
#endif
    }

    static inline VALUE getMin()
    {
        // see above.
#ifdef _MSC_BUILD
        return (std::numeric_limits<VALUE>::min)();
#else
        return std::numeric_limits<VALUE>::min();
#endif
    }
};

}

#endif
