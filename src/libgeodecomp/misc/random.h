#ifndef LIBGEODECOMP_MISC_RANDOM_H
#define LIBGEODECOMP_MISC_RANDOM_H

#include <libgeodecomp/misc/limits.h>

namespace LibGeoDecomp {

// Hardwire this warning to off as MSVC would otherwise complain about
// inline functions not being included in object files:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

void seedMT(unsigned seed);
unsigned randomMT();

/**
 * LibGeoDecomp's internal wrapper for generating pseudo random
 * numbers.
 */
class Random
{
public:
    static inline unsigned genUnsigned(const unsigned max = Limits<unsigned>::getMax())
    {
        return randomMT() % (unsigned)max;
    }

    static inline double genDouble(const double max = 1.0)
    {
        return (double)genUnsigned() / double(Limits<unsigned>::getMax()) * max;
    }

    static inline void seed(const unsigned newSeed)
    {
        seedMT(newSeed);
    }
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
