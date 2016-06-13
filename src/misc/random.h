#ifndef LIBGEODECOMP_MISC_RANDOM_H
#define LIBGEODECOMP_MISC_RANDOM_H

#include <limits>

namespace LibGeoDecomp {

void seedMT(unsigned seed);
unsigned randomMT();

/**
 * LibGeoDecomp's internal wrapper for generating pseudo random
 * numbers.
 */
class Random
{
public:
    static inline unsigned genUnsigned(const unsigned max = std::numeric_limits<unsigned>::max())
    {
        return randomMT() % (unsigned)max;
    }

    static inline double genDouble(const double max = 1.0)
    {
        return (double)genUnsigned() / double(std::numeric_limits<unsigned>::max()) * max;
    }

    static inline void seed(const unsigned newSeed)
    {
        seedMT(newSeed);
    }
};

}

#endif
