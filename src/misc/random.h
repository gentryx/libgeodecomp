#ifndef LIBGEODECOMP_MISC_RANDOM_H
#define LIBGEODECOMP_MISC_RANDOM_H

#include <boost/integer_traits.hpp>

#ifndef __CODEGEARC__
#include <boost/random/mersenne_twister.hpp>
#else
#include <stdlib.h>
#endif

namespace LibGeoDecomp {

class Random
{
public:
    static inline unsigned gen_u(const unsigned& max = max_rand)
    {
#ifndef __CODEGEARC__
        return rng() % (unsigned)max;
#else
        return random(max);
#endif
    }

    static inline double gen_d(const double& max = 1.0)
    {
        return (double)gen_u() / (double)max_rand * max;
    }

private:
#ifndef __CODEGEARC__
    static boost::mt19937 rng; 
    static const long long max_rand = boost::integer_traits<unsigned>::const_max;
#else
    static const long long max_rand = boost::integer_traits<int>::const_max;

#endif
};

}

#endif
