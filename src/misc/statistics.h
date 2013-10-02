#ifndef LIBGEODECOMP_MISC_STATISTICS_H
#define LIBGEODECOMP_MISC_STATISTICS_H

#include <boost/serialization/is_bitwise_serializable.hpp>

namespace LibGeoDecomp {

class Statistics
{
public:
    friend class Typemaps;

    double totalTime;
    double computeTimeInner;
    double computeTimeGhost;
    double patchAcceptersTime;
    double patchProvidersTime;

    template <typename ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & totalTime;
        ar & computeTimeInner;
        ar & computeTimeGhost;
        ar & patchAcceptersTime;
        ar & patchProvidersTime;
    }
};

}

BOOST_IS_BITWISE_SERIALIZABLE(LibGeoDecomp::Statistics)

#endif
