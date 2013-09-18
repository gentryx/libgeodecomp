#ifndef LIBGEODECOMP_TESTBED_PERFORMANCETESTS_BENCHMARK_H
#define LIBGEODECOMP_TESTBED_PERFORMANCETESTS_BENCHMARK_H

#include <libgeodecomp/misc/coord.h>

class Benchmark
{
public:
    virtual std::string order() = 0;
    virtual std::string family() = 0;
    virtual std::string species() = 0;
    virtual double performance(const LibGeoDecomp::Coord<3>& dim) = 0;
    virtual std::string unit() = 0;
    virtual std::string device() = 0;

    double seconds(long long tStart, long long tEnd)
    {
        return (tEnd - tStart) * 0.000001;
    }
};

#endif
