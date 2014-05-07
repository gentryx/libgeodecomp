#ifndef LIBGEODECOMP_TESTBED_PERFORMANCETESTS_CPUBENCHMARK_H
#define LIBGEODECOMP_TESTBED_PERFORMANCETESTS_CPUBENCHMARK_H

#include <libflatarray/testbed/cpu_benchmark.hpp>

namespace LibGeoDecomp {

class CPUBenchmark : public LibFlatArray::cpu_benchmark
{
public:
    double performance(std::vector<int> dim)
    {
        Coord<3> c(dim[0], dim[1], dim[2]);
        return performance2(c);
    }

    virtual double performance2(const Coord<3>& dim) = 0;
};

}

#endif
