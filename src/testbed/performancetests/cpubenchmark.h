#ifndef LIBGEODECOMP_TESTBED_PERFORMANCETESTS_CPUBENCHMARK_H
#define LIBGEODECOMP_TESTBED_PERFORMANCETESTS_CPUBENCHMARK_H

#include <libflatarray/testbed/cpu_benchmark.hpp>

namespace LibGeoDecomp {

class CPUBenchmark : public LibFlatArray::cpu_benchmark
{
public:
    double performance(int dim[3])
    {
        Coord<3> c(dim[0], dim[1], dim[2]);
        return performance(c);
    }

    virtual double performance(const Coord<3>& dim) = 0;
};

}

#endif
