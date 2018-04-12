#ifndef LIBGEODECOMP_TESTBED_PERFORMANCETESTS_LBMSOA_H
#define LIBGEODECOMP_TESTBED_PERFORMANCETESTS_LBMSOA_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/misc/cpubenchmark.h>

#include <vector>

using namespace LibGeoDecomp;

template<typename CELL>
class NoOpInitializer : public SimpleInitializer<CELL>
{
public:
    typedef typename SimpleInitializer<CELL>::Topology Topology;

    NoOpInitializer(
        const Coord<3>& dimensions,
        unsigned steps) :
        SimpleInitializer<CELL>(dimensions, steps)
    {}

    virtual void grid(GridBase<CELL, Topology::DIM>* /* target */)
    {}
};

class LBMSoA : public CPUBenchmark
{
public:
    std::string family();

    std::string species();

    double performance(std::vector<int> rawDim);

    std::string unit();
};

#endif
