#ifndef LIBGEODECOMP_PARALLELIZATION_MOCKSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_MOCKSIMULATOR_H

#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/monolithicsimulator.h>
#include <libgeodecomp/storage/grid.h>

// Kill some warnings in system headers:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4548 4710 4711 4820 4996 )
#endif

#include <string>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

// Hardwire this warning to off as MSVC would otherwise complain about
// inline functions not being included in object files. Also, padding
// is fine.
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4820 )
#endif

/**
 * Test class which records some of its API calls. Used for unit tests.
 */
class MockSimulator : public MonolithicSimulator<TestCell<2> >
{
public:
    explicit MockSimulator(Initializer<TestCell<2> > *init) :
        MonolithicSimulator<TestCell<2> >(init)
    {}

    ~MockSimulator()
    {
        events += "deleted\n";
    }

    void step()
    {}

    void run() {}

    Grid<TestCell<2> > *getGrid()
    {
        return 0;
    }

    std::vector<Chronometer> gatherStatistics()
    {
        return std::vector<Chronometer>(1);
    }

    static std::string events;
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
