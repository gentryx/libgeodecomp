#ifndef LIBGEODECOMP_PARALLELIZATION_MOCKSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_MOCKSIMULATOR_H

#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/monolithicsimulator.h>
#include <libgeodecomp/storage/grid.h>

namespace LibGeoDecomp {

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

}

#endif
