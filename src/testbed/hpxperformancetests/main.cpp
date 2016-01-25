#include <iostream>
#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>
#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/testbed/performancetests/cpubenchmark.h>
#include <libgeodecomp/loadbalancer/oozebalancer.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/parallelization/hpxsimulator.h>

using namespace LibGeoDecomp;

class ChargedParticle
{
public:
    template<typename HOOD>
    void update(const HOOD& hood, int nanoStep)
    {
        // fixme
    }

private:
    FloatCoord<3> pos;
    FloatCoord<3> vel;
    double charge;
};

class UnstructuredBusyworkCell
{
public:
    template<typename HOOD>
    void update(const HOOD& hood, int nanoStep)
    {
        // fixme
    }

    template <class ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & d;
    }

private:
    double d;
};

LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(UnstructuredBusyworkCell)

class CellInitializer : public SimpleInitializer<UnstructuredBusyworkCell>
{
public:
    CellInitializer() : SimpleInitializer<UnstructuredBusyworkCell>(Coord<2>(160, 90), 800)
    {}

    virtual void grid(GridBase<UnstructuredBusyworkCell, 2> *ret)
    {
        // fixme
    }
};

void runSimulation()
{
    typedef HpxSimulator<UnstructuredBusyworkCell, RecursiveBisectionPartition<2> > SimulatorType;

    CellInitializer *init = new CellInitializer();

    SimulatorType sim(
        init,
        std::vector<double>(1, 1.0),
        new TracingBalancer(new OozeBalancer()),
        100,
        1);

    sim.run();
}

int hpx_main()
{
    runSimulation();
    return hpx::finalize();
}


int main(int argc, char **argv)
{
    return hpx::init(argc, argv);
}
