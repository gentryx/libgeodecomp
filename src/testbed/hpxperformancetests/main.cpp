#include <iostream>
#include <libgeodecomp/geometry/partitions/unstructuredstripingpartition.h>
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

    template <class ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & pos;
        ar & vel;
        ar & charge;
    }

private:
    FloatCoord<3> pos;
    FloatCoord<3> vel;
    double charge;
};

LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(ChargedParticle)

class UnstructuredBusyworkCell
{
public:
    static const int C = 4;
    static const int SIGMA = 1;

    class API :
        public APITraits::HasUpdateLineX,
        public APITraits::HasUnstructuredTopology,
        public APITraits::HasSellType<double>,
        public APITraits::HasSellMatrices<1>,
        public APITraits::HasSellC<C>,
        public APITraits::HasSellSigma<SIGMA>
    {};

    explicit UnstructuredBusyworkCell(double x = 0, double y = 0) :
        x(x),
        y(y)
    {}

    template<typename HOOD>
    void update(const HOOD& hood, int nanoStep)
    {
        // fixme
    }

    template<typename HOOD_NEW, typename HOOD_OLD>
    static void updateLineX(HOOD_NEW& hoodNew, int indexEnd, HOOD_OLD& hoodOld, unsigned /* nanoStep */)
    {
        // fixme
    }

    template <class ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & x;
        ar & y;
    }

private:
    double x;
    double y;
};

LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(UnstructuredBusyworkCell)

/**
 * Connects cells in a structure that corresponds to a regular grid of
 * the given width.
 */
class CellInitializer : public SimpleInitializer<UnstructuredBusyworkCell>
{
public:
    CellInitializer(int width) :
        SimpleInitializer<UnstructuredBusyworkCell>(Coord<1>(1000), 100),
        width(width)
    {}

    virtual void grid(GridBase<UnstructuredBusyworkCell, 1> *ret)
    {
        CoordBox<1> boundingBox = ret->boundingBox();
        for (CoordBox<1>::Iterator i = boundingBox.begin(); i != boundingBox.end(); ++i) {
            UnstructuredBusyworkCell cell(i->x() % width, i->x() / width);
            ret->set(*i, cell);
        }
    }

    boost::shared_ptr<Adjacency> getAdjacency(const Region<1>& region) const
    {
        boost::shared_ptr<Adjacency> adjacency;
        return adjacency;
    }

private:
    int width;
};

void runSimulation()
{
    typedef HpxSimulator<UnstructuredBusyworkCell, UnstructuredStripingPartition> SimulatorType;

    CellInitializer *init = new CellInitializer(100);

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
