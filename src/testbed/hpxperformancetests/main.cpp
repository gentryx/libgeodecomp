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
        public APITraits::HasUnstructuredTopology,
        public APITraits::HasSellType<double>,
        public APITraits::HasSellMatrices<1>,
        public APITraits::HasSellC<C>,
        public APITraits::HasSellSigma<SIGMA>
    {};

    explicit UnstructuredBusyworkCell(double x = 0, double y = 0) :
        x(x),
        y(y),
        cReal(0),
        cImag(0)
    {}

    template<typename HOOD>
    void update(const HOOD& hood, int nanoStep)
    {
        *this = hood[hood.index()];

        for (int i = 0; i < 100; ++i) {
            cReal = cReal * cReal - cImag * cImag;
            cImag = 2 * cImag * cReal;
        }

        for (auto i = hood.begin(); i != hood.end(); ++i) {
            cReal += hood[*i.first()].x;
            cImag += hood[*i.first()].y;
        }
    }

    template <class ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & x;
        ar & y;
        ar & cReal;
        ar & cImag;
    }

private:
    double x;
    double y;
    double cReal;
    double cImag;
};

LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(UnstructuredBusyworkCell)

/**
 * Connects cells in a structure that corresponds to a regular grid of
 * the given width.
 */
class CellInitializer : public SimpleInitializer<UnstructuredBusyworkCell>
{
public:
    CellInitializer(Coord<2> dim, int steps) :
        SimpleInitializer<UnstructuredBusyworkCell>(Coord<1>(dim.prod()), steps),
        dim(dim)
    {}

    virtual void grid(GridBase<UnstructuredBusyworkCell, 1> *ret)
    {
        CoordBox<1> boundingBox = ret->boundingBox();
        for (CoordBox<1>::Iterator i = boundingBox.begin(); i != boundingBox.end(); ++i) {
            UnstructuredBusyworkCell cell(i->x() % width(), i->x() / width());
            ret->set(*i, cell);
        }
    }

    boost::shared_ptr<Adjacency> getAdjacency(const Region<1>& region) const
    {
        boost::shared_ptr<Adjacency> adjacency(new RegionBasedAdjacency);

        for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            int id = i->x();
            int x = id % width();
            int y = id / width();
            int west = y * width() + (width() + x - 1) % width();
            int east = y * width() + (width() + x + 1) % width();
            int north = ((height() + y - 1) % height()) * width() + x;
            int south = ((height() + y + 1) % height()) * width() + x;
            adjacency->insert(id, west);
            adjacency->insert(id, east);
            adjacency->insert(id, north);
            adjacency->insert(id, south);
        }

        return adjacency;
    }

private:
    Coord<2> dim;

    int width() const
    {
        return dim.x();
    }

    int height() const
    {
        return dim.y();
    }
};

void runSimulation()
{
    // fixme: we need tests {update, updateLineX} x {AoS, SoA} x {fine-grained parallelism / no fine-grained parallelism}
    typedef HpxSimulator<UnstructuredBusyworkCell, UnstructuredStripingPartition> SimulatorType;

    int steps = 10000;
    CellInitializer *init = new CellInitializer(Coord<2>(100, 50), steps);

    SimulatorType sim(
        init,
        std::vector<double>(1, 1.0),
        new TracingBalancer(new OozeBalancer()),
        10,
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
