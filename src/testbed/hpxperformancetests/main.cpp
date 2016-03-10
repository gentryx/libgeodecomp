#include <iostream>
#include <libgeodecomp/geometry/partitions/unstructuredstripingpartition.h>
#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/testbed/performancetests/cpubenchmark.h>
#include <libgeodecomp/loadbalancer/oozebalancer.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/parallelization/hpxsimulator.h>

#include <libflatarray/testbed/cpu_benchmark.hpp>
#include <libflatarray/testbed/evaluate.hpp>

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
    static const int ITERATIONS = 10000;
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

    // template<typename HOOD_NEW, typename HOOD_OLD>
    // static void updateLineX(HOOD_NEW& hoodNew, int indexEnd, HOOD_OLD& hoodOld, unsigned /* nanoStep */)
    // {
    // fixme
    //     std::cout << "updateLineX\n";
    // }

    template<typename HOOD>
    void update(const HOOD& hood, int /* nanoStep */)
    {
        *this = hood[hood.index()];

        for (int i = 0; i < ITERATIONS; ++i) {
            cReal = cReal * cReal - cImag * cImag;
            cImag = 2 * cImag * cReal;
        }

        for (auto i = hood.begin(); i != hood.end(); ++i) {
            cReal += hood[i.first()].x * i.second();
            cImag += hood[i.first()].y * i.second();
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

class HPXBusyworkCellIron : public CPUBenchmark
{
public:
    std::string family()
    {
        return "HPXBusyworkCell";
    }

    std::string species()
    {
        return "iron";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<2> dim(rawDim[0], rawDim[1]);
        int steps = rawDim[2];

        typedef HpxSimulator<UnstructuredBusyworkCell, UnstructuredStripingPartition> SimulatorType;

        CellInitializer *init = new CellInitializer(dim, steps);

        SimulatorType sim(
            init,
            std::vector<double>(1, 1.0),
            new TracingBalancer(new OozeBalancer()),
            steps,
            1,
            "hpxperformancetests_HPXBusyworkCellIron");

        double seconds;
        {
            ScopedTimer t(&seconds);
            sim.run();
        }

        double latticeUpdates = 1.0 * dim.prod() * steps;
        double flopsPerLatticeUpdate = UnstructuredBusyworkCell::ITERATIONS * 5 + 4 * 4;
        double gflops = latticeUpdates * flopsPerLatticeUpdate / seconds * 1e-9;

        return gflops;
    }

    std::string unit()
    {
        return "GFLOP/s";
    }
};

int hpx_main(int argc, char **argv)
{
        // fixme: we need tests {update, updateLineX} x {AoS, SoA} x {fine-grained parallelism / no fine-grained parallelism} x {structured, unstructured} x {HPX, OpenMP, CUDA} x {memory bound, compute bound}

    if ((argc < 3) || (argc == 4) || (argc > 5)) {
        std::cerr << "usage: " << argv[0] << " [-n,--name SUBSTRING] REVISION CUDA_DEVICE \n"
                  << "  - optional: only run tests whose name contains a SUBSTRING,\n"
                  << "  - REVISION is purely for output reasons,\n"
                  << "  - CUDA_DEVICE causes CUDA tests to run on the device with the given ID.\n";
        return 1;
    }

    std::string name = "";
    int argumentIndex = 1;
    if (argc == 5) {
        if ((std::string(argv[1]) == "-n") ||
            (std::string(argv[1]) == "--name")) {
            name = std::string(argv[2]);
        }
        argumentIndex = 3;
    }
    std::string revision = argv[argumentIndex + 0];

    std::stringstream s;
    s << argv[argumentIndex + 1];
    int cudaDevice;
    s >> cudaDevice;

    LibFlatArray::evaluate eval(name, revision);
    eval.print_header();

    std::vector<int> size;
    size << 100 << 100 << 100;
    eval(HPXBusyworkCellIron(), size);
    return hpx::finalize();
}


int main(int argc, char **argv)
{
    return hpx::init(argc, argv);
}
