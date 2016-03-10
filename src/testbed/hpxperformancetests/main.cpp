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

template<typename ADDITIONAL_API>
class UnstructuredBusyworkCellBase
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
        public APITraits::HasSellSigma<SIGMA>,
        public ADDITIONAL_API
    {};

    inline
    explicit UnstructuredBusyworkCellBase(double x = 0, double y = 0) :
        x(x),
        y(y),
        cReal(0),
        cImag(0)
    {}

    template<typename HOOD>
    inline void update(const HOOD& hood, int /* nanoStep */)
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
    inline void serialize(ARCHIVE& ar, unsigned)
    {
        ar & x;
        ar & y;
        ar & cReal;
        ar & cImag;
    }

protected:
    double x;
    double y;
    double cReal;
    double cImag;
};

class EmptyAPI
{};

class APIWithUpdateLineX :
    public APITraits::HasUpdateLineX
{};


class UnstructuredBusyworkCell : public UnstructuredBusyworkCellBase<EmptyAPI>
{
public:
    inline
    explicit UnstructuredBusyworkCell(double x = 0, double y = 0) :
        UnstructuredBusyworkCellBase<EmptyAPI>(x, y)
    {}
};

class UnstructuredBusyworkCellWithUpdateLineX : public UnstructuredBusyworkCellBase<APIWithUpdateLineX>
{
public:
    inline
    explicit UnstructuredBusyworkCellWithUpdateLineX(double x = 0, double y = 0) :
        UnstructuredBusyworkCellBase<APIWithUpdateLineX>(x, y)
    {}

    template<typename HOOD_NEW, typename HOOD_OLD>
    inline static void updateLineX(HOOD_NEW& hoodNew, int indexEnd, HOOD_OLD& hoodOld, unsigned /* nanoStep */)
    {
        for (; hoodOld.index() < indexEnd; ++hoodOld) {
            UnstructuredBusyworkCellWithUpdateLineX& self = hoodNew[hoodOld.index()];
            self = hoodOld[hoodOld.index()];

            for (int i = 0; i < ITERATIONS; ++i) {
                self.cReal = self.cReal * self.cReal - self.cImag * self.cImag;
                self.cImag = 2 * self.cImag * self.cReal;
            }

            for (auto i = hoodOld.begin(); i != hoodOld.end(); ++i) {
                self.cReal += hoodOld[i.first()].x * i.second();
                self.cImag += hoodOld[i.first()].y * i.second();
            }
        }
    }
};

LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(UnstructuredBusyworkCell)
LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(UnstructuredBusyworkCellWithUpdateLineX)




/**
 * Connects cells in a structure that corresponds to a regular grid of
 * the given width.
 */
template<typename CELL_TYPE>
class CellInitializer : public SimpleInitializer<CELL_TYPE>
{
public:
    CellInitializer(Coord<2> dim, int steps) :
        SimpleInitializer<CELL_TYPE>(Coord<1>(dim.prod()), steps),
        dim(dim)
    {}

    virtual void grid(GridBase<CELL_TYPE, 1> *ret)
    {
        CoordBox<1> boundingBox = ret->boundingBox();
        for (CoordBox<1>::Iterator i = boundingBox.begin(); i != boundingBox.end(); ++i) {
            CELL_TYPE cell(i->x() % width(), i->x() / width());
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

template<typename CELL>
class HPXBusyworkCellTest : public CPUBenchmark
{
public:
    std::string family()
    {
        return "HPXBusyworkCell";
    }

    std::string unit()
    {
        return "GFLOP/s";
    }

    double performance(std::vector<int> rawDim)
    {
        Coord<2> dim(rawDim[0], rawDim[1]);
        int steps = rawDim[2];

        typedef HpxSimulator<CELL, UnstructuredStripingPartition> SimulatorType;

        CellInitializer<CELL> *init = new CellInitializer<CELL>(dim, steps);

        SimulatorType sim(
            init,
            std::vector<double>(1, 1.0),
            new TracingBalancer(new OozeBalancer()),
            steps,
            1,
            "hpxperformancetests_HPXBusyworkCellTest_" + species() + dim.toString());

        double seconds;
        {
            ScopedTimer t(&seconds);
            sim.run();
        }

        double latticeUpdates = 1.0 * dim.prod() * steps;
        double flopsPerLatticeUpdate = CELL::ITERATIONS * 5 + 4 * 4;
        double gflops = latticeUpdates * flopsPerLatticeUpdate / seconds * 1e-9;

        return gflops;
    }
};

class HPXBusyworkCellIron : public HPXBusyworkCellTest<UnstructuredBusyworkCell>
{
public:
    std::string species()
    {
        return "iron";
    }
};

class HPXBusyworkCellSilver : public HPXBusyworkCellTest<UnstructuredBusyworkCellWithUpdateLineX>
{
public:
    std::string species()
    {
        return "silver";
    }
};

int hpx_main(int argc, char **argv)
{
    // fixme: we need tests {update, updateLineX} x {AoS, SoA} x {fine-grained parallelism / no fine-grained parallelism} x {structured, unstructured} x {HPX, OpenMP, CUDA} x {memory bound, compute bound}:
    // fixme: same for functional tests

    // Function    | Memory | Parallelism | Grid         | Threading | Model         | TestClass
    // ----------- | ------ | ----------- | ------------ | --------- | ------------- | ---------
    // update      | AoS    | coarse      | structured   | OpenMP    | compute-bound | 
    // updateLineX | AoS    | coarse      | structured   | OpenMP    | compute-bound | 
    // update      | SoA    | coarse      | structured   | OpenMP    | compute-bound | *1
    // updateLineX | SoA    | coarse      | structured   | OpenMP    | compute-bound | 
    // update      | AoS    | fine        | structured   | OpenMP    | compute-bound | 
    // updateLineX | AoS    | fine        | structured   | OpenMP    | compute-bound | 
    // update      | SoA    | fine        | structured   | OpenMP    | compute-bound | *1
    // updateLineX | SoA    | fine        | structured   | OpenMP    | compute-bound | 
    // update      | AoS    | coarse      | unstructured | OpenMP    | compute-bound | 
    // updateLineX | AoS    | coarse      | unstructured | OpenMP    | compute-bound | 
    // update      | SoA    | coarse      | unstructured | OpenMP    | compute-bound | *1
    // updateLineX | SoA    | coarse      | unstructured | OpenMP    | compute-bound | 
    // update      | AoS    | fine        | unstructured | OpenMP    | compute-bound | 
    // updateLineX | AoS    | fine        | unstructured | OpenMP    | compute-bound | 
    // update      | SoA    | fine        | unstructured | OpenMP    | compute-bound | *1
    // updateLineX | SoA    | fine        | unstructured | OpenMP    | compute-bound | 
    // update      | AoS    | coarse      | structured   | HPX       | compute-bound | 
    // updateLineX | AoS    | coarse      | structured   | HPX       | compute-bound | 
    // update      | SoA    | coarse      | structured   | HPX       | compute-bound | *1
    // updateLineX | SoA    | coarse      | structured   | HPX       | compute-bound | 
    // update      | AoS    | fine        | structured   | HPX       | compute-bound | 
    // updateLineX | AoS    | fine        | structured   | HPX       | compute-bound | 
    // update      | SoA    | fine        | structured   | HPX       | compute-bound | *1
    // updateLineX | SoA    | fine        | structured   | HPX       | compute-bound | 
    // update      | AoS    | coarse      | unstructured | HPX       | compute-bound | HPXBusyworkCellIron
    // updateLineX | AoS    | coarse      | unstructured | HPX       | compute-bound | HPXBusyworkCellSilver
    // update      | SoA    | coarse      | unstructured | HPX       | compute-bound | *1
    // updateLineX | SoA    | coarse      | unstructured | HPX       | compute-bound | fixme4 platinum
    // update      | AoS    | fine        | unstructured | HPX       | compute-bound | fixme1 bronze
    // updateLineX | AoS    | fine        | unstructured | HPX       | compute-bound | fixme3 gold
    // update      | SoA    | fine        | unstructured | HPX       | compute-bound | *1
    // updateLineX | SoA    | fine        | unstructured | HPX       | compute-bound | fixme5 titanium
    // update      | AoS    | coarse      | structured   | CUDA      | compute-bound | 
    // updateLineX | AoS    | coarse      | structured   | CUDA      | compute-bound | 
    // update      | SoA    | coarse      | structured   | CUDA      | compute-bound | *1
    // updateLineX | SoA    | coarse      | structured   | CUDA      | compute-bound | 
    // update      | AoS    | fine        | structured   | CUDA      | compute-bound | 
    // updateLineX | AoS    | fine        | structured   | CUDA      | compute-bound | 
    // update      | SoA    | fine        | structured   | CUDA      | compute-bound | *1
    // updateLineX | SoA    | fine        | structured   | CUDA      | compute-bound | 
    // update      | AoS    | coarse      | unstructured | CUDA      | compute-bound | 
    // updateLineX | AoS    | coarse      | unstructured | CUDA      | compute-bound | 
    // update      | SoA    | coarse      | unstructured | CUDA      | compute-bound | *1
    // updateLineX | SoA    | coarse      | unstructured | CUDA      | compute-bound | 
    // update      | AoS    | fine        | unstructured | CUDA      | compute-bound | 
    // updateLineX | AoS    | fine        | unstructured | CUDA      | compute-bound | 
    // update      | SoA    | fine        | unstructured | CUDA      | compute-bound | *1
    // updateLineX | SoA    | fine        | unstructured | CUDA      | compute-bound | 
    // update      | AoS    | coarse      | structured   | OpenMP    | memory-bound  | 
    // updateLineX | AoS    | coarse      | structured   | OpenMP    | memory-bound  | 
    // update      | SoA    | coarse      | structured   | OpenMP    | memory-bound  | *1
    // updateLineX | SoA    | coarse      | structured   | OpenMP    | memory-bound  | 
    // update      | AoS    | fine        | structured   | OpenMP    | memory-bound  | 
    // updateLineX | AoS    | fine        | structured   | OpenMP    | memory-bound  | 
    // update      | SoA    | fine        | structured   | OpenMP    | memory-bound  | *1
    // updateLineX | SoA    | fine        | structured   | OpenMP    | memory-bound  | 
    // update      | AoS    | coarse      | unstructured | OpenMP    | memory-bound  | 
    // updateLineX | AoS    | coarse      | unstructured | OpenMP    | memory-bound  | 
    // update      | SoA    | coarse      | unstructured | OpenMP    | memory-bound  | *1
    // updateLineX | SoA    | coarse      | unstructured | OpenMP    | memory-bound  | 
    // update      | AoS    | fine        | unstructured | OpenMP    | memory-bound  | 
    // updateLineX | AoS    | fine        | unstructured | OpenMP    | memory-bound  | 
    // update      | SoA    | fine        | unstructured | OpenMP    | memory-bound  | *1
    // updateLineX | SoA    | fine        | unstructured | OpenMP    | memory-bound  | 
    // update      | AoS    | coarse      | structured   | HPX       | memory-bound  | 
    // updateLineX | AoS    | coarse      | structured   | HPX       | memory-bound  | 
    // update      | SoA    | coarse      | structured   | HPX       | memory-bound  | *1
    // updateLineX | SoA    | coarse      | structured   | HPX       | memory-bound  | 
    // update      | AoS    | fine        | structured   | HPX       | memory-bound  | 
    // updateLineX | AoS    | fine        | structured   | HPX       | memory-bound  | 
    // update      | SoA    | fine        | structured   | HPX       | memory-bound  | *1
    // updateLineX | SoA    | fine        | structured   | HPX       | memory-bound  | 
    // update      | AoS    | coarse      | unstructured | HPX       | memory-bound  | 
    // updateLineX | AoS    | coarse      | unstructured | HPX       | memory-bound  | 
    // update      | SoA    | coarse      | unstructured | HPX       | memory-bound  | *1
    // updateLineX | SoA    | coarse      | unstructured | HPX       | memory-bound  | 
    // update      | AoS    | fine        | unstructured | HPX       | memory-bound  | 
    // updateLineX | AoS    | fine        | unstructured | HPX       | memory-bound  | 
    // update      | SoA    | fine        | unstructured | HPX       | memory-bound  | *1
    // updateLineX | SoA    | fine        | unstructured | HPX       | memory-bound  | 
    // update      | AoS    | coarse      | structured   | CUDA      | memory-bound  | 
    // updateLineX | AoS    | coarse      | structured   | CUDA      | memory-bound  | 
    // update      | SoA    | coarse      | structured   | CUDA      | memory-bound  | *1
    // updateLineX | SoA    | coarse      | structured   | CUDA      | memory-bound  | 
    // update      | AoS    | fine        | structured   | CUDA      | memory-bound  | 
    // updateLineX | AoS    | fine        | structured   | CUDA      | memory-bound  | 
    // update      | SoA    | fine        | structured   | CUDA      | memory-bound  | *1
    // updateLineX | SoA    | fine        | structured   | CUDA      | memory-bound  | 
    // update      | AoS    | coarse      | unstructured | CUDA      | memory-bound  | 
    // updateLineX | AoS    | coarse      | unstructured | CUDA      | memory-bound  | 
    // update      | SoA    | coarse      | unstructured | CUDA      | memory-bound  | *1
    // updateLineX | SoA    | coarse      | unstructured | CUDA      | memory-bound  | 
    // update      | AoS    | fine        | unstructured | CUDA      | memory-bound  | 
    // updateLineX | AoS    | fine        | unstructured | CUDA      | memory-bound  | 
    // update      | SoA    | fine        | unstructured | CUDA      | memory-bound  | *1
    // updateLineX | SoA    | fine        | unstructured | CUDA      | memory-bound  |
    //
    // *1) left out as deemed impratical

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

    std::vector<Coord<3> > sizes;
    sizes << Coord<3>( 10,  10, 10000)
          << Coord<3>(100, 100,   100);

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(HPXBusyworkCellIron(), toVector(sizes[i]));
    }

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(HPXBusyworkCellSilver(), toVector(sizes[i]));
    }

    return hpx::finalize();
}


int main(int argc, char **argv)
{
    return hpx::init(argc, argv);
}
