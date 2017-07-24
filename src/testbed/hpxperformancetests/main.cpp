#include <hpx/hpx_init.hpp>

#include <iostream>
#include <libgeodecomp/geometry/partitions/unstructuredstripingpartition.h>
#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/loadbalancer/oozebalancer.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/parallelization/hpxdataflowsimulator.h>
#include <libgeodecomp/parallelization/hpxsimulator.h>

#include <libflatarray/short_vec.hpp>
#include <libflatarray/testbed/cpu_benchmark.hpp>
#include <libflatarray/testbed/evaluate.hpp>

#include "../performancetests/cpubenchmark.h"

using namespace LibGeoDecomp;

template<typename ADDITIONAL_API>
class UnstructuredBusyworkCellBase
{
public:
    LIBFLATARRAY_ACCESS

    static const int ITERATIONS = 10000;
    static const int C = 4;
    static const int SIGMA = 1;

    class API :
        public APITraits::HasUnstructuredTopology,
        public APITraits::HasSellType<double>,
        public APITraits::HasSellMatrices<1>,
        public APITraits::HasSellC<C>,
        public APITraits::HasSellSigma<SIGMA>,
        public APITraits::HasThreadedUpdate<8>,
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
            double cRealOld = cReal;
            cReal = cReal * cReal - cImag * cImag;
            cImag = 2 * cImag * cRealOld;
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

class APIWithSoAAndUpdateLineX :
    public APITraits::HasSoA,
    public APITraits::HasUpdateLineX,
    public LibFlatArray::api_traits::has_default_1d_sizes
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
        for (; hoodOld.index() < indexEnd; ++hoodNew, ++hoodOld) {
            UnstructuredBusyworkCellWithUpdateLineX& self = *hoodNew;
            self = hoodOld[hoodOld.index()];

            for (int i = 0; i < ITERATIONS; ++i) {
                double cRealOld = self.cReal;
                self.cReal = self.cReal * self.cReal - self.cImag * self.cImag;
                self.cImag = 2 * self.cImag * cRealOld;
            }

            for (auto i = hoodOld.begin(); i != hoodOld.end(); ++i) {
                self.cReal += hoodOld[i.first()].x * i.second();
                self.cImag += hoodOld[i.first()].y * i.second();
            }
        }
    }
};

class UnstructuredBusyworkCellWithSoAAndUpdateLineX : public UnstructuredBusyworkCellBase<APIWithSoAAndUpdateLineX>
{
public:
    inline
    explicit UnstructuredBusyworkCellWithSoAAndUpdateLineX(double x = 0, double y = 0) :
        UnstructuredBusyworkCellBase<APIWithSoAAndUpdateLineX>(x, y)
    {}

    template<typename HOOD_NEW, typename HOOD_OLD>
    inline static void updateLineX(HOOD_NEW& hoodNew, int indexEnd, HOOD_OLD& hoodOld, unsigned /* nanoStep */)
    {
        typedef LibFlatArray::short_vec<double, C> ShortVec;

        for (; hoodNew.index() < indexEnd; hoodNew += C, ++hoodOld) {
            ShortVec x = &hoodOld->x();
            ShortVec y = &hoodOld->y();
            ShortVec cReal = &hoodOld->cReal();
            ShortVec cImag = &hoodOld->cImag();

            for (int i = 0; i < ITERATIONS; ++i) {
                ShortVec cRealOld = cReal;
                cReal = cReal * cReal - cImag * cImag;
                cImag = ShortVec(2.0) * cImag * cRealOld;
            }

            for (const auto& j: hoodOld.weights(0)) {
                ShortVec weights;
                ShortVec otherX;
                ShortVec otherY;
                weights.load_aligned(j.second());
                otherX.gather(&hoodOld->x(), j.first());
                otherY.gather(&hoodOld->y(), j.first());
                cReal += otherX * weights;
                cImag += otherY * weights;
            }

            &hoodNew->x() << x;
            &hoodNew->y() << y;
            &hoodNew->cReal() << cReal;
            &hoodNew->cImag() << cImag;
        }
    }
};

LIBFLATARRAY_REGISTER_SOA(
    UnstructuredBusyworkCellWithSoAAndUpdateLineX,
    ((double)(x))
    ((double)(y))
    ((double)(cReal))
    ((double)(cImag)) )

LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(UnstructuredBusyworkCell)
LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(UnstructuredBusyworkCellWithUpdateLineX)
LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(UnstructuredBusyworkCellWithSoAAndUpdateLineX)

/**
 * Connects cells in a structure that corresponds to a regular grid of
 * the given width.
 */
template<typename CELL_TYPE>
class CellInitializer : public SimpleInitializer<CELL_TYPE>
{
public:
    using typename SimpleInitializer<CELL_TYPE>::AdjacencyPtr;

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

    AdjacencyPtr getAdjacency(const Region<1>& region) const
    {
        AdjacencyPtr adjacency(new RegionBasedAdjacency);

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

template<typename CELL, bool ENABLE_COARSE_GRAINED_PARALLELISM>
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
            ENABLE_COARSE_GRAINED_PARALLELISM,
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

class HPXBusyworkCellIron : public HPXBusyworkCellTest<UnstructuredBusyworkCell, true>
{
public:
    std::string species()
    {
        return "iron";
    }
};

class HPXBusyworkCellSilver : public HPXBusyworkCellTest<UnstructuredBusyworkCellWithUpdateLineX, true>
{
public:
    std::string species()
    {
        return "silver";
    }
};

class HPXBusyworkCellPlatinum : public HPXBusyworkCellTest<UnstructuredBusyworkCellWithSoAAndUpdateLineX, true>
{
public:
    std::string species()
    {
        return "platinum";
    }
};

class HPXBusyworkCellBronze : public HPXBusyworkCellTest<UnstructuredBusyworkCell, false>
{
public:
    std::string species()
    {
        return "bronze";
    }
};

class HPXBusyworkCellGold : public HPXBusyworkCellTest<UnstructuredBusyworkCellWithUpdateLineX, false>
{
public:
    std::string species()
    {
        return "gold";
    }
};

class HPXBusyworkCellTitanium : public HPXBusyworkCellTest<UnstructuredBusyworkCellWithSoAAndUpdateLineX, false>
{
public:
    std::string species()
    {
        return "titanium";
    }
};

class MessageType
{
public:
    inline MessageType(int step = -1, int id = -1, std::size_t messageSize = 69) :
        step(step),
        id(id),
        dummyData(std::vector<int>(messageSize, -1))
    {
        for (std::size_t i = 0; i < messageSize; ++i) {
            dummyData[i] = step * i;
        }
    }

    template<typename ARCHIVE>
    void serialize(ARCHIVE& archive, int)
    {
        archive & step;
        archive & id;
        archive & dummyData;
    }

    int step;
    int id;
    std::vector<int> dummyData;
};

LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(MessageType)

class DataflowTestModel
{
public:
    class API :
        public APITraits::HasUnstructuredTopology,
        public APITraits::HasCustomMessageType<MessageType>
    {};

    inline DataflowTestModel(int id = 0, int step = 0, std::size_t messageSize = 0, const std::vector<int>& neighbors = std::vector<int>()) :
        id(id),
        step(step),
        messageSize(messageSize),
        neighbors(neighbors)
    {}

    template<typename NEIGHBORHOOD, typename EVENT>
    inline void update(NEIGHBORHOOD& hood, const EVENT& event)
    {
        if (step > 0) {
            if (neighbors != hood.neighbors()) {
                std::cerr << "found bad neighbors list on cell " << id << "\n"
                          << " got: " << hood.neighbors() << "\n"
                          << " want: " << neighbors << "\n";
            }

            for (auto&& i: hood.neighbors()) {
                if (hood[i].id != i) {
                    std::cout << "cell id " << id << " saw bad ID on cell " << i
                              << " (saw " << hood[i].id << ", expected: " << i << ")\n";
                }
                if (hood[i].step != step) {
                    std::cout << "cell id " << id << " saw bad time step on cell " << i
                              << " (saw " << hood[i].step << ", expected: " << step << ")\n";
                }
                if (hood[i].dummyData.size() != messageSize) {
                    std::cout << "cell id " << id << " saw bad dummy data on  cell " << i << "\n";
                }
            }
        }

        ++step;

        for (auto&& i: hood.neighbors()) {
            hood.send(i, MessageType(step, id, messageSize));
        }

    }

private:
    int id;
    int step;
    std::size_t messageSize;
    std::vector<int> neighbors;
};

REGISTER_CELLCOMPONENT(DataflowTestModel, MessageType, fixme)


class DataflowTestInitializer : public Initializer<DataflowTestModel>
{
public:
    using typename Initializer<DataflowTestModel>::AdjacencyPtr;

    DataflowTestInitializer(const Coord<2>& dim, int myMaxSteps, int messageSize = 27) :
        dim(dim),
        myMaxSteps(myMaxSteps),
        messageSize(messageSize)
    {}

    void grid(GridBase<DataflowTestModel, 1> *grid)
    {
        Region<1> boundingRegion = grid->boundingRegion();
        GridBase<DataflowTestModel, 1>::SparseMatrix weights;

        for (auto&& i: boundingRegion) {
            std::vector<int> neighbors = genNeighborList(i.x());
            DataflowTestModel cell(i.x(), 0, messageSize, neighbors);
            grid->set(i, cell);

            for (auto&& j: neighbors) {
                weights << std::make_pair(Coord<2>(i.x(), j), i.x() * 1000 + j);
            }
        }

        grid->setWeights(0, weights);
    }

    Coord<1> gridDimensions() const
    {
	return Coord<1>(dim.prod());
    }

    unsigned startStep() const
    {
	return 0;
    }

    unsigned maxSteps() const
    {
	return myMaxSteps;
    }

    AdjacencyPtr getAdjacency(const Region<1>& region) const
    {
	AdjacencyPtr adjacency(new RegionBasedAdjacency());

	for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
            std::vector<int> neighbors = genNeighborList(i->x());
            for (auto&& neighbor: neighbors) {
                adjacency->insert(i->x(), neighbor);
            }
	}

	return adjacency;
    }

private:
    Coord<2> dim;
    int myMaxSteps;
    int messageSize;

    std::vector<int> genNeighborList(int id) const
    {
        Coord<2> c(id % dim.x(), id / dim.x());

        std::vector<int> ret;
        if (c.y() > 0) {
            ret << (id - dim.x());
        }
        if (c.x() > 0) {
            ret << (id - 1);
        }
        if (c.x() < (dim.x() - 1)) {
            ret << (id + 1);
        }
        if (c.y() < (dim.y() - 1)) {
            ret << (id + dim.x());
        }

        return ret;
    }
};

class HPXDataflowGold : public CPUBenchmark
{
public:
    std::string family()
    {
        return "HPXDataflow";
    }

    std::string species()
    {
        return "gold";
    }

    std::string unit()
    {
        return "MLUPS";
    }

    double performance(std::vector<int> dim)
    {
        Coord<2> gridDim(dim[0], dim[1]);
        int maxSteps = dim[2];
        int messageSize = 27;

        double seconds;
        {
            ScopedTimer t(&seconds);

            Initializer<DataflowTestModel> *initializer = new DataflowTestInitializer(gridDim, maxSteps, messageSize);
            HPXDataflowSimulator<DataflowTestModel> sim(initializer, "HPXDataflowGoldPerformance");
            sim.run();
        }

        double latticeUpdates = 1.0 * gridDim.prod() * maxSteps;
        double glups = latticeUpdates / seconds * 1e-6;

        return glups;
    }
};

class HPXDataflowVanilla : public CPUBenchmark
{
public:
    typedef typename SharedPtr<Adjacency>::Type AdjacencyPtr;

    class DummyHood
    {
    public:
        inline DummyHood(int numCells, const AdjacencyPtr& adjacency) :
            messagesOld(numCells),
            messagesNew(numCells),
            neighborVec(numCells)
        {
            for (int i = 0; i < numCells; ++i) {
                adjacency->getNeighbors(i, &neighborVec[i]);

                for (auto&& j: neighborVec[i]) {
                    messagesOld[i][j] = MessageType();
                    messagesNew[i][j] = MessageType();
                }
            }
        }

        inline
        const MessageType& operator[](const int i)
        {
            return messagesOld[index][i];
        }

        template<typename MESSAGE_TYPE>
        inline
        void send(const int i, MESSAGE_TYPE&& message)
        {
            messagesNew[i][index] = std::forward<MESSAGE_TYPE>(message);
        }

        inline
        const std::vector<int>& neighbors() const
        {
            return neighborVec[index];
        }

        inline
        void setIndex(int i)
        {
            index = i;
        }

        inline
        void swapMessages()
        {
            std::swap(messagesOld, messagesNew);
        }

    private:
        std::vector<std::map<int, MessageType> > messagesOld;
        std::vector<std::map<int, MessageType> > messagesNew;
        std::vector<std::vector<int> > neighborVec;
        int index;
    };
    std::string family()
    {
        return "HPXDataflow";
    }

    std::string species()
    {
        return "vanilla";
    }

    std::string unit()
    {
        return "MLUPS";
    }

    double performance(std::vector<int> dim)
    {
        Coord<2> gridDim(dim[0], dim[1]);
        int maxSteps = dim[2];
        int messageSize = 27;

        double seconds;
        {
            ScopedTimer t(&seconds);

            Initializer<DataflowTestModel> *initializer = new DataflowTestInitializer(gridDim, maxSteps, messageSize);
            ReorderingUnstructuredGrid<UnstructuredGrid<DataflowTestModel> > grid(initializer->gridBox());
            initializer->grid(&grid);
            int endX = initializer->gridDimensions().x();

            Region<1> region;
            region << initializer->gridBox();
            AdjacencyPtr adjacency = initializer->getAdjacency(region);

            DummyHood hood(endX, adjacency);

            DataflowTestModel *cells = grid.data();

            int maxSteps = initializer->maxSteps();
            for (int t = 0; t < maxSteps; ++t) {
                for (int i = 0; i < endX; ++i) {
                    hood.setIndex(i);
                    cells[i].update(hood, i);
                }

                hood.swapMessages();
            }
        }

        double latticeUpdates = 1.0 * gridDim.prod() * maxSteps;
        double glups = latticeUpdates / seconds * 1e-6;

        return glups;
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
    // updateLineX | SoA    | coarse      | unstructured | HPX       | compute-bound | HPXBusyworkCellPlatinum
    // update      | AoS    | fine        | unstructured | HPX       | compute-bound | HPXBusyworkCellBronze
    // updateLineX | AoS    | fine        | unstructured | HPX       | compute-bound | HPXBusyworkCellGold
    // update      | SoA    | fine        | unstructured | HPX       | compute-bound | *1
    // updateLineX | SoA    | fine        | unstructured | HPX       | compute-bound | HPXBusyworkCellTitanium
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
    sizes << Coord<3>( 10,  10, 30000)
          << Coord<3>(100, 100,   300);

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(HPXBusyworkCellIron(), toVector(sizes[i]));
    }

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(HPXBusyworkCellBronze(), toVector(sizes[i]));
    }

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(HPXBusyworkCellSilver(), toVector(sizes[i]));
    }

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(HPXBusyworkCellGold(), toVector(sizes[i]));
    }

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(HPXBusyworkCellPlatinum(), toVector(sizes[i]));
    }

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(HPXBusyworkCellTitanium(), toVector(sizes[i]));
    }

    sizes.clear();
    sizes << Coord<3>(100, 100, 1000);

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(HPXDataflowVanilla(), toVector(sizes[i]));
        eval(HPXDataflowGold(),    toVector(sizes[i]));
    }

    return hpx::finalize();
}


int main(int argc, char **argv)
{
    return hpx::init(argc, argv);
}
