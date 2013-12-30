#include <cxxtest/TestSuite.h>

#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>
#include <libgeodecomp/io/parallelmemorywriter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/hiparsimulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/cudastepper.h>

#include <cuda.h>

using namespace LibGeoDecomp;
using namespace HiParSimulator;

namespace LibGeoDecomp {
namespace HiParSimulator {

class CUDAUpdateTestCell
{
public:
    class API :
        public APITraits::HasPredefinedMPIDataType<int>
    {};

    CUDAUpdateTestCell(int counter = 0) :
        counter(counter)
    {}

    template<typename NEIGHBORHOOD>
    __device__ __host__
    void update(const NEIGHBORHOOD& hood, int)
    {
        counter = hood[FixedCoord<0, 0, 0>()].counter + 1;
    }

    int counter;
};

class CUDAUpdateInitializer : public SimpleInitializer<CUDAUpdateTestCell>
{
public:
    CUDAUpdateInitializer(const Coord<2>& dimensions, int maxSteps) :
        SimpleInitializer<CUDAUpdateTestCell>(dimensions, maxSteps)
    {}

    void grid(GridBase<CUDAUpdateTestCell, 2> *target)
    {
        CoordBox<2> box = target->boundingBox();
        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            target->set(*i, CUDAUpdateTestCell(i->y() * 1000000 + i->x() * 1000));
        }
    }
};

class CUDAStepperBasicTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
        // HiParSimulator<TestCell<2>, RecursiveBisectionPartition<2>, CUDAStepper<TestCell<2> > > sim(
        //     new TestInitializer<TestCell<2> >(Coord<2>(17, 12)));
        // sim.run();
    }

    void testUpdate()
    {
        typedef CUDAUpdateTestCell CellType;
        int ioPeriod = 10;
        Coord<2> dim(20, 10);
        int maxT = 300;
        HiParSimulator<CellType, RecursiveBisectionPartition<2>, CUDAStepper<CellType> > sim(
            new CUDAUpdateInitializer(dim, maxT));
        ParallelMemoryWriter<CellType> *writer = new ParallelMemoryWriter<CellType>(ioPeriod);
        sim.addWriter(writer);
        sim.run();

        for (int t = 0; t <= maxT; t += ioPeriod) {
            for (int y = 0; y < dim.y(); ++y) {
                for (int x = 0; x < dim.x(); ++x) {
                    int actual = writer->getGrid(t).get(Coord<2>(x, y)).counter;
                    int expected = y * 1000000 + x * 1000 + t;

                    TS_ASSERT_EQUALS(actual, expected);
                }
            }
        }
    }
};

}
}
