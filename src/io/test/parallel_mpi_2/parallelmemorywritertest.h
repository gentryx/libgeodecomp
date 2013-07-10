#include <boost/filesystem.hpp>
#include <cxxtest/TestSuite.h>

#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/io/parallelmemorywriter.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/testhelper.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class MockSim : public DistributedSimulator<TestCell<2> >
{
public:
    typedef DistributedSimulator<TestCell<2> > ParentType;
    typedef ParentType::GridType GridType;

    MockSim(TestInitializer<TestCell<2> > *init) :
        DistributedSimulator<TestCell<2> >(init)
    {}

    void setStep(const unsigned& step)
    {
        stepNum = step;
    }

    virtual void step()
    {}

    virtual void run()
    {}
};

class ParallelMemoryWriterTest :  public CxxTest::TestSuite
{
public:
    typedef DisplacedGrid<TestCell<2>, TestCell<2>::Topology> GridType;

    void setUp()
    {
        dim = Coord<2>(10, 13);
        init = new TestInitializer<TestCell<2> >(dim);
        sim.reset(new MockSim(init));
        writer = new ParallelMemoryWriter<TestCell<2> >();
        sim->addWriter(writer);
    }

    void tearDown()
    {
        sim.reset();
    }

    void testBasic()
    {
        GridType grid(init->gridBox());
        init->grid(&grid);
        TS_ASSERT_TEST_GRID(GridType, grid, 0);

        Region<2> stripes[4];
        for (int i = 0; i < 4; ++i) {
            int startY = dim.y() * i / 4;
            int endY = dim.y() * (i + 1) / 4;

            for (int y = startY; y < endY; ++y) {
                stripes[i] << Streak<2>(Coord<2>(0, y), 10);
            }
        }

        for (int i = 0; i < 2; ++i) {
            int index = MPILayer().rank() * 2 + i;
            writer->stepFinished(
                grid,
                stripes[index],
                grid.getDimensions(),
                sim->getStep(),
                WRITER_STEP_FINISHED,
                MPILayer().rank(),
                true);
        }

        TS_ASSERT_EQUALS(writer->getGrid(  0).getDimensions(), dim);
        TS_ASSERT_EQUALS(writer->getGrid(123).getDimensions(), Coord<2>());

        TS_ASSERT_TEST_GRID(
            ParallelMemoryWriter<TestCell<2> >::GridType,
            writer->getGrid(0),
            0);

        sim->setStep(123);

        for (int i = 0; i < 2; ++i) {
            int index = MPILayer().rank() + i * 2;
            writer->stepFinished(
                grid,
                stripes[index],
                grid.getDimensions(),
                sim->getStep(),
                WRITER_STEP_FINISHED,
                MPILayer().rank(),
                true);
        }

        TS_ASSERT_EQUALS(writer->getGrids()[  0].getDimensions(), dim);
        TS_ASSERT_EQUALS(writer->getGrids()[123].getDimensions(), dim);

        TS_ASSERT_TEST_GRID(
            ParallelMemoryWriter<TestCell<2> >::GridType,
            writer->getGrid(0),
            0);
        TS_ASSERT_TEST_GRID(
            ParallelMemoryWriter<TestCell<2> >::GridType,
            writer->getGrid(123),
            0);
    }

private:
    Coord<2> dim;
    boost::shared_ptr<MockSim> sim;
    ParallelMemoryWriter<TestCell<2> > *writer;
    TestInitializer<TestCell<2> > *init;
};

}
