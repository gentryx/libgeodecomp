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
    typedef typename ParentType::GridType GridType;

    MockSim(TestInitializer<2> *init) : 
        DistributedSimulator<TestCell<2> >(init)
    {}
    
    virtual void getGridFragment(
        const GridType **grid, 
        const Region<2> **validRegion) 
    {
        *grid = myGrid;
        *validRegion = myRegion;
    }

    void setGridFragment(        
        const GridType *newGrid, 
        const Region<2> *newValidRegion) 
    {
        myGrid = newGrid;
        myRegion = newValidRegion;
    }

    void setStep(const unsigned& step)
    {
        stepNum = step;
    }

    virtual void step() 
    {}

    virtual void run() 
    {}

    const GridType *myGrid;
    const Region<2> *myRegion;
};

class ParallelMemoryWriterTest :  public CxxTest::TestSuite 
{
public:
    typedef DisplacedGrid<TestCell<2>, TestCell<2>::Topology> GridType; 

    void setUp()
    {
        dim = Coord<2>(10, 13);
        init = new TestInitializer<2>(dim);
        sim.reset(new MockSim(init));
        writer = new ParallelMemoryWriter<TestCell<2> >(&*sim);
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
            sim->setGridFragment(&grid, &stripes[index]);
            writer->stepFinished();        
        }

        TS_ASSERT_EQUALS(writer->getGrids()[  0].getDimensions(), dim);
        TS_ASSERT_EQUALS(writer->getGrids()[123].getDimensions(), Coord<2>());

        TS_ASSERT_TEST_GRID(
            ParallelMemoryWriter<TestCell<2> >::GridType, 
            writer->getGrids()[0], 
            0);

        sim->setStep(123);

        for (int i = 0; i < 2; ++i) {
            int index = MPILayer().rank() + i * 2;
            sim->setGridFragment(&grid, &stripes[index]);
            writer->stepFinished();        
        }

        TS_ASSERT_EQUALS(writer->getGrids()[  0].getDimensions(), dim);
        TS_ASSERT_EQUALS(writer->getGrids()[123].getDimensions(), dim);

        TS_ASSERT_TEST_GRID(
            ParallelMemoryWriter<TestCell<2> >::GridType, 
            writer->getGrids()[0], 
            0);
        TS_ASSERT_TEST_GRID(
            ParallelMemoryWriter<TestCell<2> >::GridType, 
            writer->getGrids()[123], 
            0);
    }

private:
    Coord<2> dim;
    boost::shared_ptr<MockSim> sim;
    ParallelMemoryWriter<TestCell<2> > *writer;
    TestInitializer<2> *init;
};

}
