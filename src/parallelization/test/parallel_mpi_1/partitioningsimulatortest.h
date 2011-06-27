#include <cxxtest/TestSuite.h>
#include <limits>
#include <sstream>
#include <map>
#include "../../../io/testinitializer.h"
#include "../../../io/mockwriter.h"
#include "../../../io/writer.h"
#include "../../../misc/testcell.h"
#include "../../../misc/testhelper.h"
#include "../../../mpilayer/mpilayer.h"
#include "../../partitioningsimulator.h"
#include "../../serialsimulator.h"

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class PartitioningSimulatorTest : public CxxTest::TestSuite 
{
private:
    double EPSILON;
    PartitioningSimulator<TestCell<2> > *_testSim;
    SerialSimulator<TestCell<2> > *_referenceSim;
    std::string _configFile;
    Initializer<TestCell<2> > *_init;
    typedef std::map<Coord<2>, int> CoordIntMap;
    MPILayer _mpilayer;

public:
    PartitioningSimulatorTest()
    {
        EPSILON = std::numeric_limits<double>::epsilon();
    }


    void setUp() 
    {
        unsigned width = 30;
        unsigned height = 20;
        unsigned steps = 2;
        _init = new TestInitializer<2>(Coord<2>(width, height), steps);
        _testSim = new PartitioningSimulator<TestCell<2> >(
            new TestInitializer<2>(Coord<2>(width, height), steps));
        _referenceSim = new  SerialSimulator<TestCell<2> >(
            new TestInitializer<2>(Coord<2>(width, height), steps));
    }

    
    void tearDown()
    {
        delete _testSim;
        delete _referenceSim;
        delete _init;
    }


    void testInnerRimAndCore()
    {
        // innerRim and Core should be disjunct and combine to form the
        CoordBox<2> partition = _testSim->partitionRectangle();
        CoordBox<2> core = _testSim->_coreRectangle;
        CoordSet innerRim = _testSim->_innerRim;
        CoordIntMap coords;
        for (CoordBoxSequence<2> s = partition.sequence();
                s.hasNext();) 
            coords[s.next()] = 1;
        for (CoordBoxSequence<2> s = core.sequence(); s.hasNext();) 
            coords[s.next()] -= 1;
        for (CoordSet::Sequence s = innerRim.sequence(); s.hasNext();) 
            coords[s.next()] -= 1;
        for (CoordIntMap::iterator i = coords.begin(); i!= coords.end(); i++) {
            std::stringstream message;
            message << "[rank " << _mpilayer.rank() 
                    << "] " << "error at coord: " << (i->first).toString();
            TSM_ASSERT_EQUALS(message.str(), 0, i->second);
        }
    }


    void testCreatePartition()
    {
        PartitioningSimulator<TestCell<2> > sim1(
            new TestInitializer<2>(), "striping");
        // striping is implemented as a special case of recursive bisection
        TS_ASSERT(dynamic_cast<PartitionRecursiveBisection*>(sim1._partition));
        PartitioningSimulator<TestCell<2> > sim2(
            new TestInitializer<2>(), "recursive_bisection");
        TS_ASSERT(dynamic_cast<PartitionRecursiveBisection*>(sim2._partition));
    }


    void testInitialization()
    {
        const Partition* partition = _testSim->_partition;
        PartitioningSimulator<TestCell<2> >::UCVMap sendMap = 
            _testSim->sendInnerRimMap(partition); 
        PartitioningSimulator<TestCell<2> >::UCVMap recvMap = 
            _testSim->recvOuterRimMap(partition); 
        // symmetry in neighboorhood 
        std::ostringstream msg; 
        msg << "In partition: " << _mpilayer.rank() << "\n" 
            << "sendMap: " << sendMap << "\n" 
            << "recvMap: " << recvMap << "\n"; 
        TSM_ASSERT_EQUALS(msg.str().c_str(), sendMap.size(), recvMap.size()); 
    }

    
    void testRun()
    {
        _testSim->run();
        _referenceSim->run();
        Grid<TestCell<2> > testGrid = _testSim->gatherWholeGrid();
        Grid<TestCell<2> > refGrid = *_referenceSim->getGrid();

        TS_ASSERT_EQUALS(_init->maxSteps(), _testSim->getStep());
        if (_mpilayer.rank() == 0) {// other ranks don't have the whole grid
            TSM_ASSERT_EQUALS(
                refGrid.diff(testGrid).c_str(), testGrid, refGrid);
            TS_ASSERT_TEST_GRID(
                Grid<TestCell<2> >, 
                testGrid, _init->maxSteps() * TestCell<2>::nanoSteps());
        }
    }

    
    void testRepartition()
    {
        Grid<TestCell<2> >* grid = _testSim->getGrid();
        for (unsigned y = 0; y < (unsigned)grid->getDimensions().y(); y++)
            for (unsigned x = 0; x < (unsigned)grid->getDimensions().x(); x++)
                (*grid)[Coord<2>(x, y)].testValue = 100 * _mpilayer.rank() + 10 * x + y;
        
        DVec powers(_mpilayer.size(), 1);
        for (unsigned i = 0; i < powers.size(); i++) {
            powers[i] = i;
        }

        Partition* pOld = _testSim->_partition;
        Splitter splitter(powers);
        Partition* pNew = new PartitionRecursiveBisection(
                pOld->getRect(), pOld->getNodes(), splitter);

        if (_mpilayer.size() > 1) TS_ASSERT(!pNew->equal(pOld));

        /*
        if (_mpilayer.rank() == 0)
            std::cout << "\nold Partition: " << pOld->toString()
                << "\n new Partition: " << pNew->toString();
                */

        //Grid<TestCell<2> > before = *_testSim->getGrid();
        
        Grid<TestCell<2> > beforeWhole = _testSim->gatherWholeGrid();
        _testSim->repartition(pOld, pNew);
        delete pOld;
        Grid<TestCell<2> > afterWhole = _testSim->gatherWholeGrid();

        //Grid<TestCell<2> > after = *_testSim->getGrid();

        /*
        usleep(_mpilayer.rank() * 100000);
        std::cout << "\n Grid before (rank " << _mpilayer.rank() << "):\n" << gridTestValues(before);
        std::cout << "\n Grid after (rank " << _mpilayer.rank() << "):\n" << gridTestValues(after);
        std::cout.flush();
        */

        if (_mpilayer.rank() == 0) // other ranks don't have the whole grid
            TSM_ASSERT_EQUALS(beforeWhole.diff(afterWhole).c_str(), beforeWhole, afterWhole);
    }


    std::string gridTestValues(Grid<TestCell<2> >& grid)
    {
        std::stringstream s;
        s << "[\n";
        for (unsigned y = 0; y < (unsigned)grid.getDimensions().y(); y++) {
            s << "[";
            for (unsigned x = 0; x < (unsigned)grid.getDimensions().x(); x++) {
                TestCell<2>& c = grid[Coord<2>(x, y)];
                s << c.testValue << ", ";
            }
            s << "]\n";
        }
        s << "]";
        return s.str();
    }
};

};
