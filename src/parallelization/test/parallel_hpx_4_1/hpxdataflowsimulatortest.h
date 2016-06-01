#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>

#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/storage/unstructuredgrid.h>
#include <libgeodecomp/parallelization/hpxdataflowsimulator.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class DummyResult
{
public:
    DummyResult(int id) :
        id(id)
    {}

    int id;
};

class DummyModel
{
public:
    class API :      
        public APITraits::HasUnstructuredTopology
    {};

    DummyModel(int id = -1) :
        id(id)
    {}

    DummyResult update(const std::vector<DummyResult>& input)
    {
	std::cout << "updating Dummy " << id << " and my neighbors are: [";
	for (int i = 0; i != input.size(); ++i) {
	    std::cout << input[i].id << " ";
	}
	std::cout << "]\n";

	return DummyResult(id);
    }

private:
    int id;
};

 class DummyInitializer : public Initializer<DummyModel>
 {
 public:
     DummyInitializer(int gridSize, int myMaxSteps) :
         gridSize(gridSize),
	 myMaxSteps(myMaxSteps)
     {}

     void grid(GridBase<DummyModel, 1> *grid)
     {
	 CoordBox<1> box = grid->boundingBox();

	 for (CoordBox<1>::Iterator i = box.begin(); i != box.end(); ++i) {
	     grid->set(*i, DummyModel(i->x()));
	 }
     }

    virtual Coord<1> gridDimensions() const
    {
	return Coord<1>(gridSize);
    }

    unsigned startStep() const
    {
	return 0;
    }

    unsigned maxSteps() const
    {
	return myMaxSteps;
    }

    boost::shared_ptr<Adjacency> getAdjacency(const Region<1>& region) const
    {
	boost::shared_ptr<Adjacency> adjacency(new RegionBasedAdjacency());

	for (Region<1>::Iterator i = region.begin(); i != region.end(); ++i) {
	    if (i->x() != 0) {
		adjacency->insert(i->x(), i->x() - 1);
	    }

	    adjacency->insert(i->x(), i->x());

	    if (i->x() != (gridSize - 1)) {
		adjacency->insert(i->x(), i->x() + 1);
	    }
	}

	return adjacency;
    }

private:
    int gridSize;
    int myMaxSteps;
 };


 class HpxDataflowSimulatorTest : public CxxTest::TestSuite
 {
 public:
     void setUp()
     {
     }

     void testBasic()
     {
	 std::cout << "starting dataflow test\n";

	 DummyInitializer initializer(5, 13);
	 UnstructuredGrid<DummyModel> grid(initializer.gridBox());
	 initializer.grid(&grid);
	 std::cout << "grid size: " << grid.boundingBox() << "\n";

	 typedef hpx::shared_future<DummyResult> ResultsFuture;
	 typedef std::map<int, ResultsFuture> Space; // futures of one time step

	 using hpx::dataflow;
	 using hpx::util::unwrapped;

	 // U[t][i] is the state of position i at time t.
	 std::vector<Space> futures(2);

	 Region<1> localRegion;
	 localRegion << initializer.gridBox();

	 for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
	     futures[0][i->x()] = hpx::make_ready_future(DummyResult(i->x()));
	 }

	 boost::shared_ptr<Adjacency> adjacency = initializer.getAdjacency(localRegion);

	 std::cout << "setting up dataflow\n";
	
	 int maxTimeSteps = initializer.maxSteps();
	 for (int t = 0; t < maxTimeSteps; ++t) {
	     Space& lastTimeStepFutures = futures[(t + 0) % 2];
	     Space& thisTimeStepFutures = futures[(t + 1) % 2];

             for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
                 auto Op = unwrapped(boost::bind(&DummyModel::update, &grid[i->x()], _1));
		 std::vector<ResultsFuture> localDependencies;

		 std::vector<int> neighbors;
		 adjacency->getNeighbors(i->x(), &neighbors);

		 for (std::vector<int>::iterator n = neighbors.begin(); n != neighbors.end(); ++n) {
		     localDependencies.push_back(lastTimeStepFutures[*n]);
		 }
		 
		 thisTimeStepFutures[i->x()] = dataflow(hpx::launch::async, Op, localDependencies);
	     }
	 }
	
	 std::cout << "waiting on futures\n";

	 std::vector<ResultsFuture> finalStep;
	 for (Region<1>::Iterator i = localRegion.begin(); i != localRegion.end(); ++i) {
	     finalStep.push_back(futures[maxTimeSteps % 2][i->x()]);
	 }
	 hpx::when_all(finalStep).get();

	 std::cout << "dataflow test done\n";
     }
 };

}
