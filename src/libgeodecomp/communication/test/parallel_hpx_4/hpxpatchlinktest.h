#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <libgeodecomp/communication/hpxpatchlink.h>
#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/storage/displacedgrid.h>
#include <libgeodecomp/storage/soagrid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class HPXPatchLinkTest : public CxxTest::TestSuite
{
public:
    void testCircle()
    {
        std::size_t rank = hpx::get_locality_id();
        std::size_t size = hpx::get_num_localities().get();
        std::size_t leftNeighbor  = (rank - 1 + size) % size;
        std::size_t rightNeighbor = (rank + 1)        % size;

        std::string basename = "HPXPatchLinkTest::testCircle";

        Coord<2> dim(30, 20);
        DisplacedGrid<double> grid(CoordBox<2>(Coord<2>(), dim), -1);
        Region<2> sendRegion;
        Region<2> recvRegionLeft;
        Region<2> recvRegionRight;
        Region<2> gridRegion;

        recvRegionLeft  << Streak<2>(Coord<2>(0, leftNeighbor),  dim.x());
        recvRegionRight << Streak<2>(Coord<2>(0, rightNeighbor), dim.x());
        sendRegion  << Streak<2>(Coord<2>(0, rank),          dim.x());
        gridRegion << CoordBox<2>(Coord<2>(), dim);

        HPXPatchLink<DisplacedGrid<double> >::Provider providerLeft( recvRegionLeft,  basename, leftNeighbor,  rank);
        HPXPatchLink<DisplacedGrid<double> >::Provider providerRight(recvRegionRight, basename, rightNeighbor, rank);
        HPXPatchLink<DisplacedGrid<double> >::Accepter accepterLeft( sendRegion,      basename, rank, leftNeighbor);
        HPXPatchLink<DisplacedGrid<double> >::Accepter accepterRight(sendRegion,      basename, rank, rightNeighbor);

        providerLeft.charge( 10, 30, 5);
        providerRight.charge(10, 30, 5);
        accepterLeft.charge( 10, 30, 5);
        accepterRight.charge(10, 30, 5);

        // Step #0
        int timeStep = 10;
        for (auto&& i: sendRegion) {
            grid[i] = rank + timeStep;
        }

        accepterLeft.put(  grid, gridRegion, dim, timeStep, rank);
        accepterRight.put( grid, gridRegion, dim, timeStep, rank);
        providerLeft.get( &grid, gridRegion, dim, timeStep, rank);
        providerRight.get(&grid, gridRegion, dim, timeStep, rank);

        double expectedValue = timeStep + leftNeighbor;
        for (auto&& i: recvRegionLeft) {
            TS_ASSERT_EQUALS(grid[i], expectedValue);
        }

        expectedValue = timeStep + rightNeighbor;
        for (auto&& i: recvRegionRight) {
            TS_ASSERT_EQUALS(grid[i], expectedValue);
        }

        // Step #1
        timeStep += 5;
        for (auto&& i: sendRegion) {
            grid[i] = rank + timeStep;
        }

        accepterLeft.put(  grid, gridRegion, dim, timeStep, rank);
        accepterRight.put( grid, gridRegion, dim, timeStep, rank);
        providerLeft.get( &grid, gridRegion, dim, timeStep, rank);
        providerRight.get(&grid, gridRegion, dim, timeStep, rank);

        expectedValue = timeStep + leftNeighbor;
        for (auto&& i: recvRegionLeft) {
            TS_ASSERT_EQUALS(grid[i], expectedValue);
        }

        expectedValue = timeStep + rightNeighbor;
        for (auto&& i: recvRegionRight) {
            TS_ASSERT_EQUALS(grid[i], expectedValue);
        }
    }

    void testSoA()
    {
        typedef SoAGrid<TestCellSoA, Topologies::Torus<3>::Topology> GridType;
        std::size_t rank = hpx::get_locality_id();
        std::size_t size = hpx::get_num_localities().get();
        std::size_t leftNeighbor  = (rank - 1 + size) % size;
        std::size_t rightNeighbor = (rank + 1)        % size;

        std::string basename = "HPXPatchLinkTest::testSoA";

        Coord<3> dim(30, 20, 10);
        GridType grid(CoordBox<3>(Coord<3>(), dim));
        Region<3> sendRegion;
        Region<3> recvRegionLeft;
        Region<3> recvRegionRight;
        Region<3> gridRegion;

        Coord<3> planeDim(dim.x(), dim.y(), 1);
        recvRegionLeft  << CoordBox<3>(Coord<3>(0, 0, leftNeighbor ), planeDim);
        recvRegionRight << CoordBox<3>(Coord<3>(0, 0, rightNeighbor), planeDim);
        sendRegion      << CoordBox<3>(Coord<3>(0, 0, rank         ), planeDim);
        gridRegion << CoordBox<3>(Coord<3>(), dim);

        HPXPatchLink<GridType>::Provider providerLeft( recvRegionLeft,  basename, leftNeighbor,  rank);
        HPXPatchLink<GridType>::Provider providerRight(recvRegionRight, basename, rightNeighbor, rank);
        HPXPatchLink<GridType>::Accepter accepterLeft( sendRegion,      basename, rank, leftNeighbor);
        HPXPatchLink<GridType>::Accepter accepterRight(sendRegion,      basename, rank, rightNeighbor);

        providerLeft.charge( 40, 110, 10);
        providerRight.charge(40, 110, 10);
        accepterLeft.charge( 40, 110, 10);
        accepterRight.charge(40, 110, 10);

        // Step #0
        int timeStep = 40;
        for (auto&& i: sendRegion) {
            TestCellSoA cell = grid.get(i);
            cell.testValue = timeStep;
            cell.pos.x() = rank;
            grid.set(i, cell);
        }

        accepterLeft.put(  grid, gridRegion, dim, timeStep, rank);
        accepterRight.put( grid, gridRegion, dim, timeStep, rank);
        providerLeft.get( &grid, gridRegion, dim, timeStep, rank);
        providerRight.get(&grid, gridRegion, dim, timeStep, rank);

        double expectedValue = timeStep;
        Coord<3> expectedPos(leftNeighbor, 0, 0);
        for (auto&& i: recvRegionLeft) {
            TestCellSoA cell = grid.get(i);
            TS_ASSERT_EQUALS(cell.testValue, expectedValue);
            TS_ASSERT_EQUALS(cell.pos,       expectedPos);
        }
        expectedPos = Coord<3>(rightNeighbor, 0, 0);
        for (auto&& i: recvRegionRight) {
            TestCellSoA cell = grid.get(i);
            TS_ASSERT_EQUALS(cell.testValue, expectedValue);
            TS_ASSERT_EQUALS(cell.pos,       expectedPos);
        }

        // Step #1
        timeStep = 50;
        for (auto&& i: sendRegion) {
            TestCellSoA cell = grid.get(i);
            cell.testValue = timeStep;
            cell.pos.x() = rank;
            grid.set(i, cell);
        }

        accepterLeft.put(  grid, gridRegion, dim, timeStep, rank);
        accepterRight.put( grid, gridRegion, dim, timeStep, rank);
        providerLeft.get( &grid, gridRegion, dim, timeStep, rank);
        providerRight.get(&grid, gridRegion, dim, timeStep, rank);

        expectedValue = timeStep;
        expectedPos = Coord<3>(leftNeighbor, 0, 0);
        for (auto&& i: recvRegionLeft) {
            TestCellSoA cell = grid.get(i);
            TS_ASSERT_EQUALS(cell.testValue, expectedValue);
            TS_ASSERT_EQUALS(cell.pos,       expectedPos);
        }
        expectedPos = Coord<3>(rightNeighbor, 0, 0);
        for (auto&& i: recvRegionRight) {
            TestCellSoA cell = grid.get(i);
            TS_ASSERT_EQUALS(cell.testValue, expectedValue);
            TS_ASSERT_EQUALS(cell.pos,       expectedPos);
        }

        // Step #2
        timeStep = 60;
        for (auto&& i: sendRegion) {
            TestCellSoA cell = grid.get(i);
            cell.testValue = timeStep;
            cell.pos.x() = rank;
            grid.set(i, cell);
        }

        accepterLeft.put(  grid, gridRegion, dim, timeStep, rank);
        accepterRight.put( grid, gridRegion, dim, timeStep, rank);
        providerLeft.get( &grid, gridRegion, dim, timeStep, rank);
        providerRight.get(&grid, gridRegion, dim, timeStep, rank);

        expectedValue = timeStep;
        expectedPos = Coord<3>(leftNeighbor, 0, 0);
        for (auto&& i: recvRegionLeft) {
            TestCellSoA cell = grid.get(i);
            TS_ASSERT_EQUALS(cell.testValue, expectedValue);
            TS_ASSERT_EQUALS(cell.pos,       expectedPos);
        }
        expectedPos = Coord<3>(rightNeighbor, 0, 0);
        for (auto&& i: recvRegionRight) {
            TestCellSoA cell = grid.get(i);
            TS_ASSERT_EQUALS(cell.testValue, expectedValue);
            TS_ASSERT_EQUALS(cell.pos,       expectedPos);
        }

        // Step #3
        timeStep = 70;
        for (auto&& i: sendRegion) {
            TestCellSoA cell = grid.get(i);
            cell.testValue = timeStep;
            cell.pos.x() = rank;
            grid.set(i, cell);
        }

        accepterLeft.put(  grid, gridRegion, dim, timeStep, rank);
        accepterRight.put( grid, gridRegion, dim, timeStep, rank);
        providerLeft.get( &grid, gridRegion, dim, timeStep, rank);
        providerRight.get(&grid, gridRegion, dim, timeStep, rank);

        expectedValue = timeStep;
        expectedPos = Coord<3>(leftNeighbor, 0, 0);
        for (auto&& i: recvRegionLeft) {
            TestCellSoA cell = grid.get(i);
            TS_ASSERT_EQUALS(cell.testValue, expectedValue);
            TS_ASSERT_EQUALS(cell.pos,       expectedPos);
        }
        expectedPos = Coord<3>(rightNeighbor, 0, 0);
        for (auto&& i: recvRegionRight) {
            TestCellSoA cell = grid.get(i);
            TS_ASSERT_EQUALS(cell.testValue, expectedValue);
            TS_ASSERT_EQUALS(cell.pos,       expectedPos);
        }

        // Step #4
        timeStep = 80;
        for (auto&& i: sendRegion) {
            TestCellSoA cell = grid.get(i);
            cell.testValue = timeStep;
            cell.pos.x() = rank;
            grid.set(i, cell);
        }

        accepterLeft.put(  grid, gridRegion, dim, timeStep, rank);
        accepterRight.put( grid, gridRegion, dim, timeStep, rank);
        providerLeft.get( &grid, gridRegion, dim, timeStep, rank);
        providerRight.get(&grid, gridRegion, dim, timeStep, rank);

        expectedValue = timeStep;
        expectedPos = Coord<3>(leftNeighbor, 0, 0);
        for (auto&& i: recvRegionLeft) {
            TestCellSoA cell = grid.get(i);
            TS_ASSERT_EQUALS(cell.testValue, expectedValue);
            TS_ASSERT_EQUALS(cell.pos,       expectedPos);
        }
        expectedPos = Coord<3>(rightNeighbor, 0, 0);
        for (auto&& i: recvRegionRight) {
            TestCellSoA cell = grid.get(i);
            TS_ASSERT_EQUALS(cell.testValue, expectedValue);
            TS_ASSERT_EQUALS(cell.pos,       expectedPos);
        }

        // Step #5
        timeStep = 90;
        for (auto&& i: sendRegion) {
            TestCellSoA cell = grid.get(i);
            cell.testValue = timeStep;
            cell.pos.x() = rank;
            grid.set(i, cell);
        }

        accepterLeft.put(  grid, gridRegion, dim, timeStep, rank);
        accepterRight.put( grid, gridRegion, dim, timeStep, rank);
        providerLeft.get( &grid, gridRegion, dim, timeStep, rank);
        providerRight.get(&grid, gridRegion, dim, timeStep, rank);

        expectedValue = timeStep;
        expectedPos = Coord<3>(leftNeighbor, 0, 0);
        for (auto&& i: recvRegionLeft) {
            TestCellSoA cell = grid.get(i);
            TS_ASSERT_EQUALS(cell.testValue, expectedValue);
            TS_ASSERT_EQUALS(cell.pos,       expectedPos);
        }
        expectedPos = Coord<3>(rightNeighbor, 0, 0);
        for (auto&& i: recvRegionRight) {
            TestCellSoA cell = grid.get(i);
            TS_ASSERT_EQUALS(cell.testValue, expectedValue);
            TS_ASSERT_EQUALS(cell.pos,       expectedPos);
        }

        // Step #6
        timeStep = 100;
        for (auto&& i: sendRegion) {
            TestCellSoA cell = grid.get(i);
            cell.testValue = timeStep;
            cell.pos.x() = rank;
            grid.set(i, cell);
        }

        accepterLeft.put(  grid, gridRegion, dim, timeStep, rank);
        accepterRight.put( grid, gridRegion, dim, timeStep, rank);
        providerLeft.get( &grid, gridRegion, dim, timeStep, rank);
        providerRight.get(&grid, gridRegion, dim, timeStep, rank);

        expectedValue = timeStep;
        expectedPos = Coord<3>(leftNeighbor, 0, 0);
        for (auto&& i: recvRegionLeft) {
            TestCellSoA cell = grid.get(i);
            TS_ASSERT_EQUALS(cell.testValue, expectedValue);
            TS_ASSERT_EQUALS(cell.pos,       expectedPos);
        }
        expectedPos = Coord<3>(rightNeighbor, 0, 0);
        for (auto&& i: recvRegionRight) {
            TestCellSoA cell = grid.get(i);
            TS_ASSERT_EQUALS(cell.testValue, expectedValue);
            TS_ASSERT_EQUALS(cell.pos,       expectedPos);
        }
    }
};

}
