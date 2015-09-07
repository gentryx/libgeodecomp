#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <libgeodecomp/communication/hpxpatchlink.h>
#include <libgeodecomp/communication/hpxserializationwrapper.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class HPXPatchLinkTest : public CxxTest::TestSuite
{
public:
    void testCircle()
    {
        std::size_t rank = hpx::get_locality_id();
        std::size_t size = hpx::find_all_localities().size();
        std::size_t leftNeighbor  = (rank - 1 + size) % size;
        std::size_t rightNeighbor = (rank + 1)        % size;

        std::string basename = "HPXPatchLinkTest::testCircle";

        Coord<2> dim(30, 20);
        DisplacedGrid<double> grid(CoordBox<2>(Coord<2>(), dim), -1);
        Region<2> sendRegion;
        Region<2> recvRegionLeft;
        Region<2> recvRegionRight;
        Region<2> gridRegion;
        int timeStep = 10;

        recvRegionLeft  << Streak<2>(Coord<2>(0, leftNeighbor),  dim.x());
        recvRegionRight << Streak<2>(Coord<2>(0, rightNeighbor), dim.x());
        sendRegion  << Streak<2>(Coord<2>(0, rank),          dim.x());
        gridRegion << CoordBox<2>(Coord<2>(), dim);

        for (auto&& i: sendRegion) {
            grid[i] = rank + timeStep;
        }

        HPXPatchLink<DisplacedGrid<double> >::Provider providerLeft( recvRegionLeft,  basename, leftNeighbor,  rank);
        HPXPatchLink<DisplacedGrid<double> >::Provider providerRight(recvRegionRight, basename, rightNeighbor, rank);
        HPXPatchLink<DisplacedGrid<double> >::Accepter accepterLeft( sendRegion,      basename, rank, leftNeighbor);
        HPXPatchLink<DisplacedGrid<double> >::Accepter accepterRight(sendRegion,      basename, rank, rightNeighbor);

        providerLeft.charge( 10, 30, 2);
        providerRight.charge(10, 30, 2);
        accepterLeft.charge( 10, 30, 2);
        accepterRight.charge(10, 30, 2);

        accepterLeft.put(  grid, gridRegion, 10);
        accepterRight.put( grid, gridRegion, 10);
        providerLeft.get( &grid, gridRegion, 10);
        providerRight.get(&grid, gridRegion, 10);

        double expectedValue = timeStep + leftNeighbor;
        for (auto&& i: recvRegionLeft) {
            TS_ASSERT_EQUALS(grid[i], expectedValue);
        }

        expectedValue = timeStep + rightNeighbor;
        for (auto&& i: recvRegionRight) {
            TS_ASSERT_EQUALS(grid[i], expectedValue);
        }
        // fixme: test next send/recv
    }

    // fixme: test soa too

};

}
