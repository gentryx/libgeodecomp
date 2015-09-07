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
        Region<2> recvRegion;
        Region<2> gridRegion;
        int timeStep = 10;

        recvRegion << Streak<2>(Coord<2>(0, leftNeighbor), dim.x());
        sendRegion << Streak<2>(Coord<2>(0, rank),         dim.x());
        gridRegion << CoordBox<2>(Coord<2>(), dim);

        for (auto&& i: sendRegion) {
            grid[i] = rank + timeStep;
        }

        HPXPatchLink<DisplacedGrid<double> >::Provider provider(recvRegion, basename, leftNeighbor, rank);
        HPXPatchLink<DisplacedGrid<double> >::Accepter accepter(sendRegion, basename, rank, rightNeighbor);

        // fixme: extend this to a bi-directional bus
        provider.charge(10, 30, 2);
        accepter.charge(10, 30, 2);

        accepter.put(grid,  gridRegion, 10);
        provider.get(&grid, gridRegion, 10);

        double expectedValue = timeStep + leftNeighbor;
        for (auto&& i: recvRegion) {
            TS_ASSERT_EQUALS(grid[i], expectedValue);
        }
        // fixme: test next send/recv
    }

    // fixme: test soa too

};

}
