#include <cxxtest/TestSuite.h>

#include <libgeodecomp/communication/boostserialization.h>
#include <libgeodecomp/communication/hpxserialization.h>
#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/communication/patchlink.h>
#include <libgeodecomp/communication/typemaps.h>
#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/geometry/stencils.h>
#include <libgeodecomp/geometry/voronoimesher.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/io/remotesteerer/pipe.h>
#include <libgeodecomp/io/serialbovwriter.h>
#include <libgeodecomp/io/silowriter.h>
#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/steerer.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/misc/nonpodtestcell.h>
#include <libgeodecomp/misc/sharedptr.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/storage/soagrid.h>
#include <libgeodecomp/storage/updatefunctor.h>

namespace LibGeoDecomp {}
using namespace LibGeoDecomp;

namespace LibGeoDecomp {

/**
 * Test model for use with Boost.Serialization
 */
class MyComplicatedCell
{
public:
    class API : public APITraits::HasBoostSerialization
    {};

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int nanoStep)
    {
    }

    template<typename ARCHIVE>
    void serialize(ARCHIVE& archive, int version)
    {
        archive & x;
        archive & cargo;
    }

    int x;
    std::vector<int> cargo;
};

class PatchLinkTest : public CxxTest::TestSuite
{
public:
    typedef DisplacedGrid<int> GridType;
    typedef PatchLink<GridType>::Accepter PatchAccepterType;
    typedef PatchLink<GridType>::Provider PatchProviderType;

    typedef TestCellSoA TestCellType;
    typedef SoAGrid<TestCellSoA, Topologies::Cube<3>::Topology> GridType2;

    typedef DisplacedGrid<MyComplicatedCell> GridType3;

    void setUp()
    {
        mpiLayer.reset(new MPILayer());

        region1.clear();
        region1 << Streak<2>(Coord<2>(2, 2), 4);
        region1 << Streak<2>(Coord<2>(2, 3), 5);
        region2.clear();
        region2 << Streak<2>(Coord<2>(0, 0), 6);
        region2 << Streak<2>(Coord<2>(4, 1), 5);

        boundingBox = CoordBox<2>(Coord<2>(0, 0), Coord<2>(7, 5));
        zeroGrid  = GridType(boundingBox, 0);
        boundingRegion.clear();
        boundingRegion << boundingBox;

        sendGrid1 = markGrid(region1, mpiLayer->rank());
        sendGrid2 = markGrid(region2, mpiLayer->rank());

        tag = 69;
    }

    void tearDown()
    {
        acc.reset();
        pro.reset();
        mpiLayer.reset();
    }

    void testBasic()
    {
        std::size_t nanoStep = 0;
        for (int sender = 0; sender < mpiLayer->size(); ++sender) {
            for (int receiver = 0; receiver < mpiLayer->size(); ++receiver) {
                if (sender != receiver) {
                    Region<2>& region  = sender % 2 ? region2 : region1;
                    GridType& sendGrid = sender % 2 ? sendGrid2 : sendGrid1;
                    std::size_t nanoStep = sender * receiver + 4711;

                    if (sender == mpiLayer->rank()) {
                        acc.reset(new PatchAccepterType(
                                      region,
                                      receiver,
                                      tag,
                                      MPI_INT));

                        acc->pushRequest(nanoStep);
                        acc->put(sendGrid, boundingRegion, boundingBox.dimensions, nanoStep, sender);
                        acc->wait();
                    }

                    if (receiver == mpiLayer->rank()) {
                        pro.reset(new PatchProviderType(
                                      region,
                                      sender,
                                      tag,
                                      MPI_INT));

                        expected = markGrid(region, sender);
                        actual = zeroGrid;

                        pro->recv(nanoStep);
                        pro->wait();
                        pro->get(&actual, boundingRegion, boundingBox.dimensions, nanoStep, receiver);
                        TS_ASSERT_EQUALS(actual, expected);
                    }
                }

                ++nanoStep;
            }
        }
    }

    void testMultiple()
    {
        std::vector<SharedPtr<PatchAccepterType>::Type> accepters;
        std::vector<SharedPtr<PatchProviderType>::Type> providers;
        int stride = 4;
        std::size_t maxNanoSteps = 31;

        for (int i = 0; i < mpiLayer->size(); ++i) {
            if (i != mpiLayer->rank()) {
                accepters << SharedPtr<PatchAccepterType>::Type(
                    new PatchAccepterType(
                        region1,
                        i,
                        genTag(mpiLayer->rank(), i),
                        MPI_INT));

                providers << SharedPtr<PatchProviderType>::Type(
                    new PatchProviderType(
                        region1,
                        i,
                        genTag(i, mpiLayer->rank()),
                        MPI_INT));
            }
        }

        for (int i = 0; i < mpiLayer->size() - 1; ++i) {
            accepters[i]->charge(0, maxNanoSteps, stride);
            providers[i]->charge(0, maxNanoSteps, stride);
        }

        for (std::size_t nanoStep = 0; nanoStep < maxNanoSteps; nanoStep += stride) {
            GridType mySendGrid = markGrid(region1, mpiLayer->rank() * 10000 + nanoStep * 100);

            for (int i = 0; i < mpiLayer->size() - 1; ++i)
                accepters[i]->put(mySendGrid, boundingRegion, boundingBox.dimensions, nanoStep, mpiLayer->rank());

            for (int i = 0; i < mpiLayer->size() - 1; ++i) {
                std::size_t senderRank = i >= mpiLayer->rank() ? i + 1 : i;
                GridType expected = markGrid(region1, senderRank * 10000 + nanoStep * 100);
                GridType actual = zeroGrid;
                providers[i]->get(&actual, boundingRegion, boundingBox.dimensions, nanoStep, senderRank);

                TS_ASSERT_EQUALS(actual, expected);
            }
        }
    }

    void testMultiple2()
    {
        std::vector<SharedPtr<PatchAccepterType>::Type> accepters;
        std::vector<SharedPtr<PatchProviderType>::Type> providers;
        int stride = 4;
        std::size_t maxNanoSteps = 100;

        for (int i = 0; i < mpiLayer->size(); ++i) {
            if (i != mpiLayer->rank()) {
                accepters << SharedPtr<PatchAccepterType>::Type(
                    new PatchAccepterType(
                        region1,
                        i,
                        genTag(mpiLayer->rank(), i),
                        MPI_INT));

                providers << SharedPtr<PatchProviderType>::Type(
                    new PatchProviderType(
                        region1,
                        i,
                        genTag(i, mpiLayer->rank()),
                        MPI_INT));
            }
        }

        for (int i = 0; i < mpiLayer->size() - 1; ++i) {
            accepters[i]->charge(0, PatchAccepter<GridType>::infinity(), stride);
            providers[i]->charge(0, PatchProvider<GridType>::infinity(), stride);
        }

        for (std::size_t nanoStep = 0; nanoStep < maxNanoSteps; nanoStep += stride) {
            GridType mySendGrid = markGrid(region1, mpiLayer->rank() * 10000 + nanoStep * 100);

            for (int i = 0; i < mpiLayer->size() - 1; ++i) {
                accepters[i]->put(mySendGrid, boundingRegion, boundingBox.dimensions, nanoStep, mpiLayer->rank());
            }

            for (int i = 0; i < mpiLayer->size() - 1; ++i) {
                std::size_t senderRank = i >= mpiLayer->rank() ? i + 1 : i;
                GridType expected = markGrid(region1, senderRank * 10000 + nanoStep * 100);
                GridType actual = zeroGrid;
                providers[i]->get(&actual, boundingRegion, boundingBox.dimensions, nanoStep, senderRank);

                TS_ASSERT_EQUALS(actual, expected);
            }
        }

        for (int i = 0; i < mpiLayer->size() - 1; ++i) {
            accepters[i]->cancel();
            providers[i]->cancel();
        }
    }

    void testSoA()
    {
        Coord<3> dim(30, 20, 10);
        CoordBox<3> box(Coord<3>(), dim);
        Region<3> boxRegion;
        boxRegion << box;

        GridType2 sendGrid(box);
        GridType2 recvGrid(box);

        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            Coord<3> offset(0, 0, mpiLayer->rank() * 100);
            sendGrid.set(*i, TestCellSoA(*i + offset, dim, 0, mpiLayer->rank()));
        }

        std::vector<Region<3> > regions(mpiLayer->size());
        for (int i = 0; i < mpiLayer->size(); ++i) {
            Coord<3> frameDim = dim;
            frameDim.z() = 1;
            CoordBox<3> frameBox(Coord<3>(0, 0, i), frameDim);
            regions[i] << frameBox;
        }

        PatchLink<GridType2>::Accepter accepter(
            regions[mpiLayer->rank()],
            mpiLayer->size() - 1,
            2701,
            MPI_CHAR);
        accepter.charge(4, 4, 1);
        accepter.put(sendGrid, boxRegion, dim, 4, mpiLayer->rank());

        std::vector<SharedPtr<PatchLink<GridType2>::Provider>::Type> providers;
        if (mpiLayer->rank() == (mpiLayer->size() - 1)) {
            for (int i = 0; i < mpiLayer->size(); ++i) {
                providers.push_back(
                    SharedPtr<PatchLink<GridType2>::Provider>::Type(
                        new PatchLink<GridType2>::Provider(
                            regions[i],
                            i,
                            2701,
                            MPI_CHAR)));

                providers[i]->charge(4, 4, 1);
            }

            for (int i = 0; i < mpiLayer->size(); ++i) {
                providers[i]->get(&recvGrid, boxRegion, dim, 4, i);
            }

            for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
                Coord<3> offset(0, 0, mpiLayer->rank() * 100);
                TestCellSoA cell = recvGrid.get(*i);

                double expectedTestValue = i->z();
                Coord<3> expectedPosition;

                if (expectedTestValue < mpiLayer->size()) {
                    expectedPosition = Coord<3>(i->x(), i->y(), i->z() * 101);
                } else {
                    expectedTestValue = 666;
                }

                TS_ASSERT_EQUALS(expectedTestValue, cell.testValue);
                TS_ASSERT_EQUALS(expectedPosition, cell.pos);
            }
        }

        accepter.wait();
    }

    void testBoostSerialization1()
    {
#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
        Coord<2> dim(30, 20);
        CoordBox<2> box(Coord<2>(), dim);
        Region<2> boxRegion;
        boxRegion << box;

        GridType3 sendGrid(box);
        GridType3 recvGrid(box);

        // Set up senders (i.e. PatchAccepters)
        std::vector<Region<2> > regions(mpiLayer->size());
        for (int i = 0; i < mpiLayer->size(); ++i) {
            regions[i] << Streak<2>(Coord<2>(0, i), dim.x());;
        }

        PatchLink<GridType3>::Accepter accepter(
            regions[mpiLayer->rank()],
            0,
            2701,
            MPI_CHAR);
        accepter.charge(4, 10, 1);

        // Set up receivers (i.e. PatchProviders)
        std::vector<SharedPtr<PatchLink<GridType3>::Provider>::Type> providers;
        if (mpiLayer->rank() == 0) {
            for (int i = 0; i < mpiLayer->size(); ++i) {
                providers.push_back(
                    SharedPtr<PatchLink<GridType3>::Provider>::Type(
                        new PatchLink<GridType3>::Provider(
                            regions[i],
                            i,
                            2701,
                            MPI_CHAR)));

                providers.back()->charge(4, 10, 1);
            }
        }

        for (int t = 4; t < 10; ++t) {
            // prep and send data:
            for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
                MyComplicatedCell cell;
                cell.cargo << i->x();
                cell.cargo << i->y();
                cell.cargo << mpiLayer->rank();
                cell.x = t;
                sendGrid.set(*i, cell);
            }
            accepter.put(sendGrid, boxRegion, dim, t, mpiLayer->rank());
            accepter.wait();

            // receive and check data:
            if (mpiLayer->rank() == 0) {
                for (int i = 0; i < mpiLayer->size(); ++i) {
                    providers[i]->get(&recvGrid, boxRegion, dim, t, i);
                }

                for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
                    MyComplicatedCell cell = recvGrid.get(*i);

                    if (i->y() < mpiLayer->size()) {
                        TS_ASSERT_EQUALS(cell.cargo.size(), std::size_t(3));
                        TS_ASSERT_EQUALS(cell.cargo[0], i->x());
                        TS_ASSERT_EQUALS(cell.cargo[1], i->y());
                        TS_ASSERT_EQUALS(cell.cargo[2], i->y());
                        TS_ASSERT_EQUALS(cell.x,        t);
                    } else {
                        TS_ASSERT_EQUALS(cell.cargo.size(), std::size_t(0));
                    }
                }
            }
        }
#endif
    }

    void testBoostSerialization2()
    {
#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
        typedef DisplacedGrid<NonPoDTestCell> GridType4;
        NonPoDTestCell::Initializer initializer;

        // set up geometry data for a block stripe decomposition:
        CoordBox<2> globalGridBox = initializer.gridBox();
        int dimX = globalGridBox.dimensions.x();
        int startRow = globalGridBox.dimensions.y() * (mpiLayer->rank() + 0) / mpiLayer->size();
        int endRow   = globalGridBox.dimensions.y() * (mpiLayer->rank() + 1) / mpiLayer->size();

        int upperGhostHeight = (mpiLayer->rank() == 0) ? 0 : 1;
        int lowerGhostHeight = (mpiLayer->rank() == 3) ? 0 : 1;

        int paddedStartRow = startRow - upperGhostHeight;
        int paddedEndRow = endRow + lowerGhostHeight;
        int paddedHeight = paddedEndRow - paddedStartRow;
        CoordBox<2> ownGridBox(
            Coord<2>(0, startRow),
            Coord<2>(dimX, endRow - startRow));
        CoordBox<2> paddedGridBox(
            Coord<2>(0, paddedStartRow),
            Coord<2>(dimX, paddedHeight));

        Region<2> upperRecvRegion;
        Region<2> lowerRecvRegion;
        Region<2> upperSendRegion;
        Region<2> lowerSendRegion;

        upperRecvRegion << CoordBox<2>(Coord<2>(0, startRow - upperGhostHeight), Coord<2>(dimX, upperGhostHeight));
        lowerRecvRegion << CoordBox<2>(Coord<2>(0, endRow),                      Coord<2>(dimX, lowerGhostHeight));
        upperSendRegion << CoordBox<2>(Coord<2>(0, startRow),                    Coord<2>(dimX, upperGhostHeight));
        lowerSendRegion << CoordBox<2>(Coord<2>(0, endRow   - lowerGhostHeight), Coord<2>(dimX, lowerGhostHeight));

        Region<2> totalRegion;
        Region<2> updateRegion;
        totalRegion << paddedGridBox;
        updateRegion << ownGridBox;

        // init grids:
        GridType4 grid1(paddedGridBox);
        GridType4 grid2(paddedGridBox);
        initializer.grid(&grid1);
        initializer.grid(&grid2);

        // set up patch links
        int maxNanoSteps = 10;
        typedef PatchLink<GridType4>::Accepter PatchAccepterType;
        typedef PatchLink<GridType4>::Provider PatchProviderType;

        std::vector<SharedPtr<PatchAccepterType>::Type> accepters;
        std::vector<SharedPtr<PatchProviderType>::Type> providers;
        if (upperGhostHeight > 0) {
            accepters << SharedPtr<PatchAccepterType>::Type(
                new PatchAccepterType(
                    upperSendRegion,
                    mpiLayer->rank() - 1,
                    666,
                    MPI_CHAR));
            providers << SharedPtr<PatchProviderType>::Type(
                new PatchProviderType(
                    upperRecvRegion,
                    mpiLayer->rank() - 1,
                    666,
                    MPI_CHAR));
            accepters.back()->charge(0, maxNanoSteps, 1);
            providers.back()->charge(0, maxNanoSteps, 1);
        }
        if (lowerGhostHeight > 0) {
            accepters << SharedPtr<PatchAccepterType>::Type(
                new PatchAccepterType(
                    lowerSendRegion,
                    mpiLayer->rank() + 1,
                    666,
                    MPI_CHAR));
            providers << SharedPtr<PatchProviderType>::Type(
                new PatchProviderType(
                    lowerRecvRegion,
                    mpiLayer->rank() + 1,
                    666,
                    MPI_CHAR));
            accepters.back()->charge(0, maxNanoSteps, 1);
            providers.back()->charge(0, maxNanoSteps, 1);
        }

        for (int t = 0; t < maxNanoSteps; ++t) {
            UpdateFunctor<NonPoDTestCell> updateFunctor;
            updateFunctor(updateRegion, Coord<2>(), Coord<2>(), grid1, &grid2, t);

            for (auto&& i: accepters) {
                i->put(
                    grid2,
                    updateRegion,
                    globalGridBox.dimensions,
                    t,
                    mpiLayer->rank());
            }

            for (auto&& i: providers) {
                i->get(
                    &grid2,
                    totalRegion,
                    globalGridBox.dimensions,
                    t,
                    mpiLayer->rank());
            }

            for (auto&& i: accepters) {
                i->wait();
            }

            using std::swap;
            swap(grid1, grid2);
        }
#endif
    }

private:
    int tag;

    GridType zeroGrid;
    GridType sendGrid1;
    GridType sendGrid2;
    GridType expected;
    GridType actual;

    CoordBox<2> boundingBox;
    Region<2> boundingRegion;
    Region<2> region1;
    Region<2> region2;

    SharedPtr<PatchAccepterType>::Type acc;
    SharedPtr<PatchProviderType>::Type pro;
    SharedPtr<MPILayer>::Type mpiLayer;

    GridType markGrid(const Region<2>& region, int id)
    {
        GridType ret = zeroGrid;

        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            ret[*i] = id + i->y() * 10 + i->x();
        }

        return ret;
    }

    int genTag(int from, int to) {
        return 100 + from * 10 + to;
    }
};

}
