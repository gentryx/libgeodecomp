#include <cxxtest/TestSuite.h>
#include <libgeodecomp/storage/fixedneighborhoodupdatefunctor.h>
#include <libgeodecomp/storage/soagrid.h>
#include <libgeodecomp/storage/updatefunctor.h>
#include <libgeodecomp/parallelization/nesting/offsethelper.h>

using namespace LibGeoDecomp;

class MySoATestCellWithTwoDoublesTorus
{
public:
    class API :
          public LibGeoDecomp::APITraits::HasFixedCoordsOnlyUpdate,
          public LibGeoDecomp::APITraits::HasUpdateLineX,
          public LibGeoDecomp::APITraits::HasStencil<LibGeoDecomp::Stencils::Moore<3, 1> >,
          public LibGeoDecomp::APITraits::HasTorusTopology<3>,
          public LibGeoDecomp::APITraits::HasSoA
    {};

    inline
    explicit MySoATestCellWithTwoDoublesTorus(const double valA = 0, const double valB = 0) :
        valA(valA),
        valB(valB)
    {}

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, const int nanoStep)
    {}

    template<typename HOOD_OLD, typename HOOD_NEW>
    static void updateLineX(HOOD_OLD& hoodOld, long indexEnd,
                            HOOD_NEW& hoodNew, long /* nanoStep */)
    {
        for (; hoodOld.index() < indexEnd; hoodOld += 1, hoodNew += 1) {
            double akku =
                hoodOld[FixedCoord< 0,  0, -1>()].valA() +
                hoodOld[FixedCoord< 0, -1,  0>()].valA() +
                hoodOld[FixedCoord<-1,  0,  0>()].valA() +
                hoodOld[FixedCoord< 1,  0,  0>()].valA() +
                hoodOld[FixedCoord< 0,  1,  0>()].valA() +
                hoodOld[FixedCoord< 0,  0,  1>()].valA();

            hoodNew.valB() = akku;
        }
    }

    double valA;
    double valB;
};

class MySoATestCellWithTwoDoublesCube
{
public:
    class API :
          public LibGeoDecomp::APITraits::HasFixedCoordsOnlyUpdate,
          public LibGeoDecomp::APITraits::HasUpdateLineX,
          public LibGeoDecomp::APITraits::HasStencil<LibGeoDecomp::Stencils::Moore<3, 1> >,
          public LibGeoDecomp::APITraits::HasCubeTopology<3>,
          public LibGeoDecomp::APITraits::HasSoA
    {};

    inline
    explicit MySoATestCellWithTwoDoublesCube(const double valA = 0, const double valB = 0) :
        valA(valA),
        valB(valB)
    {}

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, const int nanoStep)
    {}

    template<typename HOOD_OLD, typename HOOD_NEW>
    static void updateLineX(HOOD_OLD& hoodOld, long indexEnd,
                            HOOD_NEW& hoodNew, long /* nanoStep */)
    {
        for (; hoodOld.index() < indexEnd; hoodOld += 1, hoodNew += 1) {
            double akku =
                hoodOld[FixedCoord< 0,  0, -1>()].valA() +
                hoodOld[FixedCoord< 0, -1,  0>()].valA() +
                hoodOld[FixedCoord<-1,  0,  0>()].valA() +
                hoodOld[FixedCoord< 1,  0,  0>()].valA() +
                hoodOld[FixedCoord< 0,  1,  0>()].valA() +
                hoodOld[FixedCoord< 0,  0,  1>()].valA();
            hoodNew.valB() = akku;
        }
    }

    double valA;
    double valB;
};

LIBFLATARRAY_REGISTER_SOA(
    MySoATestCellWithTwoDoublesTorus,
    ((double)(valA))
    ((double)(valB)))

LIBFLATARRAY_REGISTER_SOA(
    MySoATestCellWithTwoDoublesCube,
    ((double)(valA))
    ((double)(valB)))


namespace LibGeoDecomp {

class FixedNeighborhoodUpdateFunctorTest : public CxxTest::TestSuite
{
public:

    void testTorus()
    {
        CoordBox<3> box(Coord<3>(10, 20, 30), Coord<3>(200, 100, 50));

        MySoATestCellWithTwoDoublesTorus defaultCell(666, 777);
        MySoATestCellWithTwoDoublesTorus edgeCell(-1, -1);
        SoAGrid<MySoATestCellWithTwoDoublesTorus, Topologies::Torus<3>::Topology> gridOld(box, defaultCell, edgeCell);
        SoAGrid<MySoATestCellWithTwoDoublesTorus, Topologies::Torus<3>::Topology> gridNew(box, defaultCell, edgeCell);

        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            gridOld.set(*i, MySoATestCellWithTwoDoublesTorus(i->x() + i->y() * 1000.0 + i->z() * 1000 * 1000.0));
        }

        Region<3> region;
        region
            << Streak<3>(Coord<3>(10,  20, 30), 209)
            << Streak<3>(Coord<3>(10,  21, 30), 210)
            << Streak<3>(Coord<3>(10,  30, 30), 210)
            << Streak<3>(Coord<3>(10,  30, 40), 210)
            << Streak<3>(Coord<3>(25,  31, 40), 200)
            << Streak<3>(Coord<3>(25, 119, 40), 200)
            << Streak<3>(Coord<3>(10,  30, 49), 210);

        CoordBox<3> boxNew = gridNew.boundingBox();
        CoordBox<3> boxOld = gridOld.boundingBox();
        Coord<3> offsetOld = -boxOld.origin;
        Coord<3> offsetNew = -boxNew.origin;
        Coord<3> topoDim = boxNew.dimensions;

        gridOld.callback(&gridNew, FixedNeighborhoodUpdateFunctor<
                         MySoATestCellWithTwoDoublesTorus,
                         UpdateFunctorHelpers::ConcurrencyNoP,
                         APITraits::SelectThreadedUpdate<void>::Value>(
                             &region,
                             &offsetOld,
                             &offsetNew,
                             &boxNew.dimensions,
                             &boxNew.dimensions,
                             &topoDim,
                             0,
                             0,
                             0));

        for (Region<3>::Iterator i = region.begin(); i != region.end(); ++i) {
            MySoATestCellWithTwoDoublesTorus cell = gridNew.get(*i);

            Coord<3> sum;
            sum += normalizeTorus(*i + Coord<3>( 0,  0, -1), box);
            sum += normalizeTorus(*i + Coord<3>( 0, -1,  0), box);
            sum += normalizeTorus(*i + Coord<3>(-1,  0,  0), box);
            sum += normalizeTorus(*i + Coord<3>( 1,  0,  0), box);
            sum += normalizeTorus(*i + Coord<3>( 0,  1,  0), box);
            sum += normalizeTorus(*i + Coord<3>( 0,  0,  1), box);

            double expected = sum.x() + sum.y() * 1000.0 + sum.z() * 1000 * 1000.0;
            TS_ASSERT_EQUALS(cell.valB, expected);
        }
    }

    void testTorusWithOffset()
    {
        Region<3> region;
        region << CoordBox<3>(Coord<3>(  0,   0,   0), Coord<3>(10, 10, 10));
        region << CoordBox<3>(Coord<3>(190, 190, 190), Coord<3>(10, 10, 10));

        const CoordBox<3> gridBox = CoordBox<3>(Coord<3>(0, 0, 0), Coord<3>(200, 200, 200));
        const int ghostZoneWidth = 1;

        Coord<3> offset;
        Coord<3> dimensions;

        Region<3> expandedRegion = region.expandWithTopology(
            ghostZoneWidth,
            gridBox.dimensions,
            Topologies::Torus<3>::Topology());

        OffsetHelper<3 - 1, 3, Topologies::Torus<3>::Topology>()(
            &offset,
            &dimensions,
            expandedRegion,
            gridBox);

        TS_ASSERT_EQUALS(dimensions, Coord<3>(22, 22, 22));

        CoordBox<3> shiftedBox(offset, dimensions);
        CoordBox<3> topoBox(region.boundingBox());

        MySoATestCellWithTwoDoublesTorus defaultCell(666, 777);
        MySoATestCellWithTwoDoublesTorus edgeCell(-1, -1);
        SoAGrid<MySoATestCellWithTwoDoublesTorus, Topologies::Torus<3>::Topology, true> gridOld(
            shiftedBox,
            defaultCell,
            edgeCell,
            gridBox.dimensions);
        SoAGrid<MySoATestCellWithTwoDoublesTorus, Topologies::Torus<3>::Topology, true> gridNew(
            shiftedBox,
            defaultCell,
            edgeCell,
            gridBox.dimensions);

        for (CoordBox<3>::Iterator i = shiftedBox.begin(); i != shiftedBox.end(); ++i) {
            Coord<3> c = normalizeTorus(*i, gridBox);
            double val =
                c.x() +
                c.y() * 1000.0 +
                c.z() * 1000.0 * 1000.0;
            gridOld.set(*i, MySoATestCellWithTwoDoublesTorus(val));
        }

        CoordBox<3> boxNew = gridNew.boundingBox();
        CoordBox<3> boxOld = gridOld.boundingBox();
        Coord<3> offsetOld = -boxOld.origin;
        Coord<3> offsetNew = -boxNew.origin;
        gridOld.callback(&gridNew, FixedNeighborhoodUpdateFunctor<
                         MySoATestCellWithTwoDoublesTorus,
                         UpdateFunctorHelpers::ConcurrencyNoP,
                         APITraits::SelectThreadedUpdate<void>::Value>(
                             &region,
                             &offsetOld,
                             &offsetNew,
                             &boxOld.dimensions,
                             &boxNew.dimensions,
                             &gridBox.dimensions,
                             0,
                             0,
                             0));

        for (Region<3>::Iterator i = region.begin(); i != region.end(); ++i) {
            MySoATestCellWithTwoDoublesTorus cell = gridNew.get(*i);

            Coord<3> sum;
            sum += normalizeTorus(*i + Coord<3>( 0,  0, -1), topoBox);
            sum += normalizeTorus(*i + Coord<3>( 0, -1,  0), topoBox);
            sum += normalizeTorus(*i + Coord<3>(-1,  0,  0), topoBox);
            sum += normalizeTorus(*i + Coord<3>( 1,  0,  0), topoBox);
            sum += normalizeTorus(*i + Coord<3>( 0,  1,  0), topoBox);
            sum += normalizeTorus(*i + Coord<3>( 0,  0,  1), topoBox);

            double expected = sum.x() + sum.y() * 1000.0 + sum.z() * 1000 * 1000.0;
            TS_ASSERT_EQUALS(cell.valB, expected);
        }
    }

    void testCube()
    {
        CoordBox<3> box(Coord<3>(10, 20, 30), Coord<3>(200, 100, 50));

        MySoATestCellWithTwoDoublesCube defaultCell1(666, 777);
        MySoATestCellWithTwoDoublesCube defaultCell2(888, 999);
        MySoATestCellWithTwoDoublesCube edgeCell(-1001001.0, -1);
        SoAGrid<MySoATestCellWithTwoDoublesCube, Topologies::Cube<3>::Topology> gridOld(box, defaultCell1, edgeCell);
        SoAGrid<MySoATestCellWithTwoDoublesCube, Topologies::Cube<3>::Topology> gridNew(box, defaultCell2, edgeCell);

        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            gridOld.set(*i, MySoATestCellWithTwoDoublesCube(i->x() + i->y() * 1000.0 + i->z() * 1000 * 1000.0));
        }

        Region<3> region;
        region
            << Streak<3>(Coord<3>(10,  20, 30), 209)
            << Streak<3>(Coord<3>(10,  21, 30), 210)
            << Streak<3>(Coord<3>(10,  30, 30), 210)
            << Streak<3>(Coord<3>(10,  30, 40), 210)
            << Streak<3>(Coord<3>(25,  31, 40), 200)
            << Streak<3>(Coord<3>(25, 119, 40), 200)
            << Streak<3>(Coord<3>(10,  30, 49), 210);

        Coord<3> topoDim(500, 500, 500);
        CoordBox<3> boxNew = gridNew.boundingBox();
        CoordBox<3> boxOld = gridOld.boundingBox();
        Coord<3> offsetOld = -boxOld.origin + gridOld.getEdgeRadii();
        Coord<3> offsetNew = -boxNew.origin + gridNew.getEdgeRadii();
        gridOld.callback(&gridNew, FixedNeighborhoodUpdateFunctor<
                         MySoATestCellWithTwoDoublesCube,
                         UpdateFunctorHelpers::ConcurrencyNoP,
                         APITraits::SelectThreadedUpdate<void>::Value>(
                             &region,
                             &offsetOld,
                             &offsetNew,
                             &boxNew.dimensions,
                             &boxNew.dimensions,
                             &topoDim,
                             0,
                             0,
                             0));

        for (Region<3>::Iterator i = region.begin(); i != region.end(); ++i) {
            MySoATestCellWithTwoDoublesCube cell = gridNew.get(*i);

            Coord<3> sum;
            sum += normalizeCube(*i + Coord<3>( 0,  0, -1), box);
            sum += normalizeCube(*i + Coord<3>( 0, -1,  0), box);
            sum += normalizeCube(*i + Coord<3>(-1,  0,  0), box);
            sum += normalizeCube(*i + Coord<3>( 1,  0,  0), box);
            sum += normalizeCube(*i + Coord<3>( 0,  1,  0), box);
            sum += normalizeCube(*i + Coord<3>( 0,  0,  1), box);

            double expected = sum.x() + sum.y() * 1000.0 + sum.z() * 1000 * 1000.0;
            TS_ASSERT_EQUALS(cell.valB, expected);
        }
    }

private:
    Coord<3> normalizeTorus(const Coord<3>& coord, const CoordBox<3>& box)
    {
        return box.origin + Topologies::Torus<3>::Topology::normalize(coord - box.origin, box.dimensions);
    }

    Coord<3> normalizeCube(const Coord<3>& coord, const CoordBox<3>& box)
    {
        if (!box.inBounds(coord)) {
            return Coord<3>(-1, -1, -1);
        }

        return coord;
    }
};

}
