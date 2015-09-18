#include <cxxtest/TestSuite.h>
#include <libgeodecomp/storage/fixedneighborhoodupdatefunctor.h>
#include <libgeodecomp/storage/soagrid.h>

using namespace LibGeoDecomp;

class MySoATestCellWithTwoDoubles
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
    MySoATestCellWithTwoDoubles(const double valA = 0, const double valB = 0) :
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
    MySoATestCellWithTwoDoubles,
    ((double)(valA))
    ((double)(valB)))


namespace LibGeoDecomp {

class FixedNeighborhoodUpdateFunctorTest : public CxxTest::TestSuite
{
public:

    void testBasic()
    {
        CoordBox<3> box(Coord<3>(10, 20, 30), Coord<3>(200, 100, 50));

        MySoATestCellWithTwoDoubles defaultCell(666, 777);
        MySoATestCellWithTwoDoubles edgeCell(-1, -1);
        SoAGrid<MySoATestCellWithTwoDoubles, Topologies::Torus<3>::Topology> gridOld(box, defaultCell, edgeCell);
        SoAGrid<MySoATestCellWithTwoDoubles, Topologies::Torus<3>::Topology> gridNew(box, defaultCell, edgeCell);

        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            gridOld.set(*i, MySoATestCellWithTwoDoubles(i->x() + i->y() * 1000.0 + i->z() * 1000 * 1000.0));
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
        gridOld.callback(&gridNew, FixedNeighborhoodUpdateFunctor<MySoATestCellWithTwoDoubles>(
                             &region, &offsetOld, &offsetNew, &boxNew.dimensions, 0));

        for (Region<3>::Iterator i = region.begin(); i != region.end(); ++i) {
            MySoATestCellWithTwoDoubles cell = gridNew.get(*i);

            Coord<3> sum;
            sum += normalize(*i + Coord<3>( 0,  0, -1), box);
            sum += normalize(*i + Coord<3>( 0, -1,  0), box);
            sum += normalize(*i + Coord<3>(-1,  0,  0), box);
            sum += normalize(*i + Coord<3>( 1,  0,  0), box);
            sum += normalize(*i + Coord<3>( 0,  1,  0), box);
            sum += normalize(*i + Coord<3>( 0,  0,  1), box);

            double expected = sum.x() + sum.y() * 1000.0 + sum.z() * 1000 * 1000.0;
            double delta = cell.valB - expected;
            TS_ASSERT_EQUALS(cell.valB, expected);
        }
    }

    // fixme: test with cube topo
    // fixme: test with torus topo and displacements (as the z-curve would imply them with pars of the region sitting on opposite edges of the simulation space)

private:
    Coord<3> normalize(const Coord<3>& coord, const CoordBox<3>& box)
    {
        return box.origin + Topologies::Torus<3>::Topology::normalize(coord - box.origin, box.dimensions);
    }
};

}
