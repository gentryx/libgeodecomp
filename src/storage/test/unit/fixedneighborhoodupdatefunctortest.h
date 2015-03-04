#include <cxxtest/TestSuite.h>
#include <libgeodecomp/storage/fixedneighborhoodupdatefunctor.h>
#include <libgeodecomp/storage/soagrid.h>

using namespace LibGeoDecomp;

class MySoATestCell
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
    MySoATestCell(const double valA = 0, const double valB = 0) :
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
        std::cout << "foobar " << hoodOld.index << " -> " << indexEnd << "\n";
    }

    double valA;
    double valB;
};

LIBFLATARRAY_REGISTER_SOA(
    MySoATestCell,
    ((double)(valA))
    ((double)(valB)))


namespace LibGeoDecomp {

class FixedNeighborhoodUpdateFunctorTest : public CxxTest::TestSuite
{
public:

    void testBasic()
    {
        CoordBox<3> box(Coord<3>(10, 20, 30), Coord<3>(200, 100, 50));
        std::cout << "BOOOOOOOOOOOOOOOOOOOOOOOOOMER\n";
        MySoATestCell defaultCell(666, 777);
        MySoATestCell edgeCell(-1, -1);
        SoAGrid<MySoATestCell, Topologies::Torus<3>::Topology> gridOld(box, defaultCell, edgeCell);
        SoAGrid<MySoATestCell, Topologies::Torus<3>::Topology> gridNew(box, defaultCell, edgeCell);

        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            gridOld.set(*i, MySoATestCell(i->x() + i->y() * 1000.0 + i->z() * 1000 * 1000.0));
        }

        std::cout << "BOOOOOOOOOOOOOOOOOOOOOOOOOMER\n";

        Region<3> region;
        region << Streak<3>(Coord<3>(10,  20, 30), 210)
               << Streak<3>(Coord<3>(10,  30, 30), 210)
               << Streak<3>(Coord<3>(10,  30, 50), 210)
               << Streak<3>(Coord<3>(25,  31, 50), 200)
               << Streak<3>(Coord<3>(25, 119, 50), 200)
               << Streak<3>(Coord<3>(10,  30, 79), 210);

        gridOld.callback(&gridNew, FixedNeighborhoodUpdateFunctor<MySoATestCell>(&region, gridOld.boundingBox(), 0));

        std::cout << "BOOOOOOOOOOOOOOOOOOOOOOOOOMER\n";
    }

    // fixme: test with cube topo
    // fixme: test with torus topo and displacements (as the z-curve would imply them with pars of the region sitting on opposite edges of the simulation space)
};

}
