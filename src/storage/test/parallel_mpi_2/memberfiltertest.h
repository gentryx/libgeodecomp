#include <vector>

#include <cxxtest/TestSuite.h>
#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/storage/memberfilter.h>
#include <libgeodecomp/storage/soagrid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class MemberFilterTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        // required to ensure initialization of Typemaps.
        MPILayer();

        // sanity tests to ensure that typemaps are actually initialized.
        TS_ASSERT_DIFFERS(Typemaps::lookup<Coord<2> >(), MPI_INT);
        TS_ASSERT_DIFFERS(Typemaps::lookup<Coord<2> >(), Typemaps::lookup<Coord<3> >());
    }

    void testBasics()
    {
        typedef SharedPtr<FilterBase<TestCell<2> > >::Type FilterPtr;
        FilterPtr filter(new MemberFilter<TestCell<2>, Coord<2> >(&Coord<2>::c));

        Selector<TestCell<2> > selector(
            &TestCell<2>::dimensions,
            "pos",
            filter);

        TS_ASSERT_DIFFERS(MPI_INT, MPI_CHAR);
        TS_ASSERT_EQUALS(MPI_INT, selector.mpiDatatype());
    }

    void testHostAoS()
    {
        typedef SharedPtr<FilterBase<TestCell<2> > >::Type FilterPtr;
        FilterPtr filter(new MemberFilter<TestCell<2>, CoordBox<2> >(&CoordBox<2>::dimensions));

        Selector<TestCell<2> > selector(
            &TestCell<2>::dimensions,
            "dimensions",
            filter);

        TS_ASSERT_EQUALS(Typemaps::lookup<Coord<2> >(), selector.mpiDatatype());

    }

    void testHostSoA()
    {
        typedef SharedPtr<FilterBase<TestCellSoA> >::Type FilterPtr;
        FilterPtr filter(new MemberFilter<TestCellSoA, CoordBox<3> >(&CoordBox<3>::dimensions));

        Selector<TestCellSoA> selector(
            &TestCellSoA::dimensions,
            "dimensions",
            filter);

        TS_ASSERT_DIFFERS(Typemaps::lookup<Coord<3> >(), MPI_INT);
        TS_ASSERT_EQUALS(Typemaps::lookup<Coord<3> >(), selector.mpiDatatype());
    }

};

}
