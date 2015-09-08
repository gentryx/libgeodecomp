#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/config.h>
#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/storage/selector.h>
#include <libgeodecomp/storage/soagrid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class FooCell
{
public:
    class API : public APITraits::HasOpaqueMPIDataType<FooCell>
    {};

    double d;
    char c;
    int i;
};

class SelectorTest : public CxxTest::TestSuite
{
public:

    void testMPIType()
    {
        MPI_Datatype mpiType;

        mpiType = Selector<int>().mpiDatatype();
        TS_ASSERT_EQUALS(mpiType, MPI_INT);

        mpiType = Selector<char>().mpiDatatype();
        TS_ASSERT_EQUALS(mpiType, MPI_CHAR);

        mpiType = Selector<double>().mpiDatatype();
        TS_ASSERT_EQUALS(mpiType, MPI_DOUBLE);

        Selector<FooCell> selector1(&FooCell::d, "d");
        Selector<FooCell> selector2(&FooCell::c, "c");

        mpiType = selector1.mpiDatatype();
        TS_ASSERT_EQUALS(mpiType, MPI_DOUBLE);

        mpiType = selector2.mpiDatatype();
        TS_ASSERT_EQUALS(mpiType, MPI_CHAR);
    }

};

}
