#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/io/selector.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
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


MPI_Datatype FooCell::MPIDataType = MPI_DATATYPE_NULL;

class SelectorTest : public CxxTest::TestSuite
{
public:

    void testMPIType()
    {
        MPI_Datatype mpiType;

        mpiType = SelectorHelpers::GetMPIDatatype<int>()();
        TS_ASSERT_EQUALS(mpiType, MPI_INT);

        mpiType = SelectorHelpers::GetMPIDatatype<char>()();
        TS_ASSERT_EQUALS(mpiType, MPI_CHAR);

        mpiType = SelectorHelpers::GetMPIDatatype<double>()();
        TS_ASSERT_EQUALS(mpiType, MPI_DOUBLE);

        mpiType = SelectorHelpers::GetMPIDatatype<FooCell>()();
        TS_ASSERT_EQUALS(mpiType, FooCell::MPIDataType);

        TS_ASSERT_THROWS(SelectorHelpers::GetMPIDatatype<SelectorTest>()(), std::invalid_argument);

        Selector<FooCell> selector1(&FooCell::d, "d");
        Selector<FooCell> selector2(&FooCell::c, "c");

        mpiType = selector1.mpiDatatype();
        TS_ASSERT_EQUALS(mpiType, MPI_DOUBLE);

        mpiType = selector2.mpiDatatype();
        TS_ASSERT_EQUALS(mpiType, MPI_CHAR);
    }

};

}
