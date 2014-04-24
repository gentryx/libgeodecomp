#include <libgeodecomp/storage/multicontainercell.h>

#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

DECLARE_MULTI_CONTAINER_CELL(DummyContainer,            \
                             ((std::string)(5)(labels)) \
                             ((double)(7)(prices))      \
                             )

class MultiContainerCellTest : public CxxTest::TestSuite
{
public:
    void testConstructionAndAccess()
    {
        DummyContainer cell;
        cell.labels.insert(10, "foo");
        cell.labels.insert(11, "bar");
        cell.labels.insert(15, "goo");

        cell.prices.insert(10, -666);
        cell.prices.insert(11, -0.11);
        cell.prices.insert(12, -0.12);
        cell.prices.insert(13, -0.13);
        cell.prices.insert(10, -0.10);

        TS_ASSERT_EQUALS(cell.labels.size(), std::size_t(3));
        TS_ASSERT_EQUALS(*cell.labels[10], "foo");
        TS_ASSERT_EQUALS(*cell.labels[11], "bar");
        TS_ASSERT_EQUALS(*cell.labels[15], "goo");

        TS_ASSERT_EQUALS(cell.prices.size(), std::size_t(4));
        TS_ASSERT_EQUALS(*cell.prices[10], -0.10);
        TS_ASSERT_EQUALS(*cell.prices[11], -0.11);
        TS_ASSERT_EQUALS(*cell.prices[12], -0.12);
        TS_ASSERT_EQUALS(*cell.prices[13], -0.13);
    }
};

}
