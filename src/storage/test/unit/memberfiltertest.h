#include <vector>

#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/storage/memberfilter.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class MemberFilterTest : public CxxTest::TestSuite
{
public:
    void testHostAoS()
    {
        SharedPtr<FilterBase<TestCell<2> > >::Type filter(new MemberFilter<TestCell<2>, CoordBox<2> >(&CoordBox<2>::dimensions));
        Selector<TestCell<2> > selector(
            &TestCell<2>::dimensions,
            "dimensions",
            filter);

        std::vector<TestCell<2> > vec;
        for (int i = 0; i < 30; ++i) {
            TestCell<2> cell;
            cell.dimensions = CoordBox<2>(Coord<2>(i + 100, i + 200), Coord<2>(i + 300, i + 400));
            vec << cell;
        }

        std::vector<Coord<2> > extract(vec.size());
        selector.copyMemberOut(&vec[0], MemoryLocation::HOST, (char*)&extract[0], MemoryLocation::HOST, vec.size());

        for (std::size_t i = 0; i < vec.size(); ++i) {
            TS_ASSERT_EQUALS(Coord<2>(i + 300, i + 400), extract[i]);
        }

        for (std::size_t i = 0; i < vec.size(); ++i) {
            extract[i] = Coord<2>(i + 500, i + 600);
        }
        selector.copyMemberIn((char*)&extract[0], MemoryLocation::HOST, &vec[0], MemoryLocation::HOST, vec.size());

        for (std::size_t i = 0; i < vec.size(); ++i) {
            TS_ASSERT_EQUALS(Coord<2>(i + 500, i + 600), vec[i].dimensions.dimensions);
        }
    }

    // fixme: also add cuda tests
    // fixme: add test for nesting
};

}
