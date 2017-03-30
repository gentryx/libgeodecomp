#include <vector>

#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/storage/memberfilter.h>
#include <libgeodecomp/storage/soagrid.h>

using namespace LibGeoDecomp;

class MemberMemberMember
{
public:
    double a;
    int b;
};

class MemberMember
{
public:
    MemberMemberMember member;
};

class Member
{
public:
    MemberMember member;
    float c;
};

class TestClassForMultiLevelNesting
{
public:
    Member member;
    double d;
};

namespace LibGeoDecomp {

class MemberFilterTest : public CxxTest::TestSuite
{
public:
    void testBasics()
    {
        typedef SharedPtr<FilterBase<TestCell<2> > >::Type FilterPtr;
        FilterPtr filter(new MemberFilter<TestCell<2>, Coord<2> >(&Coord<2>::c));

        Selector<TestCell<2> > selector(
            &TestCell<2>::dimensions,
            "pos",
            filter);

        TS_ASSERT_EQUALS(8, filter->sizeOf());
#ifdef LIBGEODECOMP_WITH_SILO
        TS_ASSERT_EQUALS(DB_INT, selector.siloTypeID());
#endif
        TS_ASSERT_EQUALS("INT", selector.typeName());
        TS_ASSERT_EQUALS(2, selector.arity());
    }

    void testHostAoS()
    {
        typedef SharedPtr<FilterBase<TestCell<2> > >::Type FilterPtr;
        FilterPtr filter(new MemberFilter<TestCell<2>, CoordBox<2> >(&CoordBox<2>::dimensions));

        Selector<TestCell<2> > selector(
            &TestCell<2>::dimensions,
            "dimensions",
            filter);

        std::vector<TestCell<2> > vec;
        for (int i = 0; i < 44; ++i) {
            TestCell<2> cell;
            cell.dimensions = CoordBox<2>(Coord<2>(i + 100, i + 200), Coord<2>(i + 300, i + 400));
            vec << cell;
        }

        std::vector<Coord<2> > extract(vec.size());
        selector.copyMemberOut(
            &vec[0],
            MemoryLocation::HOST,
            reinterpret_cast<char*>(&extract[0]),
            MemoryLocation::HOST,
            vec.size());

        for (std::size_t i = 0; i < vec.size(); ++i) {
            Coord<2> c(
                static_cast<int>(i) + 300,
                static_cast<int>(i) + 400);
            TS_ASSERT_EQUALS(c, extract[i]);
        }

        for (std::size_t i = 0; i < vec.size(); ++i) {
            extract[i] = Coord<2>(
                static_cast<int>(i) + 500,
                static_cast<int>(i) + 600);
        }
        selector.copyMemberIn(
            reinterpret_cast<char*>(&extract[0]),
            MemoryLocation::HOST,
            &vec[0],
            MemoryLocation::HOST,
            vec.size());

        for (std::size_t i = 0; i < vec.size(); ++i) {
            Coord<2> c(
                static_cast<int>(i) + 500,
                static_cast<int>(i) + 600);
            TS_ASSERT_EQUALS(c, vec[i].dimensions.dimensions);
        }

        TS_ASSERT_EQUALS(sizeof(Coord<2>), filter->sizeOf());
        TS_ASSERT_EQUALS(1, selector.arity());
    }

    void testNesting()
    {
        typedef SharedPtr<FilterBase<MemberMember> >::Type FilterPtr1;
        FilterPtr1 filter1(new MemberFilter<MemberMember, MemberMemberMember>(&MemberMemberMember::a));

        typedef SharedPtr<FilterBase<Member> >::Type FilterPtr2;
        FilterPtr2 filter2(new MemberFilter<Member, MemberMember>(&MemberMember::member, filter1));

        typedef SharedPtr<FilterBase<TestClassForMultiLevelNesting> >::Type FilterPtr3;
        FilterPtr3 filter3(new MemberFilter<TestClassForMultiLevelNesting, Member>(&Member::member, filter2));

        Selector<TestClassForMultiLevelNesting> selector(
            &TestClassForMultiLevelNesting::member,
            "member",
            filter3);

        std::vector<TestClassForMultiLevelNesting> vec(55);
        for (std::size_t i = 0; i < 55; ++i) {
            vec[i].member.member.member.a = i + 0.1;
        }

        std::vector<double> buf(55);
        selector.copyMemberOut(
            vec.data(),
            MemoryLocation::HOST,
            reinterpret_cast<char*>(buf.data()),
            MemoryLocation::HOST,
            55);

        for (std::size_t i = 0; i < 55; ++i) {
            TS_ASSERT_EQUALS((i + 0.1), buf[i]);
        }

        for (std::size_t i = 0; i < 55; ++i) {
            buf[i] = i + 0.2;
        }

        selector.copyMemberIn(
            reinterpret_cast<char*>(buf.data()),
            MemoryLocation::HOST,
            vec.data(),
            MemoryLocation::HOST,
            55);

        for (std::size_t i = 0; i < 55; ++i) {
            TS_ASSERT_EQUALS((i + 0.2), vec[i].member.member.member.a);
        }
    }

    void testHostSoA()
    {
        CoordBox<3> box(Coord<3>(100, 200, 300), Coord<3>(30, 20, 10));
        SoAGrid<TestCellSoA, Topologies::Torus<3>::Topology> grid(box);

        typedef SharedPtr<FilterBase<TestCellSoA> >::Type FilterPtr;
        FilterPtr filter(new MemberFilter<TestCellSoA, CoordBox<3> >(&CoordBox<3>::dimensions));

        Selector<TestCellSoA> selector(
            &TestCellSoA::dimensions,
            "dimensions",
            filter);

        for (int i = 0; i < 30; ++i) {
            TestCellSoA cell;
            cell.dimensions = CoordBox<3>(
                Coord<3>(i + 1000, i + 2000, i + 3000),
                Coord<3>(i + 4000, i + 5000, i + 6000));

            grid.set(Coord<3>(100 + i, 200, 300), cell);
        }

        std::vector<Coord<3> > vec(30);

        LibFlatArray::soa_accessor<TestCellSoA, 32, 32, 32, 0> accessor(
            reinterpret_cast<char*>(grid.data()), 0);

        selector.copyStreakOut(
            accessor.access_member(sizeof(CoordBox<3>), selector.offset()),
            MemoryLocation::HOST,
            reinterpret_cast<char*>(&vec[0]),
            MemoryLocation::HOST,
            30,
            32 * 32 * 32);

        for (int i = 0; i < 30; ++i) {
            TS_ASSERT_EQUALS(
                Coord<3>(i + 4000, i + 5000, i + 6000),
                vec[static_cast<std::size_t>(i)]);
        }

        for (int i = 0; i < 30; ++i) {
            vec[static_cast<std::size_t>(i)] = Coord<3>(i + 7777, i + 8888, i + 9999);
        }

        selector.copyStreakIn(
            reinterpret_cast<char*>(&vec[0]),
            MemoryLocation::HOST,
            accessor.access_member(sizeof(CoordBox<3>), selector.offset()),
            MemoryLocation::HOST,
            30,
            32 * 32 * 32);

        for (int i = 0; i < 30; ++i) {
            TS_ASSERT_EQUALS(
                Coord<3>(i + 7777, i + 8888, i + 9999),
                grid.get(Coord<3>(100 + i, 200, 300)).dimensions.dimensions);
        }

        TS_ASSERT_EQUALS(sizeof(Coord<3>), filter->sizeOf());
        TS_ASSERT_EQUALS(1, selector.arity());
    }
};

}
