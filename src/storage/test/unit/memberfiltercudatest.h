#include <cxxtest/TestSuite.h>
#include <vector>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/storage/memberfilter.h>
#include <libgeodecomp/storage/soagrid.h>
#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_CUDA
#include <cuda.h>
#endif

using namespace LibGeoDecomp;

class Member
{
public:
    double a;
    float b[3];
};

class AnotherSimpleTestClass
{
public:
    double d;
    Member m;
};

LIBFLATARRAY_REGISTER_SOA(
    AnotherSimpleTestClass,
    ((double)(d))
    ((Member)(m)) )

namespace LibGeoDecomp {

class MemberFilterCudaTest : public CxxTest::TestSuite
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
            TS_ASSERT_EQUALS(Coord<2>(i + 300, i + 400), extract[i]);
        }

        for (std::size_t i = 0; i < vec.size(); ++i) {
            extract[i] = Coord<2>(i + 500, i + 600);
        }
        selector.copyMemberIn(
            reinterpret_cast<char*>(&extract[0]),
            MemoryLocation::HOST,
            &vec[0],
            MemoryLocation::HOST,
            vec.size());

        for (std::size_t i = 0; i < vec.size(); ++i) {
            TS_ASSERT_EQUALS(Coord<2>(i + 500, i + 600), vec[i].dimensions.dimensions);
        }

        TS_ASSERT_EQUALS(sizeof(Coord<2>), filter->sizeOf());
        TS_ASSERT_EQUALS(1, selector.arity());
    }

    // fixme: add test for nesting

    void testHostSoA()
    {
        LibFlatArray::cuda_array<CoordBox<3> > deviceBuffer1(30);
        LibFlatArray::cuda_array<Coord<3> > deviceBuffer2(30);
        std::vector<CoordBox<3> > hostBuffer1;
        std::vector<Coord<3> > hostBuffer2;

        typedef SharedPtr<FilterBase<TestCellSoA> >::Type FilterPtr;
        FilterPtr filter(new MemberFilter<TestCellSoA, CoordBox<3> >(&CoordBox<3>::dimensions));

        Selector<TestCellSoA> selector(
            &TestCellSoA::dimensions,
            "dimensions",
            filter);

        for (int i = 0; i < 30; ++i) {
            hostBuffer1 << CoordBox<3>(
                Coord<3>(i + 1000, i + 2000, i + 3000),
                Coord<3>(i + 4000, i + 5000, i + 6000));
        }

        deviceBuffer1.load(hostBuffer1.data());

        selector.copyStreakOut(
            reinterpret_cast<char*>(deviceBuffer1.data()),
            MemoryLocation::CUDA_DEVICE,
            reinterpret_cast<char*>(deviceBuffer2.data()),
            MemoryLocation::CUDA_DEVICE,
            30,
            32 * 32 * 32);

        hostBuffer2.resize(hostBuffer1.size());
        deviceBuffer2.save(hostBuffer2.data());

        for (int i = 0; i < 30; ++i) {
            TS_ASSERT_EQUALS(Coord<3>(i + 4000, i + 5000, i + 6000), hostBuffer2[i]);
        }

        for (int i = 0; i < 30; ++i) {
            hostBuffer2[i] = Coord<3>(i + 7777, i + 8888, i + 9999);
        }
        deviceBuffer2.load(hostBuffer2.data());

        // flush buffer to ensure proper data is being written back below:
        for (int i = 0; i < 30; ++i) {
            hostBuffer2[i] = Coord<3>(-1, -1, -1);
        }

        selector.copyStreakIn(
            reinterpret_cast<char*>(deviceBuffer2.data()),
            MemoryLocation::CUDA_DEVICE,
            reinterpret_cast<char*>(deviceBuffer1.data()),
            MemoryLocation::CUDA_DEVICE,
            30,
            32 * 32 * 32);

        deviceBuffer1.save(hostBuffer1.data());

        for (int i = 0; i < 30; ++i) {
            TS_ASSERT_EQUALS(
                Coord<3>(i + 7777, i + 8888, i + 9999),
                hostBuffer1[i].dimensions);
        }

        TS_ASSERT_EQUALS(sizeof(Coord<3>), filter->sizeOf());
        TS_ASSERT_EQUALS(1, selector.arity());
    }

};

}
