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

class MicroTestCell
{
public:
    class API :
        public APITraits::HasSoA
    {};

    MicroTestCell() :
        a(123),
        c(4)
    {}

    double a;
    CoordBox<2> b;
    char c;
};

LIBFLATARRAY_REGISTER_SOA(
    MicroTestCell,
    ((double)(a))
    ((CoordBox<2>)(b))
    ((char)(c)))

namespace LibGeoDecomp {

class MemberFilterCudaTest : public CxxTest::TestSuite
{
public:
    void testBasics()
    {
        typedef SharedPtr<FilterBase<TestCell<2> > >::Type FilterPtr;
        FilterPtr filter(new MemberFilter<TestCell<2>, Coord<2> >(&Coord<2>::c));

        Selector<TestCell<2> > selector(
            &TestCell<2>::pos,
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
        typedef SharedPtr<FilterBase<MicroTestCell > >::Type FilterPtr;
        FilterPtr filter(new MemberFilter<MicroTestCell, CoordBox<2> >(&CoordBox<2>::dimensions));

        Selector<MicroTestCell > selector(
            &MicroTestCell::b,
            "dimensions",
            filter);

        std::vector<MicroTestCell > vec;
        for (int i = 0; i < 44; ++i) {
            MicroTestCell cell;
            cell.b = CoordBox<2>(Coord<2>(i + 100, i + 200), Coord<2>(i + 300, i + 400));
            vec << cell;
        }

        LibFlatArray::cuda_array<MicroTestCell > deviceBuf1(&vec[0], vec.size());
        LibFlatArray::cuda_array<Coord<2> > deviceBuf2(vec.size());

        std::vector<Coord<2> > extract(vec.size());
        selector.copyMemberOut(
            deviceBuf1.data(),
            MemoryLocation::CUDA_DEVICE,
            reinterpret_cast<char*>(deviceBuf2.data()),
            MemoryLocation::CUDA_DEVICE,
            vec.size());

        deviceBuf2.save(&extract[0]);

        for (std::size_t i = 0; i < vec.size(); ++i) {
            TS_ASSERT_EQUALS(Coord<2>(i + 300, i + 400), extract[i]);
        }

        for (std::size_t i = 0; i < vec.size(); ++i) {
            extract[i] = Coord<2>(i + 500, i + 600);
        }

        deviceBuf2.load(&extract[0]);

        selector.copyMemberIn(
            reinterpret_cast<char*>(deviceBuf2.data()),
            MemoryLocation::CUDA_DEVICE,
            deviceBuf1.data(),
            MemoryLocation::CUDA_DEVICE,
            vec.size());

        deviceBuf1.save(vec.data());

        for (std::size_t i = 0; i < vec.size(); ++i) {
            TS_ASSERT_EQUALS(Coord<2>(i + 500, i + 600), vec[i].b.dimensions);
        }

        TS_ASSERT_EQUALS(sizeof(Coord<2>), filter->sizeOf());
        TS_ASSERT_EQUALS(1, selector.arity());
    }

    void testHostSoA()
    {
        LibFlatArray::cuda_array<CoordBox<2> > deviceBuffer1(30);
        LibFlatArray::cuda_array<Coord<2> > deviceBuffer2(30);
        std::vector<CoordBox<2> > hostBuffer1;
        std::vector<Coord<2> > hostBuffer2;

        typedef SharedPtr<FilterBase<MicroTestCell> >::Type FilterPtr;
        FilterPtr filter(new MemberFilter<MicroTestCell, CoordBox<2> >(&CoordBox<2>::dimensions));

        Selector<MicroTestCell> selector(
            &MicroTestCell::b,
            "dimensions",
            filter);

        for (int i = 0; i < 30; ++i) {
            hostBuffer1 << CoordBox<2>(
                Coord<2>(i + 1000, i + 2000),
                Coord<2>(i + 4000, i + 5000));
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
            TS_ASSERT_EQUALS(Coord<2>(i + 4000, i + 5000), hostBuffer2[i]);
        }

        for (int i = 0; i < 30; ++i) {
            hostBuffer2[i] = Coord<2>(i + 7777, i + 8888);
        }
        deviceBuffer2.load(hostBuffer2.data());

        // flush buffer to ensure proper data is being written back below:
        for (int i = 0; i < 30; ++i) {
            hostBuffer2[i] = Coord<2>(-1, -1);
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
                Coord<2>(i + 7777, i + 8888),
                hostBuffer1[i].dimensions);
        }

        TS_ASSERT_EQUALS(sizeof(Coord<2>), filter->sizeOf());
        TS_ASSERT_EQUALS(1, selector.arity());
    }

};

}
