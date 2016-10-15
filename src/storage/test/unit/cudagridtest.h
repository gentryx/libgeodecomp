#include <cxxtest/TestSuite.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/storage/cudagrid.h>
#include <libgeodecomp/storage/displacedgrid.h>
#include <libgeodecomp/storage/serializationbuffer.h>

#include <cuda.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class SimpleCUDATestCell
{
public:
    SimpleCUDATestCell(double val = -1, int counter = -2) :
        val(val),
        counter(counter)
    {}

    bool operator==(const SimpleCUDATestCell& other) const
    {
        return (val == other.val) && (counter == other.counter);
    }

    double val;
    int counter;
};

class CUDAGridTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
#ifdef LIBGEODECOMP_WITH_CUDA

        Coord<2> dim(30, 10);
        CoordBox<2> box(Coord<2>(), dim);

        DisplacedGrid<int> source(box, -1);
        DisplacedGrid<int> target(box, -2);
        CUDAGrid<int> buffer(box);

        Region<2> region;
        for (int y = 0; y < 10; ++y) {
            region << Streak<2>(Coord<2>(10 + y, y), 25 + y / 2);
        }

        int counter = 0;
        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            source[*i] = counter++;
        }

        buffer.loadRegion(source,  region);
        buffer.saveRegion(&target, region);
        buffer.setEdge(-4711);

        counter = 0;
        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            int expected = -2;
            if (region.count(*i)) {
                expected = counter++;
            }

            TS_ASSERT_EQUALS(target[*i], expected);
        }

        TS_ASSERT_EQUALS(buffer.getEdge(), -4711);
        buffer.setEdge(123);
        TS_ASSERT_EQUALS(buffer.getEdge(), 123);
#endif
    }

    void testRegionConstructor()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        Coord<2> dim(40, 20);
        CoordBox<2> box(Coord<2>(), dim);
        Region<2> boundingRegion;
        boundingRegion << box;

        DisplacedGrid<int> source(box, -1);
        DisplacedGrid<int> target(box, -2);
        CUDAGrid<int> buffer(boundingRegion);
        TS_ASSERT_EQUALS(buffer.boundingBox(), box);

        Region<2> region;
        for (int y = 0; y < 20; ++y) {
            region << Streak<2>(Coord<2>(10 + y, y), 25 + y / 2);
        }

        int counter = 0;
        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            source[*i] = counter++;
        }

        buffer.loadRegion(source,  region);
        buffer.saveRegion(&target, region);
        buffer.setEdge(-4711);

        counter = 0;
        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            int expected = -2;
            if (region.count(*i)) {
                expected = counter++;
            }

            TS_ASSERT_EQUALS(target[*i], expected);
        }

        TS_ASSERT_EQUALS(buffer.getEdge(), -4711);
        buffer.setEdge(123);
        TS_ASSERT_EQUALS(buffer.getEdge(), 123);
#endif
    }

    void test3d()
    {
#ifdef LIBGEODECOMP_WITH_CUDA

        typedef TestInitializer<TestCell<3> > TestCellInitializer;
        typedef TestCellInitializer::Topology Topology;

        Coord<3> dim1(35, 20, 20);
        Coord<3> dim2(75, 40, 40);
        CoordBox<3> box1(Coord<3>(), dim1);
        CoordBox<3> box2(Coord<3>(-10, -10, -10), dim2);
        TestCellInitializer init;

        DisplacedGrid<TestCell<3>, Topology> source(box1);
        DisplacedGrid<TestCell<3>, Topology> target(box2);
        init.grid(&source);
        CUDAGrid<TestCell<3>, Topology> buffer(box1);

        Region<3> region;
        for (int y = 0; y < 10; ++y) {
            region << Streak<3>(Coord<3>(y, y + 10, 2 * y), 25 + y / 2);
        }

        for (CoordBox<3>::Iterator i = box2.begin(); i != box2.end(); ++i) {
            TS_ASSERT_DIFFERS(source[*i], target[*i]);
        }

        buffer.loadRegion(source,  region);
        buffer.saveRegion(&target, region);

        for (CoordBox<3>::Iterator i = box2.begin(); i != box2.end(); ++i) {
            TestCell<3> expected ;
            if (region.count(*i)) {
                expected = source[*i];
            }

            TS_ASSERT_EQUALS(expected, target[*i]);
        }

#endif
    }

    void testTopologicalCorrectness()
    {
#ifdef LIBGEODECOMP_WITH_CUDA

        // simulation space: (0,0), (100, 100),
        //
        // coordinates covered by topologically correct grids are
        // marked with "x" below. North-west corner marked with "A",
        // south-west as "B", north-east as "C", and south-east as
        // "D":
        //
        //     0123456789 ...               90 ...  99
        //  0: xxxxxxxxxxxxxxxxxxxx   ...   xxxxxxxxxx
        //  1: xxxxxxxxxxxxxxxxxxxx   ...   xxxxxxxxxx
        //  2: xxxxxxxxxxxxxxxxxxxx   ...   xxxxxxxxxx
        //  3: xxxxxxxxxxxxxxxxxxxx   ...   xxxxxxxxxx
        //  4: xxxxxxxxxxxxxxxxxxxx   ...   xxxxxxxxxx
        //  5: xxxxxxxxxxxxxxxxxxxx   ...   xxxxxxxxxx
        //  6: xxxxxxxxxxxxxxxxxxxD   ...   Bxxxxxxxxx
        //  7:
        //  8:
        //
        // ...
        //
        // 95:
        // 96:
        // 97: xxxxxxxxxxxxxxxxxxxC   ...   Axxxxxxxxx
        // 98: xxxxxxxxxxxxxxxxxxxx   ...   xxxxxxxxxx
        // 99: xxxxxxxxxxxxxxxxxxxx   ...   xxxxxxxxxx


        Coord<2> dim(30, 10);
        Coord<2> offset(90, 97);
        Coord<2> topologicalDim(100, 100);
        CoordBox<2> box(offset, dim);

        CoordBox<2> globalBox(Coord<2>(), topologicalDim);

        DisplacedGrid<double> source(globalBox, -1);
        DisplacedGrid<double, Topologies::Torus<2>::Topology, true> target(box, -2, 0, topologicalDim);
        CUDAGrid<double, Topologies::Torus<2>::Topology, true> buffer(box, -10, -20, topologicalDim);

        Region<2> region;
        // cover all 4 sectors sketched out above:
        region << Streak<2>(Coord<2>( 0,  0),  20);
        region << Streak<2>(Coord<2>(90,  6), 100);
        region << Streak<2>(Coord<2>( 5, 97),  10);
        region << Streak<2>(Coord<2>(95, 99),  97);

        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            Coord<2> c = Topologies::Torus<2>::Topology::normalize(*i, topologicalDim);

            source[*i] = 1000.0 + c.x() * 100.0 + c.y() / 100.0;
        }

        buffer.loadRegion(source,  region);
        buffer.saveRegion(&target, region);

        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            Coord<2> c = Topologies::Torus<2>::Topology::normalize(*i, topologicalDim);

            double expected = -2;
            if (region.count(c)) {
                expected =  1000.0 + c.x() * 100.0 + c.y() / 100.0;
            }

            TS_ASSERT_EQUALS(target[c], expected);
        }

#endif
    }

    void testConstructor()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        Coord<2> dim(30, 10);
        CoordBox<2> box(Coord<2>(), dim);
        Region<2> region;
        region << box;

        DisplacedGrid<int> hostGrid1(box);
        DisplacedGrid<int> hostGrid2(box, -1);
        DisplacedGrid<int> hostGrid3(box, -2);
        int counter = 0;
        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            hostGrid1[*i] = ++counter;

            TS_ASSERT_DIFFERS(hostGrid1[*i], hostGrid2[*i]);
            TS_ASSERT_DIFFERS(hostGrid1[*i], hostGrid3[*i]);
            TS_ASSERT_DIFFERS(hostGrid2[*i], hostGrid3[*i]);
        }

        CUDAGrid<int> deviceGrid1(box);
        deviceGrid1.loadRegion(hostGrid1, region);
        CUDAGrid<int> deviceGrid2(deviceGrid1);
        CUDAGrid<int> deviceGrid3 = deviceGrid1;

        TS_ASSERT_DIFFERS(deviceGrid1.data(), deviceGrid2.data());
        TS_ASSERT_DIFFERS(deviceGrid1.data(), deviceGrid3.data());

        deviceGrid2.saveRegion(&hostGrid2, region);
        deviceGrid3.saveRegion(&hostGrid3, region);

        counter = 0;
        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            ++counter;

            TS_ASSERT_EQUALS(counter, hostGrid1[*i]);
            TS_ASSERT_EQUALS(counter, hostGrid2[*i]);
            TS_ASSERT_EQUALS(counter, hostGrid3[*i]);
        }
#endif
    }

    void testGridBaseCompatibilityAndBoundingBox()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        Coord<3> dim(30, 10, 5);
        CoordBox<3> box(Coord<3>(), dim);
        CUDAGrid<int, Topologies::Torus<3>::Topology> deviceGrid(box);

        GridBase<int, 3>& gridReference = deviceGrid;
        TS_ASSERT_EQUALS(gridReference.boundingBox(), deviceGrid.boundingBox());

        Region<3> boundingRegion;
        boundingRegion << box;
        TS_ASSERT_EQUALS(boundingRegion, deviceGrid.boundingRegion());
#endif
    }

    void testGetSetOfSingleCells2D()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        Coord<2> dim(40, 50);
        Coord<2> origin(30, 20);
        CoordBox<2> box(origin, dim);

        CUDAGrid<SimpleCUDATestCell, Topologies::Cube<2>::Topology, true> grid(
            box,
            SimpleCUDATestCell(),
            SimpleCUDATestCell(),
            box.dimensions);

        TS_ASSERT_EQUALS(SimpleCUDATestCell(-1, -2), grid.get(Coord<2>(-3, -2)));
        TS_ASSERT_EQUALS(SimpleCUDATestCell(-1, -2), grid.getEdge());

        grid.set(Coord<2>(30, 20), SimpleCUDATestCell(1.2, 3));
        grid.set(Coord<2>(31, 20), SimpleCUDATestCell(4.5, 6));
        grid.set(Coord<2>(69, 69), SimpleCUDATestCell(7.8, 9));
        grid.set(Coord<2>(20, 20), SimpleCUDATestCell(-4, -8));

        TS_ASSERT_EQUALS(SimpleCUDATestCell(1.2, 3), grid.get(Coord<2>(30, 20)));
        TS_ASSERT_EQUALS(SimpleCUDATestCell(4.5, 6), grid.get(Coord<2>(31, 20)));
        TS_ASSERT_EQUALS(SimpleCUDATestCell(7.8, 9), grid.get(Coord<2>(69, 69)));
        // this ensures topology is honored and the edge cell has been
        // set in the out-of-bounds access above:
        TS_ASSERT_EQUALS(SimpleCUDATestCell(-4, -8), grid.getEdge());
#endif
    }

    void testGetSetOfSingleCells3D()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        Coord<3> dim(30, 10, 5);
        Coord<3> origin(-3, -2, -1);
        CoordBox<3> box(origin, dim);

        CUDAGrid<SimpleCUDATestCell, Topologies::Torus<3>::Topology, true> grid(
            box,
            SimpleCUDATestCell(),
            SimpleCUDATestCell(),
            box.dimensions);

        grid.set(Coord<3>(-3, -2, -1), SimpleCUDATestCell(1.2, 3));
        grid.set(Coord<3>(-2, -2, -1), SimpleCUDATestCell(4.5, 6));
        grid.set(Coord<3>(26,  7,  3), SimpleCUDATestCell(7.8, 9));
        grid.set(Coord<3>(-4, -2, -1), SimpleCUDATestCell(6.6, 6));

        TS_ASSERT_EQUALS(SimpleCUDATestCell(1.2, 3), grid.get(Coord<3>(-3, -2, -1)));
        TS_ASSERT_EQUALS(SimpleCUDATestCell(4.5, 6), grid.get(Coord<3>(-2, -2, -1)));
        TS_ASSERT_EQUALS(SimpleCUDATestCell(7.8, 9), grid.get(Coord<3>(26,  7,  3)));
        // important for ensuring the torus topology was honored in
        // the out-of-bounds access above:
        TS_ASSERT_EQUALS(SimpleCUDATestCell(6.6, 6), grid.get(Coord<3>(26, -2, -1)));
#endif
    }

    void testGetSetOfMultipleCells2D()
    {
#ifdef LIBGEODECOMP_WITH_CUDA
        Coord<2> dim(40, 50);
        Coord<2> origin(30, 20);
        CoordBox<2> box(origin, dim);

        CUDAGrid<SimpleCUDATestCell, Topologies::Cube<2>::Topology> grid(box);

        std::vector<SimpleCUDATestCell> source;
        std::vector<SimpleCUDATestCell> target(6);
        source << SimpleCUDATestCell(1.1, 1)
               << SimpleCUDATestCell(2.2, 2)
               << SimpleCUDATestCell(3.3, 3)
               << SimpleCUDATestCell(4.4, 4)
               << SimpleCUDATestCell(5.5, 5)
               << SimpleCUDATestCell(6.6, 6);
        Streak<2> streak(Coord<2>(35, 27), 41);

        grid.set(streak, &source[0]);
        grid.get(streak, &target[0]);

        TS_ASSERT_EQUALS(source, target);
#endif
    }

    void testSelectorLoadSaveMember()
    {
        Coord<2> origin(30, 20);
        Coord<2> dim(40, 50);
        CoordBox<2> box(origin, dim);

        CUDAGrid<TestCell<2> > grid(box);
        Selector<TestCell<2> > selector = MAKE_SELECTOR(TestCell<2>, testValue);

        std::vector<double> buffer;
        for (int i = 0; i < 90; ++i) {
            buffer << i;
        }

        Region<2> region;
        for (int i = 0; i < 30; ++i) {
            region << Coord<2>(30 + i, 30);
        }

        for (int i = 0; i < 30; ++i) {
            region << Coord<2>(35 + i, 40);
        }

        for (int i = 0; i < 30; ++i) {
            region << Coord<2>(40 + i, 50);
        }

        grid.loadMember(&buffer[0], MemoryLocation::HOST, selector, region);

        int counter = 0;
        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            TS_ASSERT_EQUALS(counter, grid.get(*i).testValue);
            ++counter;
        }
    }

    void testLoadSaveRegionFromToBuffer()
    {
        typedef SerializationBuffer<TestCell<3> > SerializationBuf;

        Coord<3> origin(30, 20, 10);
        Coord<3> dim(40, 50, 60);
        CoordBox<3> box(origin, dim);

        Region<3> region;
        region << Streak<3>(Coord<3>(30, 20, 10), 60)
               << Streak<3>(Coord<3>(40, 30, 20), 55);

        CUDAGrid<TestCell<3>, Topologies::Cube<3>::Topology> realGrid(box);
        GridBase<TestCell<3>, 3>& grid = realGrid;

        int counter = 0;
        std::vector<TestCell<3> > lineBuf(dim.x());

        for (CoordBox<3>::StreakIterator i = box.beginStreak(); i != box.endStreak(); ++i) {
            for (int x = 0; x < dim.x(); ++x) {
                lineBuf[x].testValue = counter++;
            }
            grid.set(*i, lineBuf.data());
        }

        SerializationBuf::BufferType buffer = SerializationBuf::create(region);

        // check saveRegion()
        grid.saveRegion(&buffer, region);

        int offset = (40 - 30) + (30 - 20) * 40 + (20 - 10) * 40 * 50;

        for (int i = 0; i < 30; ++i) {
            TS_ASSERT_EQUALS(buffer[i].testValue, double(i));
        }
        for (int i = 30; i < 45; ++i) {
            TS_ASSERT_EQUALS(buffer[i].testValue, double(i - 30 + offset));
        }

        // check loadRegion()
        for (std::size_t i = 0; i < region.size(); ++i) {
            buffer[i].testValue = 4711 + i;
        }

        grid.loadRegion(buffer, region);
        counter = 4711;

        for (Region<3>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            lineBuf.resize(i->length());
            grid.get(*i, lineBuf.data());

            for (int x = 0; x < i->length(); ++x) {
                TS_ASSERT_EQUALS(lineBuf[x].testValue, double(counter));
                ++counter;
            }
        }
    }

    void testLoadSaveRegionFromToBufferWithOffset()
    {
        typedef SerializationBuffer<TestCell<3> > SerializationBuf;

        Coord<3> origin(30, 20, 10);
        Coord<3> dim(40, 50, 60);
        CoordBox<3> box(origin, dim);

        Region<3> region;
        region << Streak<3>(Coord<3>(30, 20, 10), 60)
               << Streak<3>(Coord<3>(40, 30, 20), 55);

        CUDAGrid<TestCell<3>, Topologies::Cube<3>::Topology> realGrid(box);
        GridBase<TestCell<3>, 3>& grid = realGrid;

        int counter = 0;
        std::vector<TestCell<3> > lineBuf(dim.x());

        for (CoordBox<3>::StreakIterator i = box.beginStreak(); i != box.endStreak(); ++i) {
            for (int x = 0; x < dim.x(); ++x) {
                lineBuf[x].testValue = counter++;
            }
            grid.set(*i, lineBuf.data());
        }

        SerializationBuf::BufferType buffer = SerializationBuf::create(region);

        // check saveRegion()
        grid.saveRegion(&buffer, region, Coord<3>(4, 5, 6));

        int offset = (34 - 30) + (25 - 20) * 40 + (16 - 10) * 40 * 50;
        for (int i = 0; i < 30; ++i) {
            TS_ASSERT_EQUALS(buffer[i].testValue, double(i + offset));
        }

        offset = (44 - 30) + (35 - 20) * 40 + (26 - 10) * 40 * 50;
        for (int i = 30; i < 45; ++i) {
            TS_ASSERT_EQUALS(buffer[i].testValue, double(i - 30 + offset));
        }

        // check loadRegion()
        for (std::size_t i = 0; i < region.size(); ++i) {
            buffer[i].testValue = 4711 + i;
        }

        grid.loadRegion(buffer, region, Coord<3>(4, 5, 6));
        counter = 4711;

        for (Region<3>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            lineBuf.resize(i->length());
            Streak<3> streak = *i;
            streak.origin += Coord<3>(4, 5, 6);
            streak.endX += 4;
            grid.get(streak, lineBuf.data());

            for (int x = 0; x < i->length(); ++x) {
                TS_ASSERT_EQUALS(lineBuf[x].testValue, double(counter));
                ++counter;
            }
        }
    }

    void testResize()
    {
        Coord<3> origin(20, 21, 22);
        Coord<3> dim(30, 20, 10);
        CoordBox<3> box(origin, dim);

        TestCell<3> innerCell;
        TestCell<3> edgeCell;
        innerCell.isEdgeCell = false;
        edgeCell.isEdgeCell = true;

        CUDAGrid<TestCell<3>, Topologies::Cube<3>::Topology> grid(box, innerCell, edgeCell);
        TS_ASSERT_EQUALS(edgeCell, grid.getEdge());

        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            TS_ASSERT_EQUALS(innerCell, grid.get(*i));
        }

        origin = Coord<3>(30, 31, 32);
        dim = Coord<3>(40, 50, 60);
        box = CoordBox<3>(origin, dim);
        grid.resize(box);
        TS_ASSERT_EQUALS(box, grid.boundingBox());
        TS_ASSERT_EQUALS(edgeCell, grid.getEdge());

        int counter = 0;
        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            TestCell<3> cell;
            cell.testValue = ++counter;
            cell.isEdgeCell = false;
            cell.isValid = true;

            grid.set(*i, cell);
        }

        counter = 0;
        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            TestCell<3> cell;
            cell.testValue = ++counter;
            cell.isEdgeCell = false;
            cell.isValid = true;

            TS_ASSERT_EQUALS(cell, grid.get(*i));
        }
    }

};

}
