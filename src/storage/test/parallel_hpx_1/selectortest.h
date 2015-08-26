#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <libgeodecomp/storage/selector.h>
#include <libgeodecomp/communication/hpxserializationwrapper.h>

using namespace LibGeoDecomp;

namespace HPXSerializationTest {

class TestStructA
{
public:
    int x;
    double y[5];
};


}

namespace LibGeoDecomp {

class SelectorTest : public CxxTest::TestSuite
{
public:
    void testSerializationWithScalarMember()
    {
        Selector<HPXSerializationTest::TestStructA> selector1(&HPXSerializationTest::TestStructA::x, "x");
        Selector<HPXSerializationTest::TestStructA> selector2;

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << selector1;

        hpx::serialization::input_archive inputArchive(buffer);
        inputArchive >> selector2;

        std::vector<HPXSerializationTest::TestStructA> source(10);
        for (int i = 0; i < 10; ++i) {
            source[i].x = i + 4711;
        }

        std::vector<int> target(10);
        selector2.copyMemberOut(&source[0], reinterpret_cast<char*>(&target[0]), 10);

        for (int i = 0; i < 10; ++i) {
            TS_ASSERT_EQUALS(i + 4711, target[i]);
        }
    }

    void testSerializationWithScalarMemberViaPointer()
    {
        boost::shared_ptr<Selector<HPXSerializationTest::TestStructA> > selector1(
            new Selector<HPXSerializationTest::TestStructA>(&HPXSerializationTest::TestStructA::x, "x"));
        boost::shared_ptr<Selector<HPXSerializationTest::TestStructA> > selector2;

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << selector1;

        hpx::serialization::input_archive inputArchive(buffer);
        inputArchive >> selector2;

        std::vector<HPXSerializationTest::TestStructA> source(400);
        for (int i = 0; i < 400; ++i) {
            source[i].x = i + 9876;
        }

        std::vector<int> target(400);
        selector2->copyMemberOut(&source[0], reinterpret_cast<char*>(&target[0]), 400);

        for (int i = 0; i < 400; ++i) {
            TS_ASSERT_EQUALS(i + 9876, target[i]);
        }
    }

    void testSerializationWithArrayMember()
    {
        Selector<HPXSerializationTest::TestStructA> selector1(&HPXSerializationTest::TestStructA::y, "y");
        Selector<HPXSerializationTest::TestStructA> selector2;

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << selector1;

        hpx::serialization::input_archive inputArchive(buffer);
        inputArchive >> selector2;

        std::vector<HPXSerializationTest::TestStructA> source(20);
        for (int i = 0; i < 20; ++i) {
            for (int j = 0; j < 5; ++j) {
                source[i].y[j] = i * 1000 + j;
            }
        }

        std::vector<double> target(20 * 5);
        selector2.copyMemberOut(&source[0], reinterpret_cast<char*>(&target[0]), 20);

        for (int i = 0; i < 20; ++i) {
            for (int j = 0; j < 5; ++j) {
                TS_ASSERT_EQUALS((i * 1000 + j), target[i * 5 + j]);
            }
        }
    }
};

}
