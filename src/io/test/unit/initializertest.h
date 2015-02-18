#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/io/simpleinitializer.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class DummyCell
{
public:
    class API :
        public APITraits::HasTorusTopology<3>
    {};
};

class DummyInitializer : public SimpleInitializer<DummyCell>
{
public:
    DummyInitializer() :
        SimpleInitializer<DummyCell>(Coord<3>(200, 100, 50), 1000)
    {}

    void grid(GridBase<DummyCell, 3> *grid)
    {}
};

class InitializerTest : public CxxTest::TestSuite
{
public:

    std::vector<std::string> files;

    void testSeedRNG()
    {
        DummyInitializer init;

        std::vector<double> actual;
        std::vector<double> expected;

        init.seedRNG(Coord<3>(199, 0, 0));
        for (int i = 0; i < 10; ++i) {
            expected << Random::gen_d();
        }
        init.seedRNG(Coord<3>(-1, 0, 0));
        for (int i = 0; i < 10; ++i) {
            actual << Random::gen_d();
        }
        TS_ASSERT_EQUALS(actual, expected);
        actual.clear();
        expected.clear();

        init.seedRNG(Coord<3>(50, 99, 25));
        for (int i = 0; i < 10; ++i) {
            expected << Random::gen_d();
        }
        init.seedRNG(Coord<3>(50, -1, 25));
        for (int i = 0; i < 10; ++i) {
            actual << Random::gen_d();
        }
        TS_ASSERT_EQUALS(actual, expected);
        actual.clear();
        expected.clear();

        init.seedRNG(Coord<3>(160, 99, 49));
        for (int i = 0; i < 10; ++i) {
            expected << Random::gen_d();
        }
        init.seedRNG(Coord<3>(160, 99, -1));
        for (int i = 0; i < 10; ++i) {
            actual << Random::gen_d();
        }
        TS_ASSERT_EQUALS(actual, expected);
        actual.clear();
        expected.clear();

        init.seedRNG(Coord<3>(160, 99, 49));
        for (int i = 0; i < 10; ++i) {
            expected << Random::gen_d();
        }
        init.seedRNG(Coord<3>(161, 99, -1));
        for (int i = 0; i < 10; ++i) {
            actual << Random::gen_d();
        }
        TS_ASSERT_DIFFERS(actual, expected);
        actual.clear();
        expected.clear();

    }
};

}
