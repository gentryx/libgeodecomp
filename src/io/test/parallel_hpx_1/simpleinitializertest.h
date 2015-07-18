#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <libgeodecomp/communication/hpxserialization.h>
#include <libgeodecomp/io/simpleinitializer.h>

using namespace LibGeoDecomp;

namespace HPXSerializationTest {

template<typename T>
class BasicDummy
{
public:
    BasicDummy(Coord<2> c = Coord<2>())
    {}

    virtual ~BasicDummy()
    {}

    virtual Coord<2> gridDimensions() = 0;
    virtual int maxSteps() = 0;
    virtual std::string getMessage() = 0;
};

template<typename T>
class Dummy : public BasicDummy<T>
{
public:
    Dummy(Coord<2> c, int s) :
        BasicDummy<T>(c),
        c(c),
        s(s)
    {}

    Coord<2> gridDimensions()
    {
        return c;
    }

    int maxSteps()
    {
        return s;
    }

    Coord<2> c;
    int s;
};

class Concrete : public Dummy<int>
{
public:
    Concrete(Coord<2> dim = Coord<2>(), int steps = 0, std::string message = "") :
        Dummy<int>(dim, steps),
        message(message)
    {}

    void grid(GridBase<int, 2> *target)
    {}

    std::string getMessage()
    {
        return message;
    }


    std::string message;
};

template <typename Archive, typename T>
void serialize(Archive & archive, BasicDummy<T>& dummy, unsigned)
{
}

template <typename Archive, typename T>
void serialize(Archive & archive, Dummy<T>& dummy, unsigned)
{
    archive
        & hpx::serialization::base_object<BasicDummy<int> >(dummy)
        & dummy.c
        & dummy.s;
}

template<class ARCHIVE>
void serialize(ARCHIVE& archive, Concrete& init, const unsigned version)
{
    archive
        & hpx::serialization::base_object<Dummy<int> >(init)
        & init.message;
}

}

HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC_TEMPLATE((template <class T>), HPXSerializationTest::BasicDummy<T>)
HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC_TEMPLATE((template <class T>), HPXSerializationTest::Dummy<T>)
HPX_SERIALIZATION_REGISTER_CLASS(HPXSerializationTest::Concrete);

namespace HPXSerializationTest {

class TestInitializer : public SimpleInitializer<int>
{
public:
    TestInitializer(
        int defaultValue = 0,
        const Coord<2>& dim = Coord<2>(),
        unsigned lastStep = 0) :
        SimpleInitializer(dim, lastStep),
        defaultValue(defaultValue)
    {}

    void grid(GridBase<int, 2> *grid)
    {
        grid->set(Coord<2>(), defaultValue);
    }

    int defaultValue;
};

template<class ARCHIVE>
void serialize(ARCHIVE& archive, TestInitializer& init, const unsigned version)
{
    archive
        & hpx::serialization::base_object<SimpleInitializer<int> >(init)
        & init.defaultValue;
}

}

HPX_SERIALIZATION_REGISTER_CLASS(HPXSerializationTest::TestInitializer);

namespace LibGeoDecomp {

class SimpleInitializerTest : public CxxTest::TestSuite
{
public:
    void testSerializationOfTestClassesByReference()
    {
        HPXSerializationTest::Concrete init1(Coord<2>(123, 456), 789, "Dodge");
        HPXSerializationTest::Concrete init2(Coord<2>( -1,  -1),   1, "Fuski");

        TS_ASSERT_EQUALS(init2.gridDimensions(), Coord<2>(-1, -1));
        TS_ASSERT_EQUALS(init2.maxSteps(),       1);
        TS_ASSERT_EQUALS(init2.message,          "Fuski");

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << init1;

        hpx::serialization::input_archive inputArchive(buffer);
        inputArchive >> init2;

        TS_ASSERT_EQUALS(init2.gridDimensions(), Coord<2>(123, 456));
        TS_ASSERT_EQUALS(init2.maxSteps(),       789);
        TS_ASSERT_EQUALS(init2.message,          "Dodge");
    }

    void testSerializationOfTestClassesViaPolymorphicSharedPointer()
    {
        boost::shared_ptr<HPXSerializationTest::BasicDummy<int> > init1(
            new HPXSerializationTest::Concrete(Coord<2>(123, 456), 789, "Dodge"));
        boost::shared_ptr<HPXSerializationTest::BasicDummy<int> > init2;

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << init1;

        hpx::serialization::input_archive inputArchive(buffer);
        inputArchive >> init2;

        TS_ASSERT_EQUALS(init2->gridDimensions(), Coord<2>(123, 456));
        TS_ASSERT_EQUALS(init2->maxSteps(),       789);
        TS_ASSERT_EQUALS(init2->getMessage(),     "Dodge");
    }

    void testSerializationOfInitializerByReference()
    {
        HPXSerializationTest::TestInitializer init1(888, Coord<2>(666, 777), 999);
        HPXSerializationTest::TestInitializer init2(333, Coord<2>(111, 222), 444);
        Grid<int> grid(Coord<2>(1, 1), -1);

        init2.grid(&grid);
        TS_ASSERT_EQUALS(init2.gridDimensions(), Coord<2>(111, 222));
        TS_ASSERT_EQUALS(init2.startStep(),      0);
        TS_ASSERT_EQUALS(init2.maxSteps(),       444);
        TS_ASSERT_EQUALS(grid.get(Coord<2>()),   333);

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << init1;

        hpx::serialization::input_archive inputArchive(buffer);
        inputArchive >> init2;

        init2.grid(&grid);
        TS_ASSERT_EQUALS(init2.gridDimensions(), Coord<2>(666, 777));
        TS_ASSERT_EQUALS(init2.startStep(),      0);
        TS_ASSERT_EQUALS(init2.maxSteps(),       999);
        TS_ASSERT_EQUALS(grid.get(Coord<2>()),   888);
    }
};

}
