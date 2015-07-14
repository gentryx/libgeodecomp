#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <libgeodecomp/communication/serialization.h>
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
void serialize(Archive & ar, BasicDummy<T> & d1, unsigned)
{
}

template <typename Archive, typename T>
void serialize(Archive & ar, Dummy<T> & d1, unsigned)
{
    ar
        & hpx::serialization::base_object<BasicDummy<int> >(d1)
        & d1.c
        & d1.s;
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
};

}
