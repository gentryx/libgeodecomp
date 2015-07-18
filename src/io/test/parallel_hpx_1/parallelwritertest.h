#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <libgeodecomp/communication/hpxserialization.h>
#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/io/simpleinitializer.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {
class ParallelWriterTest;
}

namespace HPXSerializationTest {

class FancyTestWriter : public ParallelWriter<double>
{
public:
    template <typename Archive>
    friend void serialize(Archive & archive, FancyTestWriter& writer, unsigned);

    friend class LibGeoDecomp::ParallelWriterTest;

    using ParallelWriter<double>::GridType;
    using ParallelWriter<double>::RegionType;
    using ParallelWriter<double>::CoordType;

    FancyTestWriter(std::string prefix = "", unsigned period = 1, double d = 0) :
        ParallelWriter(prefix, period),
        cargo(d)
    {}

    virtual void grid(GridBase<double, 2> *target)
    {}

    ParallelWriter<double> *clone() const
    {
        std::cout << "DANGER!\n";
        return 0;
    }

    void stepFinished(
        const GridType& grid,
        const RegionType& validRegion,
        const CoordType& globalDimensions,
        unsigned step,
        WriterEvent event,
        std::size_t rank,
        bool lastCall)
    {
    }

private:
    double cargo;
};

template <typename Archive>
void serialize(Archive & archive, FancyTestWriter& writer, unsigned)
{
    archive & hpx::serialization::base_object<ParallelWriter<double> >(writer);
    archive & writer.cargo;
}

}

HPX_SERIALIZATION_REGISTER_CLASS(HPXSerializationTest::FancyTestWriter);

namespace LibGeoDecomp {

class ParallelWriterTest : public CxxTest::TestSuite
{
public:
    void testSerializationOfInitializerByReference()
    {
        HPXSerializationTest::FancyTestWriter writer1("foo", 10, 3.14);
        HPXSerializationTest::FancyTestWriter writer2("bar",  1, 0);

        TS_ASSERT_EQUALS(writer2.getPrefix(), "bar");
        TS_ASSERT_EQUALS(writer2.getPeriod(), 1);
        TS_ASSERT_EQUALS(writer2.cargo,       0);

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << writer1;

        hpx::serialization::input_archive inputArchive(buffer);
        inputArchive >> writer2;

        TS_ASSERT_EQUALS(writer2.getPrefix(), "foo");
        TS_ASSERT_EQUALS(writer2.getPeriod(), 10);
        TS_ASSERT_EQUALS(writer2.cargo,       3.14);
    }

    void testSerializationOfInitializerBySmartPointer()
    {
        boost::shared_ptr<LibGeoDecomp::ParallelWriter<double> > writer1(
            new HPXSerializationTest::FancyTestWriter("goo", 20, 6644));

        boost::shared_ptr<LibGeoDecomp::ParallelWriter<double> > writer2(
            new HPXSerializationTest::FancyTestWriter("car", 30, 7755));

        TS_ASSERT_EQUALS(writer2->getPrefix(), "car");
        TS_ASSERT_EQUALS(writer2->getPeriod(), 30);
        TS_ASSERT_EQUALS(dynamic_cast<HPXSerializationTest::FancyTestWriter*>(writer2.get())->cargo, 7755);

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << writer1;

        hpx::serialization::input_archive inputArchive(buffer);
        inputArchive >> writer2;

        TS_ASSERT_EQUALS(writer2->getPrefix(), "goo");
        TS_ASSERT_EQUALS(writer2->getPeriod(), 20);
        TS_ASSERT_EQUALS(dynamic_cast<HPXSerializationTest::FancyTestWriter*>(writer2.get())->cargo, 6644);
    }
};

}
