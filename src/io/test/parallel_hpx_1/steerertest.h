#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <libgeodecomp/io/steerer.h>
#include <libgeodecomp/communication/hpxserialization.h>

using namespace LibGeoDecomp;

namespace HPXSerializationTest {
class FancySteerer : public Steerer<double>
{
public:
    explicit
    FancySteerer(double addent = 0) :
        Steerer<double>(1),
        addent(addent)
    {}

    virtual void nextStep(
        GridType *grid,
        const Region<2>& validRegion,
        const CoordType& globalDimensions,
        unsigned step,
        SteererEvent event,
        std::size_t rank,
        bool lastCall,
        Steerer<double>::SteererFeedback *feedback)
    {
        for (Region<2>::Iterator i = validRegion.begin(); i != validRegion.end(); ++i) {
            grid->set(*i, grid->get(*i) + addent);
        }
    }

    double addent;
};

}

namespace hpx { namespace serialization {
template <typename Archive>
void serialize(Archive& archive, HPXSerializationTest::FancySteerer& steerer, unsigned)
{
    archive & hpx::serialization::base_object<Steerer<double> >(steerer);
    archive & steerer.addent;
}

} }

HPX_SERIALIZATION_REGISTER_CLASS(HPXSerializationTest::FancySteerer);
HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC(HPXSerializationTest::FancySteerer);

namespace LibGeoDecomp {

class SteererTest : public CxxTest::TestSuite
{
public:
    void testSerializationOfInitializerByReference()
    {
        Coord<2> dim(20, 10);
        Region<2> region;
        region << CoordBox<2>(Coord<2>(), dim);
        Grid<double> grid(dim, -1);
        std::vector<char> buffer;

        {
            HPXSerializationTest::FancySteerer steerer1(4);
            HPXSerializationTest::FancySteerer steerer2(5);

            hpx::serialization::output_archive outputArchive(buffer);
            outputArchive << steerer1;
            outputArchive << steerer2;

            for (int y = 0; y < dim.y(); ++y) {
                for (int x = 0; x < dim.x(); ++x) {
                    TS_ASSERT_EQUALS(grid[Coord<2>(x, y)], -1);
                }
            }
        }

        {
            HPXSerializationTest::FancySteerer steerer(-1);
            hpx::serialization::input_archive inputArchive(buffer);

            inputArchive >> steerer;
            steerer.nextStep(&grid, region, dim, 0, SteererEvent::STEERER_NEXT_STEP, 0, true, 0);

            for (int y = 0; y < dim.y(); ++y) {
                for (int x = 0; x < dim.x(); ++x) {
                    TS_ASSERT_EQUALS(grid[Coord<2>(x, y)], 3);
                }
            }

            inputArchive >> steerer;
            steerer.nextStep(&grid, region, dim, 0, SteererEvent::STEERER_NEXT_STEP, 0, true, 0);

            for (int y = 0; y < dim.y(); ++y) {
                for (int x = 0; x < dim.x(); ++x) {
                    TS_ASSERT_EQUALS(grid[Coord<2>(x, y)], 8);
                }
            }
        }
    }

    void testSerializationOfInitializerViaPointer()
    {
        Coord<2> dim(20, 10);
        Region<2> region;
        region << CoordBox<2>(Coord<2>(), dim);
        Grid<double> grid(dim, -1);
        std::vector<char> buffer;

        {
            boost::shared_ptr<Steerer<double> > steerer1(new HPXSerializationTest::FancySteerer(4));
            boost::shared_ptr<Steerer<double> > steerer2(new HPXSerializationTest::FancySteerer(5));

            hpx::serialization::output_archive outputArchive(buffer);
            outputArchive << steerer1;
            outputArchive << steerer2;

            for (int y = 0; y < dim.y(); ++y) {
                for (int x = 0; x < dim.x(); ++x) {
                    TS_ASSERT_EQUALS(grid[Coord<2>(x, y)], -1);
                }
            }
        }

        {
            boost::shared_ptr<Steerer<double> >  steerer;
            hpx::serialization::input_archive inputArchive(buffer);

            inputArchive >> steerer;
            steerer->nextStep(&grid, region, dim, 0, SteererEvent::STEERER_NEXT_STEP, 0, true, 0);

            for (int y = 0; y < dim.y(); ++y) {
                for (int x = 0; x < dim.x(); ++x) {
                    TS_ASSERT_EQUALS(grid[Coord<2>(x, y)], 3);
                }
            }

            inputArchive >> steerer;
            steerer->nextStep(&grid, region, dim, 0, SteererEvent::STEERER_NEXT_STEP, 0, true, 0);

            for (int y = 0; y < dim.y(); ++y) {
                for (int x = 0; x < dim.x(); ++x) {
                    TS_ASSERT_EQUALS(grid[Coord<2>(x, y)], 8);
                }
            }
        }
    }

};

}
