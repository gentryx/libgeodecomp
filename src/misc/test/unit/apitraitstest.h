#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/stringops.h>
#include <libgeodecomp/storage/grid.h>
#include <libgeodecomp/storage/updatefunctor.h>

using namespace LibGeoDecomp;

std::ostringstream myTestEvents;

class MySimpleDummyCell
{
public:
    explicit MySimpleDummyCell(int val = 0) :
        val(val)
    {};

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, unsigned nanoStep)
    {
        myTestEvents << "MySimpleDummyCell::update(" << hood[FixedCoord<-1, 0, 0>()].val << ")\n";
    }

    int val;
};

class MyFancyDummyCell
{
public:
    class API :
        public APITraits::HasTorusTopology<3>,
        public APITraits::HasStencil<Stencils::Moore<3, 1> >,
        public APITraits::HasNanoSteps<3>,
        public APITraits::HasStaticData<double>,
        public APITraits::HasCustomRegularGrid
    {
    public:
        inline FloatCoord<3> getRegularGridSpacing()
        {
            return FloatCoord<3>(30, 20, 10);
        }

        inline FloatCoord<3> getRegularGridOrigin()
        {
            return FloatCoord<3>(5, 6, 7);
        }

        template<int UNUSED_DIM>
        inline std::vector<std::string> getRegularGridAxisUnits()
        {
            std::vector<std::string> ret;
            ret << "m"
                << "nm"
                << "km";
            return ret;
        }
    };

    static double staticData;

    explicit MyFancyDummyCell(int val = 0) :
        val(val)
    {};

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, unsigned nanoStep)
    {
        myTestEvents << "MyFancyDummyCell::update(" << hood[FixedCoord<-1, 0, 0>()].val << ")\n";
    }

    int val;
};

double MyFancyDummyCell::staticData;

namespace LibGeoDecomp {

class APITraitsTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        myTestEvents.str("");
    }

    void testBasicCallback1()
    {
        Coord<2> gridDim(10, 5);
        typedef APITraits::SelectTopology<MySimpleDummyCell>::Value Topology;
        Grid<MySimpleDummyCell, Topology> gridOld(gridDim, MySimpleDummyCell(666), MySimpleDummyCell(31));
        Grid<MySimpleDummyCell, Topology> gridNew(gridDim);

        Streak<2> streak(Coord<2>(0, 0), 5);
        Region<2> region;
        region << streak;
        UpdateFunctor<MySimpleDummyCell>()(region, Coord<2>(), Coord<2>(), gridOld, &gridNew, 0);

        std::string expected;
        for (int i = 0; i < streak.length(); ++i) {
            int num = 666;
            if (i == 0) {
                num = 31;
            }
            expected += "MySimpleDummyCell::update(" + StringOps::itoa(num) + ")\n";
        }
        TS_ASSERT_EQUALS(myTestEvents.str(), expected);
    }

    void testBasicCallback2()
    {
        Coord<3> gridDim(10, 5, 10);
        typedef APITraits::SelectTopology<MyFancyDummyCell>::Value Topology;
        Grid<MyFancyDummyCell, Topology> gridOld(gridDim, MyFancyDummyCell(666), MyFancyDummyCell(31));
        Grid<MyFancyDummyCell, Topology> gridNew(gridDim);

        Streak<3> streak(Coord<3>(0, 0, 0), 4);
        Region<3> region;
        region << streak;
        UpdateFunctor<MyFancyDummyCell>()(region, Coord<3>(), Coord<3>(), gridOld, &gridNew, 0);

        std::string expected;
        for (int i = 0; i < streak.length(); ++i) {
            expected += "MyFancyDummyCell::update(666)\n";
        }
        TS_ASSERT_EQUALS(myTestEvents.str(), expected);
    }

    void testStaticDataHandling()
    {
        typedef APITraits::SelectStaticData<MyFancyDummyCell>::Value StaticData;
        StaticData data(12.34);
        std::swap(MyFancyDummyCell::staticData, data);
        TS_ASSERT_EQUALS(12.34, MyFancyDummyCell::staticData);
    }

    class TestCell1
    {
    public:
        class API
        {
        public:
            std::string ping() {
                return "ok";
            }
        };
    };

    void testSelectAPI()
    {
        TS_ASSERT_EQUALS("ok", APITraits::SelectAPI<TestCell1>::Value().ping());
    }

    void testSelectRegularGrid()
    {
        FloatCoord<2> quadrantDimA;
        FloatCoord<3> quadrantDimB;

        FloatCoord<2> originA;
        FloatCoord<3> originB;

        std::vector<std::string> axisUnitsA;
        std::vector<std::string> axisUnitsB;

        std::vector<std::string> expectedUnitsA;
        std::vector<std::string> expectedUnitsB;

        expectedUnitsA << ""
                       << "";

        expectedUnitsB << "m"
                       << "nm"
                       << "km";

        APITraits::SelectRegularGrid<MySimpleDummyCell>::value(&quadrantDimA, &originA, &axisUnitsA);
        APITraits::SelectRegularGrid<MyFancyDummyCell >::value(&quadrantDimB, &originB, &axisUnitsB);

        TS_ASSERT_EQUALS(FloatCoord<2>(1, 1),       quadrantDimA);
        TS_ASSERT_EQUALS(FloatCoord<3>(30, 20, 10), quadrantDimB);

        TS_ASSERT_EQUALS(FloatCoord<2>(0, 0),    originA);
        TS_ASSERT_EQUALS(FloatCoord<3>(5, 6, 7), originB);

        TS_ASSERT_EQUALS(axisUnitsA.size(), std::size_t(2));
        TS_ASSERT_EQUALS(axisUnitsB.size(), std::size_t(3));

        TS_ASSERT_EQUALS(axisUnitsA, expectedUnitsA);
        TS_ASSERT_EQUALS(axisUnitsB, expectedUnitsB);

    }

    void testTrueTypeFalseTypeOperators()
    {
        typedef APITraits::TrueType TrueType;;
        typedef APITraits::FalseType FalseType;

        // test ==
        TS_ASSERT_EQUALS(TrueType(), TrueType());
        TS_ASSERT_EQUALS(FalseType(), FalseType());

        TS_ASSERT_EQUALS(true,  TrueType());
        TS_ASSERT_EQUALS(false, FalseType());

        // test !=
        TS_ASSERT(TrueType()  != FalseType());
        TS_ASSERT(FalseType() != TrueType());

        TS_ASSERT(true  != FalseType());
        TS_ASSERT(FalseType()  != true);

        TS_ASSERT(false  != TrueType());
        TS_ASSERT(TrueType()  != false);

        // test !
        TS_ASSERT_EQUALS(TrueType(),  !FalseType());
        TS_ASSERT_EQUALS(FalseType(), !TrueType());

        // test &&
        TS_ASSERT_EQUALS(TrueType(),  TrueType()  && TrueType());
        TS_ASSERT_EQUALS(FalseType(), TrueType()  && FalseType());
        TS_ASSERT_EQUALS(FalseType(), FalseType() && TrueType());
        TS_ASSERT_EQUALS(FalseType(), FalseType() && FalseType());

        TS_ASSERT_EQUALS(TrueType(),  true  && TrueType());
        TS_ASSERT_EQUALS(FalseType(), true  && FalseType());
        TS_ASSERT_EQUALS(FalseType(), false && TrueType());
        TS_ASSERT_EQUALS(FalseType(), false && FalseType());

        TS_ASSERT_EQUALS(TrueType(),  TrueType()  && true);
        TS_ASSERT_EQUALS(FalseType(), TrueType()  && false);
        TS_ASSERT_EQUALS(FalseType(), FalseType() && true);
        TS_ASSERT_EQUALS(FalseType(), FalseType() && false);

    }
};

}
