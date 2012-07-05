#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/pointerneighborhood.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class PointerNeighborhoodTest : public CxxTest::TestSuite 
{
public:
    void testMoore2D()
    {
        std::string cTW("TW");
        std::string cT("T");
        std::string cTE("TE");
        std::string cW("W");
        std::string cC("C");
        std::string cE("E");
        std::string cBW("BW");
        std::string cB("B");
        std::string cBE("BE");
        std::string *cells[] = {&cTW, &cT, &cTE, &cW, &cC, &cE, &cBW, &cB, &cBE};
        PointerNeighborhood<std::string, Stencils::Moore<2, 1> > hood(cells);

        TS_ASSERT_EQUALS("TW", (hood[FixedCoord<-1, -1>()]));
        TS_ASSERT_EQUALS("C",  (hood[FixedCoord< 0,  0>()]));
        TS_ASSERT_EQUALS("BW", (hood[FixedCoord<-1,  1>()]));
        TS_ASSERT_EQUALS("BE", (hood[FixedCoord< 1,  1>()]));
    }

    void testMoore3D()
    {
        std::string cSTW("STW");
        std::string cST("ST");
        std::string cSTE("STE");
        std::string cSW("SW");
        std::string cS("S");
        std::string cSE("SE");
        std::string cSBW("SBW");
        std::string cSB("SB");
        std::string cSBE("SBE");

        std::string cTW("TW");
        std::string cT("T");
        std::string cTE("TE");
        std::string cW("W");
        std::string cC("C");
        std::string cE("E");
        std::string cBW("BW");
        std::string cB("B");
        std::string cBE("BE");

        std::string cNTW("NTW");
        std::string cNT("NT");
        std::string cNTE("NTE");
        std::string cNW("NW");
        std::string cN("N");
        std::string cNE("NE");
        std::string cNBW("NBW");
        std::string cNB("NB");
        std::string cNBE("NBE");
        std::string *cells[] = {
            &cSTW, &cST, &cSTE, &cSW, &cS, &cSE, &cSBW, &cSB, &cSBE,
            &cTW, &cT, &cTE, &cW, &cC, &cE, &cBW, &cB, &cBE,
            &cNTW, &cNT, &cNTE, &cNW, &cN, &cNE, &cNBW, &cNB, &cNBE};

        PointerNeighborhood<std::string, Stencils::Moore<3, 1> > hood(cells);

        TS_ASSERT_EQUALS("STW", (hood[FixedCoord<-1, -1, -1>()]));
        TS_ASSERT_EQUALS("C",   (hood[FixedCoord< 0,  0,  0>()]));
        TS_ASSERT_EQUALS("NBE", (hood[FixedCoord< 1,  1,  1>()]));
        TS_ASSERT_EQUALS("B",   (hood[FixedCoord< 0,  1,  0>()]));
        TS_ASSERT_EQUALS("SBE", (hood[FixedCoord< 1,  1, -1>()]));
    }

    void testVonNeumann3D()
    {
        std::string cS("S");
        std::string cT("T");
        std::string cW("W");
        std::string cC("C");
        std::string cE("E");
        std::string cB("B");
        std::string cN("N");
        std::string *cells[] = {&cS, &cT,&cW, &cC, &cE, &cB, &cN};
        PointerNeighborhood<std::string, Stencils::VonNeumann<3, 1> > hood(cells);

        TS_ASSERT_EQUALS("C",   (hood[FixedCoord< 0,  0,  0>()]));
        TS_ASSERT_EQUALS("T",   (hood[FixedCoord< 0, -1,  0>()]));
        TS_ASSERT_EQUALS("E",   (hood[FixedCoord< 1,  0,  0>()]));
        TS_ASSERT_EQUALS("N",   (hood[FixedCoord< 0,  0,  1>()]));
    }
};

}

