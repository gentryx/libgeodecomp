#ifndef _libgeodecomp_misc_testhelper_h_
#define _libgeodecomp_misc_testhelper_h_

#ifdef __CODEGEARC__
#include <math.h>
#else
#include <cmath>
#endif

#include <boost/filesystem.hpp>
#include <fstream>
#include <sstream>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/misc/stringops.h>
#include <libgeodecomp/misc/supervector.h>

/**
 * This macro differs from TS_ASSERT_DELTA in that the error margin is relative
 * rather than absolute
 */
#define TSM_ASSERT_ROUGHLY_EQUALS_DOUBLE(msg, va, vb, accuracy)                          \
{                                                                                        \
    double tsa_comp1_a = va;                                                             \
    double tsa_comp1_b = vb;                                                             \
    double tsa_delta = std::max(fabs(tsa_comp1_a), fabs(tsa_comp1_b)) * accuracy;        \
    TSM_ASSERT_DELTA(msg, tsa_comp1_a, tsa_comp1_b, tsa_delta);                          \
} while (0)


#define TSM_ASSERT_EQUALS_DOUBLE(msg, va, vb)                                            \
{                                                                                        \
    TSM_ASSERT_ROUGHLY_EQUALS_DOUBLE(msg, va, vb, 2.0e-13);                              \
} while (0)


#define TS_ASSERT_EQUALS_DOUBLE(va, vb)                                                  \
{                                                                                        \
    TSM_ASSERT_EQUALS_DOUBLE("", va, vb);                                                \
} while (0)


/**
 * ASSERT_EQUAL macro that generates a sensible error message automatically
 */
#define TSMA_ASSERT_EQUALS(actual, expected)                                             \
{                                                                                        \
    std::ostringstream message;                                                          \
    message                                                                              \
        << "\nExpected\n" << expected                                                    \
        << "\nbut was\n" << actual;                                                      \
    TSM_ASSERT_EQUALS(message.str().c_str(), actual, expected);                          \
} while (0)


#define TS_ASSERT_EQUALS_DVEC(va, vb)                                                    \
{                                                                                        \
    DVec tsa_comp2_a = va;                                                               \
    DVec tsa_comp2_b = vb;                                                               \
    TS_ASSERT_EQUALS(tsa_comp2_a.size(), tsa_comp2_b.size());                            \
    for (unsigned i = 0; i < tsa_comp2_b.size(); i++)                                    \
        TS_ASSERT_EQUALS_DOUBLE(tsa_comp2_a[i], tsa_comp2_b[i]);                         \
} while (0)


#define TS_ASSERT_FILE(filename)                                                         \
{                                                                                        \
    boost::filesystem::path path(filename);                                              \
    TSM_ASSERT("File " + filename + " should exist, but doesn't",       \
        boost::filesystem::exists(path));                                                \
} while (0)


#define TS_ASSERT_NO_FILE(filename)                                                      \
{                                                                                        \
    boost::filesystem::path path(filename);                                              \
    TSM_ASSERT("File " + filename + " should not exist, but does",                       \
        !boost::filesystem::exists(path));                                               \
} while (0)

#define TS_ASSERT_FILE_CONTENTS_EQUAL(_filename1, _filename2)           \
    {                                                                   \
        std::string filename1 = _filename1;                             \
        std::string filename2 = _filename2;                             \
        TS_ASSERT_FILE(filename1);                                      \
        TS_ASSERT_FILE(filename2);                                      \
        std::ifstream in1(filename1.c_str());                           \
        std::ifstream in2(filename2.c_str());                           \
                                                                        \
        char ch1, ch2;                                                  \
        int counter = 0;                                                \
        while (in1.get(ch1)) {                                          \
            if (in2.get(ch2)) {                                         \
                std::string message = "Contents differ at byte " +      \
                    StringConv::itoa(counter);                          \
                TSM_ASSERT_EQUALS(message.c_str(), ch1, ch2);           \
            } else {                                                    \
                TSM_ASSERT("File lengths differ", false);               \
                break;                                                  \
            }                                                           \
            counter++;                                                  \
        }                                                               \
        if (in2.get(ch2)) {                                             \
            TSM_ASSERT("File lengths differ", false);                   \
        }                                                               \
    } while(0)

// fixme: recheck that all those methods do the right thing
// fixme: these macros are so ugly...
// fixme: ugly that i have to specify the type here, could get rid of
// that with a template function



#define TS_ASSERT_TEST_GRID(_GRID_TYPE, _GRID, _EXPECTED_CYCLE)         \
    {                                                                   \
        _GRID_TYPE assertGrid = _GRID;                                  \
        unsigned expectedCycle = _EXPECTED_CYCLE;                       \
        bool ollKorrect = true;                                         \
        std::ostringstream message;                                     \
                                                                        \
        TS_ASSERT(assertGrid.getEdgeCell().isEdgeCell);                 \
        if (!assertGrid.getEdgeCell().isEdgeCell) {                     \
            ollKorrect = false;                                         \
            message << "edgeCell isn't an edgeCell\n";                  \
        }                                                               \
        if (!assertGrid.getEdgeCell().valid()) {                        \
            ollKorrect = false;                                         \
            message << "edgeCell isn't valid\n";                        \
        }                                                               \
        CoordBoxSequence<_GRID_TYPE::DIMENSIONS> i =                    \
            assertGrid.boundingBox().sequence();                        \
        while (i.hasNext()) {                                           \
            LibGeoDecomp::Coord<_GRID_TYPE::DIMENSIONS> c = i.next();   \
            bool flag = assertGrid[c].valid();                          \
            flag &= (assertGrid[c].isEdgeCell == false);                \
            flag &= (assertGrid[c].cycleCounter == expectedCycle);      \
            TS_ASSERT(flag);                                            \
            ollKorrect &= flag;                                         \
            if (!flag)                                                  \
                message << "TS_ASSERT_GRID failed at Coord " << c << "\n"; \
        }                                                               \
                                                                        \
        if (!ollKorrect)                                                \
            std::cout << "message: " << message.str();                  \
    } while(0)

#define TS_ASSERT_TEST_GRID_REGION(_GRID_TYPE, _GRID, _REGION, _EXPECTED_CYCLE) \
    {                                                                   \
        _GRID_TYPE assertGrid = _GRID;                                  \
        Region<2> assertRegion = _REGION;                               \
        unsigned expectedCycle = _EXPECTED_CYCLE;                       \
        bool ollKorrect = true;                                         \
        std::ostringstream message;                                     \
                                                                        \
        ollKorrect &= assertGrid.getEdgeCell().isEdgeCell;              \
        ollKorrect &= assertGrid.getEdgeCell().valid();                 \
        Region<2>::Iterator end = assertRegion.end();                   \
        for (Region<2>::Iterator i = assertRegion.begin(); i != end; ++i) { \
            bool flag = assertGrid[*i].valid();                         \
            flag &= (assertGrid[*i].isEdgeCell == false);               \
            flag &= (assertGrid[*i].cycleCounter == expectedCycle);     \
            TS_ASSERT(flag);                                            \
            ollKorrect &= flag;                                         \
            if (!flag)                                                  \
                message << "TS_ASSERT_GRID_REGION failed at Coord " << *i << "\n"; \
        }                                                               \
                                                                        \
        if (!ollKorrect)                                                \
            std::cout << message.str() << assertGrid;                   \
    } while(0)


#define TS_ASSERT_TEST_GRID_NO_CYCLE(_GRID_TYPE, _GRID)                 \
    {                                                                   \
        _GRID_TYPE assertGrid = _GRID;                                  \
        bool ollKorrect = true;                                         \
        std::ostringstream message;                                     \
                                                                        \
        ollKorrect &= assertGrid.getEdgeCell().isEdgeCell;              \
        ollKorrect &= assertGrid.getEdgeCell().valid();                 \
        for (unsigned x = 0; x < assertGrid.getDimensions().x(); x++) {   \
            for (unsigned y = 0; y < assertGrid.getDimensions().y(); y++) { \
                LibGeoDecomp::Coord<2> c(x, y);                         \
                Coord<2> origin = assertGrid.boundingBox().origin;      \
                bool flag = assertGrid[c + origin].valid();             \
                flag &= (assertGrid[c + origin].isEdgeCell == false);   \
                TS_ASSERT(flag);                                        \
                ollKorrect &= flag;                                     \
                if (!flag)                                              \
                    message << "TS_ASSERT_GRID_NO_CYCLE failed at Coord " << c << "\n"; \
            }                                                           \
        }                                                               \
                                                                        \
        if (!ollKorrect)                                                \
            std::cout << message.str() << assertGrid;                   \
    } while(0)

#endif
