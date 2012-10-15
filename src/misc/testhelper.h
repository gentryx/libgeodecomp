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
#define TSM_ASSERT_ROUGHLY_EQUALS_DOUBLE(msg, va, vb, accuracy)         \
    {                                                                   \
        double tsa_comp1_a = va;                                        \
        double tsa_comp1_b = vb;                                        \
        double tsa_delta = std::max(fabs(tsa_comp1_a), fabs(tsa_comp1_b)) * accuracy; \
        TSM_ASSERT_DELTA(msg, tsa_comp1_a, tsa_comp1_b, tsa_delta);     \
    }


#define TSM_ASSERT_EQUALS_DOUBLE(msg, va, vb)                           \
    {                                                                   \
        TSM_ASSERT_ROUGHLY_EQUALS_DOUBLE(msg, va, vb, 2.0e-13);         \
    }


#define TS_ASSERT_EQUALS_DOUBLE(va, vb)                                 \
    {                                                                   \
        TSM_ASSERT_EQUALS_DOUBLE("", va, vb);                           \
    }


#define TS_ASSERT_EQUALS_DOUBLE_VEC(va, vb)                             \
    {                                                                   \
        SuperVector<double> tsa_comp2_a = va;                           \
        SuperVector<double> tsa_comp2_b = vb;                           \
        TS_ASSERT_EQUALS(tsa_comp2_a.size(), tsa_comp2_b.size());       \
        for (unsigned i = 0; i < tsa_comp2_b.size(); i++) {             \
            TS_ASSERT_EQUALS_DOUBLE(tsa_comp2_a[i], tsa_comp2_b[i]);    \
        }                                                               \
    }


#define TS_ASSERT_FILE(filename)                                        \
    {                                                                   \
        boost::filesystem::path path(filename);                         \
        TSM_ASSERT("File " + filename + " should exist, but doesn't",   \
                   boost::filesystem::exists(path));                    \
    }

#define TS_ASSERT_NO_FILE(filename)                                     \
    {                                                                   \
        boost::filesystem::path path(filename);                         \
        TSM_ASSERT("File " + filename + " should not exist, but does",  \
                   !boost::filesystem::exists(path));                   \
    }

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
    }

#define TS_ASSERT_TEST_GRID2(_GRID_TYPE, _GRID, _EXPECTED_CYCLE, TYPENAME) \
    {                                                                   \
        const _GRID_TYPE& assertGrid = _GRID;                           \
        unsigned expectedCycle = _EXPECTED_CYCLE;                       \
        bool ollKorrect = true;                                         \
        std::ostringstream message;                                     \
                                                                        \
        TS_ASSERT(assertGrid.atEdge().isEdgeCell);                      \
        if (!assertGrid.atEdge().isEdgeCell) {                          \
            ollKorrect = false;                                         \
            message << "edgeCell isn't an edgeCell\n";                  \
        }                                                               \
        if (!assertGrid.atEdge().valid()) {                             \
            ollKorrect = false;                                         \
            message << "edgeCell isn't valid\n";                        \
        }                                                               \
        CoordBox<_GRID_TYPE::DIM> box = assertGrid.boundingBox();       \
        for (TYPENAME CoordBox<_GRID_TYPE::DIM>::Iterator i = box.begin(); i != box.end(); ++i) { \
            bool flagValid   = assertGrid.at(*i).valid();               \
            bool flagEdge    = (assertGrid.at(*i).isEdgeCell == false); \
            bool flagCounter = (assertGrid.at(*i).cycleCounter == expectedCycle); \
            message << "actual: " << (assertGrid.at(*i).cycleCounter) << " expected: " << expectedCycle << "\n"; \
            TS_ASSERT(flagValid);                                       \
            TS_ASSERT(flagEdge);                                        \
            TS_ASSERT(flagCounter);                                     \
            bool flag = flagValid && flagEdge && flagCounter;           \
            ollKorrect &= flag;                                         \
            if (!flag) {                                                \
                message << "TS_ASSERT_TEST_GRID failed at Coord " << *i << "\n"; \
            }                                                           \
        }                                                               \
                                                                        \
        if (!ollKorrect) {                                              \
            std::cout << "message: " << message.str();                  \
        }                                                               \
    } 

#define TS_ASSERT_TEST_GRID(_GRID_TYPE, _GRID, _EXPECTED_CYCLE) \
    TS_ASSERT_TEST_GRID2(_GRID_TYPE, _GRID, _EXPECTED_CYCLE, )

#define TS_ASSERT_TEST_GRID_REGION(_GRID_TYPE, _GRID, _REGION, _EXPECTED_CYCLE) \
    {                                                                   \
        const _GRID_TYPE& assertGrid = _GRID;                           \
        Region<_GRID_TYPE::DIM> assertRegion = _REGION;                 \
        unsigned expectedCycle = _EXPECTED_CYCLE;                       \
        bool ollKorrect = true;                                         \
        std::ostringstream message;                                     \
                                                                        \
        ollKorrect &= assertGrid.atEdge().isEdgeCell;                   \
        ollKorrect &= assertGrid.atEdge().valid();                      \
        Region<_GRID_TYPE::DIM>::Iterator end = assertRegion.end();     \
        for (Region<_GRID_TYPE::DIM>::Iterator i = assertRegion.begin(); i != end; ++i) { \
            bool flag = assertGrid.at(*i).valid();                      \
            flag &= (assertGrid.at(*i).isEdgeCell == false);            \
            flag &= (assertGrid.at(*i).cycleCounter == expectedCycle);  \
            TS_ASSERT(flag);                                            \
            ollKorrect &= flag;                                         \
            if (!flag)                                                  \
                message << "TS_ASSERT_TEST_GRID_REGION failed at Coord " << *i << "\n"; \
        }                                                               \
    }

#endif
