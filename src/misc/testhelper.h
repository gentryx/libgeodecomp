#ifndef LIBGEODECOMP_MISC_TESTHELPER_H
#define LIBGEODECOMP_MISC_TESTHELPER_H

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/misc/stringops.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/storage/grid.h>

#ifdef __CODEGEARC__
#include <math.h>
#else
#include <cmath>
#endif

#include <boost/filesystem.hpp>
#include <fstream>
#include <sstream>
#include <cxxtest/TestSuite.h>

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
        std::vector<double> tsa_comp2_a = va;                           \
        std::vector<double> tsa_comp2_b = vb;                           \
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
                    StringOps::itoa(counter);                           \
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
        TS_ASSERT(assertGrid.getEdge().edgeCell());                     \
        if (!assertGrid.getEdge().edgeCell()) {                         \
            ollKorrect = false;                                         \
            message << "edgeCell isn't an edgeCell\n";                  \
        }                                                               \
        if (!assertGrid.getEdge().valid()) {                            \
            ollKorrect = false;                                         \
            message << "edgeCell isn't valid\n";                        \
        }                                                               \
        CoordBox<_GRID_TYPE::DIM> box = assertGrid.boundingBox();       \
        for (TYPENAME CoordBox<_GRID_TYPE::DIM>::Iterator i = box.begin(); i != box.end(); ++i) { \
            bool flagValid   = assertGrid.get(*i).valid();              \
            bool flagEdge    = (assertGrid.get(*i).edgeCell() == false); \
            bool flagCounter = (assertGrid.get(*i).cycleCounter == expectedCycle); \
            if (assertGrid.get(*i).cycleCounter != expectedCycle) {     \
                message << "actual: " << (assertGrid.get(*i).cycleCounter) \
                        << " expected: " << expectedCycle << "\n";      \
            }                                                           \
            TS_ASSERT(flagValid);                                       \
            TS_ASSERT(flagEdge);                                        \
            TS_ASSERT(flagCounter);                                     \
            bool flag = flagValid && flagEdge && flagCounter;           \
            ollKorrect &= flag;                                         \
            if (!flag) {                                                \
                message << "TS_ASSERT_TEST_GRID failed at Coord " << *i << "\n" \
                        << "  flagValid: " << flagValid << "\n"         \
                        << "  flagEdge: " << flagEdge << "\n"           \
                        << "  flagCounter: " << flagCounter << "\n";    \
            }                                                           \
        }                                                               \
                                                                        \
        if (!ollKorrect) {                                              \
            std::cout << "message: " << message.str();                  \
        }                                                               \
    }

#define TS_ASSERT_TEST_GRID(_GRID_TYPE, _GRID, _EXPECTED_CYCLE) \
    TS_ASSERT_TEST_GRID2(_GRID_TYPE, _GRID, _EXPECTED_CYCLE, )

#define TS_ASSERT_TEST_GRID_REGION2(_GRID_TYPE, _GRID, _REGION, _EXPECTED_CYCLE, _TYPENAME) \
    {                                                                   \
        const _GRID_TYPE& assertGrid = _GRID;                           \
        Region<_GRID_TYPE::DIM> assertRegion = _REGION;                 \
        unsigned expectedCycle = _EXPECTED_CYCLE;                       \
        bool ollKorrect = true;                                         \
                                                                        \
        ollKorrect &= assertGrid.getEdge().edgeCell();                  \
        ollKorrect &= assertGrid.getEdge().valid();                     \
        _TYPENAME Region<_GRID_TYPE::DIM>::Iterator end = assertRegion.end();     \
        for (_TYPENAME Region<_GRID_TYPE::DIM>::Iterator i = assertRegion.begin(); i != end; ++i) { \
            bool flag = assertGrid.get(*i).valid();                     \
            if (!flag) {                                                \
                std::cout << "TestCell not valid\n";                    \
            }                                                           \
            flag &= (assertGrid.get(*i).edgeCell() == false);           \
            if (!flag) {                                                \
                std::cout << "TestCell claims to be edge cell\n";       \
            }                                                           \
            flag &= (assertGrid.get(*i).cycleCounter == expectedCycle); \
            if (!flag) {                                                \
                std::cout << "TestCell cycle counter doesn't match expected value (" \
                          << assertGrid.get(*i).cycleCounter << " != "  \
                          << expectedCycle << ")\n";                    \
            }                                                           \
            TS_ASSERT(flag);                                            \
            ollKorrect &= flag;                                         \
            if (!flag) {                                                \
                std::cout << "TS_ASSERT_TEST_GRID_REGION failed at Coord " << *i << "\n"; \
            }                                                           \
        }                                                               \
    }

#define TS_ASSERT_TEST_GRID_REGION(_GRID_TYPE, _GRID, _REGION, _EXPECTED_CYCLE) \
    TS_ASSERT_TEST_GRID_REGION2(_GRID_TYPE, _GRID, _REGION, _EXPECTED_CYCLE, )

#endif
