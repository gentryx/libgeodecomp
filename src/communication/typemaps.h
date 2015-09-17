#include<libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_MPI
#ifndef LIBGEODECOMP_TYPEMAPS_H
#define LIBGEODECOMP_TYPEMAPS_H

#include <mpi.h>
#include <complex>
#include <geometry/coord.h>
#include <geometry/coord.h>
#include <geometry/coord.h>
#include <geometry/coordbox.h>
#include <geometry/coordbox.h>
#include <geometry/coordbox.h>
#include <geometry/coordbox.h>
#include <storage/fixedarray.h>
#include <geometry/floatcoord.h>
#include <geometry/floatcoord.h>
#include <geometry/floatcoord.h>
#include <geometry/floatcoord.h>
#include <geometry/streak.h>
#include <geometry/streak.h>
#include <geometry/streak.h>
#include <geometry/streak.h>
#include <misc/testcell.h>
#include <misc/testcell.h>
#include <misc/testcell.h>
#include <misc/testcell.h>
#include <misc/chronometer.h>

extern MPI_Datatype MPI_LIBGEODECOMP_COORD_1_;
extern MPI_Datatype MPI_LIBGEODECOMP_COORD_2_;
extern MPI_Datatype MPI_LIBGEODECOMP_COORD_3_;
extern MPI_Datatype MPI_LIBGEODECOMP_COORDBOX_1_;
extern MPI_Datatype MPI_LIBGEODECOMP_COORDBOX_2_;
extern MPI_Datatype MPI_LIBGEODECOMP_COORDBOX_3_;
extern MPI_Datatype MPI_LIBGEODECOMP_COORDBOXMPIDATATYPEHELPER;
extern MPI_Datatype MPI_LIBGEODECOMP_FIXEDARRAY_DOUBLE_CHRONOMETER_NUM_INTERVALS_;
extern MPI_Datatype MPI_LIBGEODECOMP_FLOATCOORD_1_;
extern MPI_Datatype MPI_LIBGEODECOMP_FLOATCOORD_2_;
extern MPI_Datatype MPI_LIBGEODECOMP_FLOATCOORD_3_;
extern MPI_Datatype MPI_LIBGEODECOMP_FLOATCOORDMPIDATATYPEHELPER;
extern MPI_Datatype MPI_LIBGEODECOMP_STREAK_1_;
extern MPI_Datatype MPI_LIBGEODECOMP_STREAK_2_;
extern MPI_Datatype MPI_LIBGEODECOMP_STREAK_3_;
extern MPI_Datatype MPI_LIBGEODECOMP_STREAKMPIDATATYPEHELPER;
extern MPI_Datatype MPI_LIBGEODECOMP_TESTCELL_1_;
extern MPI_Datatype MPI_LIBGEODECOMP_TESTCELL_2_;
extern MPI_Datatype MPI_LIBGEODECOMP_TESTCELL_3_;
extern MPI_Datatype MPI_LIBGEODECOMP_TESTCELLMPIDATATYPEHELPER;
extern MPI_Datatype MPI_LIBGEODECOMP_CHRONOMETER;

namespace LibGeoDecomp {
/**
 * Utility class which can set up and yield MPI datatypes for custom datatypes.
 *
 * AUTO-GENERATED CODE. DO NOT EDIT. CHANGES WILL BE LOST. Refer to
 * typemapgenerator for further reference.
 */
class Typemaps
{
public:
    /**
     * Sets up MPI datatypes for all registered types.
     */
    static void initializeMaps();

    /**
     * Avoids duplicate initialization
     */
    static inline void initializeMapsIfUninitialized()
    {
        if (!initialized()) {
            initializeMaps();
        }
    }

    /**
     * Query initialization state
     */
    static inline bool initialized()
    {
        return mapsCreated;
    }

    /**
     * Performs an internal lookup. Works for custom, registered types
     * and for built-in types (char, int, std::size_t...). Compilation
     * will fail for unknown types.
     */
    template<typename T>
    static inline MPI_Datatype lookup()
    {
        return lookup((T*)0);
    }

private:
    template<typename T>
    static MPI_Aint getAddress(T *address)
    {
        MPI_Aint ret;
        MPI_Get_address(address, &ret);
        return ret;
    }

    static bool mapsCreated;

    static MPI_Datatype generateMapLibGeoDecomp_Coord_1_();
    static MPI_Datatype generateMapLibGeoDecomp_Coord_2_();
    static MPI_Datatype generateMapLibGeoDecomp_Coord_3_();
    static MPI_Datatype generateMapLibGeoDecomp_CoordBox_1_();
    static MPI_Datatype generateMapLibGeoDecomp_CoordBox_2_();
    static MPI_Datatype generateMapLibGeoDecomp_CoordBox_3_();
    static MPI_Datatype generateMapLibGeoDecomp_CoordBoxMPIDatatypeHelper();
    static MPI_Datatype generateMapLibGeoDecomp_FixedArray_double_Chronometer_NUM_INTERVALS_();
    static MPI_Datatype generateMapLibGeoDecomp_FloatCoord_1_();
    static MPI_Datatype generateMapLibGeoDecomp_FloatCoord_2_();
    static MPI_Datatype generateMapLibGeoDecomp_FloatCoord_3_();
    static MPI_Datatype generateMapLibGeoDecomp_FloatCoordMPIDatatypeHelper();
    static MPI_Datatype generateMapLibGeoDecomp_Streak_1_();
    static MPI_Datatype generateMapLibGeoDecomp_Streak_2_();
    static MPI_Datatype generateMapLibGeoDecomp_Streak_3_();
    static MPI_Datatype generateMapLibGeoDecomp_StreakMPIDatatypeHelper();
    static MPI_Datatype generateMapLibGeoDecomp_TestCell_1_();
    static MPI_Datatype generateMapLibGeoDecomp_TestCell_2_();
    static MPI_Datatype generateMapLibGeoDecomp_TestCell_3_();
    static MPI_Datatype generateMapLibGeoDecomp_TestCellMPIDatatypeHelper();
    static MPI_Datatype generateMapLibGeoDecomp_Chronometer();

public:
    static inline MPI_Datatype lookup(bool*)
    {
        return MPI_CHAR;
    }

    static inline MPI_Datatype lookup(char*)
    {
        return MPI_CHAR;
    }

    static inline MPI_Datatype lookup(double*)
    {
        return MPI_DOUBLE;
    }

    static inline MPI_Datatype lookup(float*)
    {
        return MPI_FLOAT;
    }

    static inline MPI_Datatype lookup(int*)
    {
        return MPI_INT;
    }

    static inline MPI_Datatype lookup(long*)
    {
        return MPI_LONG;
    }

    static inline MPI_Datatype lookup(long double*)
    {
        return MPI_LONG_DOUBLE;
    }

    static inline MPI_Datatype lookup(long long*)
    {
        return MPI_LONG_LONG;
    }

    static inline MPI_Datatype lookup(short*)
    {
        return MPI_SHORT;
    }

    static inline MPI_Datatype lookup(signed char*)
    {
        return MPI_SIGNED_CHAR;
    }

    static inline MPI_Datatype lookup(std::complex<double>*)
    {
        return MPI_DOUBLE_COMPLEX;
    }

    static inline MPI_Datatype lookup(std::complex<float>*)
    {
        return MPI_COMPLEX;
    }

    static inline MPI_Datatype lookup(unsigned*)
    {
        return MPI_UNSIGNED;
    }

    static inline MPI_Datatype lookup(unsigned char*)
    {
        return MPI_UNSIGNED_CHAR;
    }

    static inline MPI_Datatype lookup(unsigned long*)
    {
        return MPI_UNSIGNED_LONG;
    }

    static inline MPI_Datatype lookup(unsigned long long*)
    {
        return MPI_UNSIGNED_LONG_LONG;
    }

    static inline MPI_Datatype lookup(unsigned short*)
    {
        return MPI_UNSIGNED_SHORT;
    }

    static inline MPI_Datatype lookup(wchar_t*)
    {
        return MPI_WCHAR;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::Coord<1 >*)
    {
        return MPI_LIBGEODECOMP_COORD_1_;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::Coord<2 >*)
    {
        return MPI_LIBGEODECOMP_COORD_2_;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::Coord<3 >*)
    {
        return MPI_LIBGEODECOMP_COORD_3_;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::CoordBox<1 >*)
    {
        return MPI_LIBGEODECOMP_COORDBOX_1_;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::CoordBox<2 >*)
    {
        return MPI_LIBGEODECOMP_COORDBOX_2_;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::CoordBox<3 >*)
    {
        return MPI_LIBGEODECOMP_COORDBOX_3_;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::CoordBoxMPIDatatypeHelper*)
    {
        return MPI_LIBGEODECOMP_COORDBOXMPIDATATYPEHELPER;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::FixedArray<double,Chronometer::NUM_INTERVALS >*)
    {
        return MPI_LIBGEODECOMP_FIXEDARRAY_DOUBLE_CHRONOMETER_NUM_INTERVALS_;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::FloatCoord<1 >*)
    {
        return MPI_LIBGEODECOMP_FLOATCOORD_1_;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::FloatCoord<2 >*)
    {
        return MPI_LIBGEODECOMP_FLOATCOORD_2_;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::FloatCoord<3 >*)
    {
        return MPI_LIBGEODECOMP_FLOATCOORD_3_;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::FloatCoordMPIDatatypeHelper*)
    {
        return MPI_LIBGEODECOMP_FLOATCOORDMPIDATATYPEHELPER;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::Streak<1 >*)
    {
        return MPI_LIBGEODECOMP_STREAK_1_;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::Streak<2 >*)
    {
        return MPI_LIBGEODECOMP_STREAK_2_;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::Streak<3 >*)
    {
        return MPI_LIBGEODECOMP_STREAK_3_;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::StreakMPIDatatypeHelper*)
    {
        return MPI_LIBGEODECOMP_STREAKMPIDATATYPEHELPER;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::TestCell<1 >*)
    {
        return MPI_LIBGEODECOMP_TESTCELL_1_;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::TestCell<2 >*)
    {
        return MPI_LIBGEODECOMP_TESTCELL_2_;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::TestCell<3 >*)
    {
        return MPI_LIBGEODECOMP_TESTCELL_3_;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::TestCellMPIDatatypeHelper*)
    {
        return MPI_LIBGEODECOMP_TESTCELLMPIDATATYPEHELPER;
    }

    static inline MPI_Datatype lookup(LibGeoDecomp::Chronometer*)
    {
        return MPI_LIBGEODECOMP_CHRONOMETER;
    }

};
}

#endif

#endif
