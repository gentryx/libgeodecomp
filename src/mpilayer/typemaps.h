#include<libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_typemaps_h_
#define _libgeodecomp_typemaps_h_

#include <complex>
#include <mpi.h>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/simplecell.h>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/streak.h>
#include <libgeodecomp/misc/streak.h>
#include <libgeodecomp/misc/streak.h>
#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/streak.h>
#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/testcell.h>

namespace MPI {
    extern Datatype LIBGEODECOMP_COORD_1_;
    extern Datatype LIBGEODECOMP_COORD_3_;
    extern Datatype LIBGEODECOMP_SIMPLECELL;
    extern Datatype LIBGEODECOMP_COORD_2_;
    extern Datatype LIBGEODECOMP_STREAK_1_;
    extern Datatype LIBGEODECOMP_STREAK_2_;
    extern Datatype LIBGEODECOMP_STREAK_3_;
    extern Datatype LIBGEODECOMP_COORDBOX_1_;
    extern Datatype LIBGEODECOMP_COORDBOX_2_;
    extern Datatype LIBGEODECOMP_COORDBOX_3_;
    extern Datatype LIBGEODECOMP_STREAKMPIDATATYPEHELPER;
    extern Datatype LIBGEODECOMP_COORDBOXMPIDATATYPEHELPER;
    extern Datatype LIBGEODECOMP_TESTCELL_1_;
    extern Datatype LIBGEODECOMP_TESTCELL_2_;
    extern Datatype LIBGEODECOMP_TESTCELL_3_;
    extern Datatype LIBGEODECOMP_TESTCELLMPIDATATYPEHELPER;
}

namespace LibGeoDecomp {
class Typemaps {
public:
    static void initializeMaps();

    template<typename T>
    static inline MPI::Datatype lookup() {
        return lookup((T*)0);
    }

private:
    static MPI::Datatype generateMapLibGeoDecomp_Coord_1_();
    static MPI::Datatype generateMapLibGeoDecomp_Coord_3_();
    static MPI::Datatype generateMapLibGeoDecomp_SimpleCell();
    static MPI::Datatype generateMapLibGeoDecomp_Coord_2_();
    static MPI::Datatype generateMapLibGeoDecomp_Streak_1_();
    static MPI::Datatype generateMapLibGeoDecomp_Streak_2_();
    static MPI::Datatype generateMapLibGeoDecomp_Streak_3_();
    static MPI::Datatype generateMapLibGeoDecomp_CoordBox_1_();
    static MPI::Datatype generateMapLibGeoDecomp_CoordBox_2_();
    static MPI::Datatype generateMapLibGeoDecomp_CoordBox_3_();
    static MPI::Datatype generateMapLibGeoDecomp_StreakMPIDatatypeHelper();
    static MPI::Datatype generateMapLibGeoDecomp_CoordBoxMPIDatatypeHelper();
    static MPI::Datatype generateMapLibGeoDecomp_TestCell_1_();
    static MPI::Datatype generateMapLibGeoDecomp_TestCell_2_();
    static MPI::Datatype generateMapLibGeoDecomp_TestCell_3_();
    static MPI::Datatype generateMapLibGeoDecomp_TestCellMPIDatatypeHelper();

    static inline MPI::Datatype lookup(bool*) { return MPI::BOOL; }
    static inline MPI::Datatype lookup(char*) { return MPI::CHAR; }
    static inline MPI::Datatype lookup(double*) { return MPI::DOUBLE; }
    static inline MPI::Datatype lookup(float*) { return MPI::FLOAT; }
    static inline MPI::Datatype lookup(int*) { return MPI::INT; }
    static inline MPI::Datatype lookup(long*) { return MPI::LONG; }
    static inline MPI::Datatype lookup(long double*) { return MPI::LONG_DOUBLE; }
    static inline MPI::Datatype lookup(long long*) { return MPI::LONG_LONG; }
    static inline MPI::Datatype lookup(short*) { return MPI::SHORT; }
    static inline MPI::Datatype lookup(signed char*) { return MPI::SIGNED_CHAR; }
    static inline MPI::Datatype lookup(std::complex<double>*) { return MPI::DOUBLE_COMPLEX; }
    static inline MPI::Datatype lookup(std::complex<float>*) { return MPI::COMPLEX; }
    static inline MPI::Datatype lookup(std::complex<long double>*) { return MPI::LONG_DOUBLE_COMPLEX; }
    static inline MPI::Datatype lookup(unsigned*) { return MPI::UNSIGNED; }
    static inline MPI::Datatype lookup(unsigned char*) { return MPI::UNSIGNED_CHAR; }
    static inline MPI::Datatype lookup(unsigned long*) { return MPI::UNSIGNED_LONG; }
    static inline MPI::Datatype lookup(unsigned long long*) { return MPI::UNSIGNED_LONG_LONG; }
    static inline MPI::Datatype lookup(unsigned short*) { return MPI::UNSIGNED_SHORT; }
    static inline MPI::Datatype lookup(wchar_t*) { return MPI::WCHAR; }
    static inline MPI::Datatype lookup(LibGeoDecomp::Coord<1 >*) { return MPI::LIBGEODECOMP_COORD_1_; }
    static inline MPI::Datatype lookup(LibGeoDecomp::Coord<3 >*) { return MPI::LIBGEODECOMP_COORD_3_; }
    static inline MPI::Datatype lookup(LibGeoDecomp::SimpleCell*) { return MPI::LIBGEODECOMP_SIMPLECELL; }
    static inline MPI::Datatype lookup(LibGeoDecomp::Coord<2 >*) { return MPI::LIBGEODECOMP_COORD_2_; }
    static inline MPI::Datatype lookup(LibGeoDecomp::Streak<1 >*) { return MPI::LIBGEODECOMP_STREAK_1_; }
    static inline MPI::Datatype lookup(LibGeoDecomp::Streak<2 >*) { return MPI::LIBGEODECOMP_STREAK_2_; }
    static inline MPI::Datatype lookup(LibGeoDecomp::Streak<3 >*) { return MPI::LIBGEODECOMP_STREAK_3_; }
    static inline MPI::Datatype lookup(LibGeoDecomp::CoordBox<1 >*) { return MPI::LIBGEODECOMP_COORDBOX_1_; }
    static inline MPI::Datatype lookup(LibGeoDecomp::CoordBox<2 >*) { return MPI::LIBGEODECOMP_COORDBOX_2_; }
    static inline MPI::Datatype lookup(LibGeoDecomp::CoordBox<3 >*) { return MPI::LIBGEODECOMP_COORDBOX_3_; }
    static inline MPI::Datatype lookup(LibGeoDecomp::StreakMPIDatatypeHelper*) { return MPI::LIBGEODECOMP_STREAKMPIDATATYPEHELPER; }
    static inline MPI::Datatype lookup(LibGeoDecomp::CoordBoxMPIDatatypeHelper*) { return MPI::LIBGEODECOMP_COORDBOXMPIDATATYPEHELPER; }
    static inline MPI::Datatype lookup(LibGeoDecomp::TestCell<1 >*) { return MPI::LIBGEODECOMP_TESTCELL_1_; }
    static inline MPI::Datatype lookup(LibGeoDecomp::TestCell<2 >*) { return MPI::LIBGEODECOMP_TESTCELL_2_; }
    static inline MPI::Datatype lookup(LibGeoDecomp::TestCell<3 >*) { return MPI::LIBGEODECOMP_TESTCELL_3_; }
    static inline MPI::Datatype lookup(LibGeoDecomp::TestCellMPIDatatypeHelper*) { return MPI::LIBGEODECOMP_TESTCELLMPIDATATYPEHELPER; }
};
};

#endif

#endif
