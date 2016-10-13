#include<libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_MPI
#include "typemaps.h"
#include <algorithm>
#include <stdexcept>

MPI_Datatype MPI_LIBGEODECOMP_COORD_1_;
MPI_Datatype MPI_LIBGEODECOMP_COORD_2_;
MPI_Datatype MPI_LIBGEODECOMP_COORD_3_;
MPI_Datatype MPI_LIBGEODECOMP_COORDBOX_1_;
MPI_Datatype MPI_LIBGEODECOMP_COORDBOX_2_;
MPI_Datatype MPI_LIBGEODECOMP_COORDBOX_3_;
MPI_Datatype MPI_LIBGEODECOMP_COORDBOXMPIDATATYPEHELPER;
MPI_Datatype MPI_LIBGEODECOMP_FIXEDARRAY_DOUBLE_100_;
MPI_Datatype MPI_LIBGEODECOMP_FIXEDARRAY_DOUBLE_CHRONOMETER_NUM_INTERVALS_;
MPI_Datatype MPI_LIBGEODECOMP_FIXEDARRAY_INT_100_;
MPI_Datatype MPI_LIBGEODECOMP_FLOATCOORD_1_;
MPI_Datatype MPI_LIBGEODECOMP_FLOATCOORD_2_;
MPI_Datatype MPI_LIBGEODECOMP_FLOATCOORD_3_;
MPI_Datatype MPI_LIBGEODECOMP_FLOATCOORDMPIDATATYPEHELPER;
MPI_Datatype MPI_LIBGEODECOMP_STREAK_1_;
MPI_Datatype MPI_LIBGEODECOMP_STREAK_2_;
MPI_Datatype MPI_LIBGEODECOMP_STREAK_3_;
MPI_Datatype MPI_LIBGEODECOMP_STREAKMPIDATATYPEHELPER;
MPI_Datatype MPI_LIBGEODECOMP_TESTCELL_1_;
MPI_Datatype MPI_LIBGEODECOMP_TESTCELL_2_;
MPI_Datatype MPI_LIBGEODECOMP_TESTCELL_3_;
MPI_Datatype MPI_LIBGEODECOMP_TESTCELLMPIDATATYPEHELPER;
MPI_Datatype MPI_LIBGEODECOMP_UNSTRUCTUREDTESTCELL_UNSTRUCTUREDTESTCELLHELPERS_EMPTYAPI_;
MPI_Datatype MPI_LIBGEODECOMP_UNSTRUCTUREDTESTCELLMPIDATATYPEHELPER;
MPI_Datatype MPI_LIBGEODECOMP_CHRONOMETER;

namespace LibGeoDecomp {

// Member Specification, holds all relevant information for a given member.
class MemberSpec
{
public:
    MemberSpec(MPI_Aint address, MPI_Datatype type, int length) :
        address(address),
        type(type),
        length(length)
    {}

    MPI_Aint address;
    MPI_Datatype type;
    int length;
};

bool addressLower(MemberSpec a, MemberSpec b)
{
    return a.address < b.address;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_Coord_1_() {
    char fakeObject[sizeof(LibGeoDecomp::Coord<1 >)];
    LibGeoDecomp::Coord<1 > *obj = (LibGeoDecomp::Coord<1 >*)fakeObject;

    const int count = 1;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->c), lookup<int >(), 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_Coord_2_() {
    char fakeObject[sizeof(LibGeoDecomp::Coord<2 >)];
    LibGeoDecomp::Coord<2 > *obj = (LibGeoDecomp::Coord<2 >*)fakeObject;

    const int count = 1;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->c), lookup<int >(), 2)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_Coord_3_() {
    char fakeObject[sizeof(LibGeoDecomp::Coord<3 >)];
    LibGeoDecomp::Coord<3 > *obj = (LibGeoDecomp::Coord<3 >*)fakeObject;

    const int count = 1;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->c), lookup<int >(), 3)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_CoordBox_1_() {
    char fakeObject[sizeof(LibGeoDecomp::CoordBox<1 >)];
    LibGeoDecomp::CoordBox<1 > *obj = (LibGeoDecomp::CoordBox<1 >*)fakeObject;

    const int count = 2;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->dimensions), lookup<Coord<1 > >(), 1),
        MemberSpec(getAddress(&obj->origin), lookup<Coord<1 > >(), 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_CoordBox_2_() {
    char fakeObject[sizeof(LibGeoDecomp::CoordBox<2 >)];
    LibGeoDecomp::CoordBox<2 > *obj = (LibGeoDecomp::CoordBox<2 >*)fakeObject;

    const int count = 2;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->dimensions), lookup<Coord<2 > >(), 1),
        MemberSpec(getAddress(&obj->origin), lookup<Coord<2 > >(), 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_CoordBox_3_() {
    char fakeObject[sizeof(LibGeoDecomp::CoordBox<3 >)];
    LibGeoDecomp::CoordBox<3 > *obj = (LibGeoDecomp::CoordBox<3 >*)fakeObject;

    const int count = 2;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->dimensions), lookup<Coord<3 > >(), 1),
        MemberSpec(getAddress(&obj->origin), lookup<Coord<3 > >(), 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_CoordBoxMPIDatatypeHelper() {
    char fakeObject[sizeof(LibGeoDecomp::CoordBoxMPIDatatypeHelper)];
    LibGeoDecomp::CoordBoxMPIDatatypeHelper *obj = (LibGeoDecomp::CoordBoxMPIDatatypeHelper*)fakeObject;

    const int count = 3;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->a), lookup<CoordBox<1 > >(), 1),
        MemberSpec(getAddress(&obj->b), lookup<CoordBox<2 > >(), 1),
        MemberSpec(getAddress(&obj->c), lookup<CoordBox<3 > >(), 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_FixedArray_double_100_() {
    char fakeObject[sizeof(LibGeoDecomp::FixedArray<double,100 >)];
    LibGeoDecomp::FixedArray<double,100 > *obj = (LibGeoDecomp::FixedArray<double,100 >*)fakeObject;

    const int count = 2;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->elements), lookup<std::size_t >(), 1),
        MemberSpec(getAddress(&obj->store), lookup<double >(), 100)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_FixedArray_double_Chronometer_NUM_INTERVALS_() {
    char fakeObject[sizeof(LibGeoDecomp::FixedArray<double,Chronometer::NUM_INTERVALS >)];
    LibGeoDecomp::FixedArray<double,Chronometer::NUM_INTERVALS > *obj = (LibGeoDecomp::FixedArray<double,Chronometer::NUM_INTERVALS >*)fakeObject;

    const int count = 2;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->elements), lookup<std::size_t >(), 1),
        MemberSpec(getAddress(&obj->store), lookup<double >(), Chronometer::NUM_INTERVALS)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_FixedArray_int_100_() {
    char fakeObject[sizeof(LibGeoDecomp::FixedArray<int,100 >)];
    LibGeoDecomp::FixedArray<int,100 > *obj = (LibGeoDecomp::FixedArray<int,100 >*)fakeObject;

    const int count = 2;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->elements), lookup<std::size_t >(), 1),
        MemberSpec(getAddress(&obj->store), lookup<int >(), 100)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_FloatCoord_1_() {
    char fakeObject[sizeof(LibGeoDecomp::FloatCoord<1 >)];
    LibGeoDecomp::FloatCoord<1 > *obj = (LibGeoDecomp::FloatCoord<1 >*)fakeObject;

    const int count = 1;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->c), lookup<double >(), 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_FloatCoord_2_() {
    char fakeObject[sizeof(LibGeoDecomp::FloatCoord<2 >)];
    LibGeoDecomp::FloatCoord<2 > *obj = (LibGeoDecomp::FloatCoord<2 >*)fakeObject;

    const int count = 1;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->c), lookup<double >(), 2)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_FloatCoord_3_() {
    char fakeObject[sizeof(LibGeoDecomp::FloatCoord<3 >)];
    LibGeoDecomp::FloatCoord<3 > *obj = (LibGeoDecomp::FloatCoord<3 >*)fakeObject;

    const int count = 1;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->c), lookup<double >(), 3)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_FloatCoordMPIDatatypeHelper() {
    char fakeObject[sizeof(LibGeoDecomp::FloatCoordMPIDatatypeHelper)];
    LibGeoDecomp::FloatCoordMPIDatatypeHelper *obj = (LibGeoDecomp::FloatCoordMPIDatatypeHelper*)fakeObject;

    const int count = 3;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->a), lookup<FloatCoord<1 > >(), 1),
        MemberSpec(getAddress(&obj->b), lookup<FloatCoord<2 > >(), 1),
        MemberSpec(getAddress(&obj->c), lookup<FloatCoord<3 > >(), 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_Streak_1_() {
    char fakeObject[sizeof(LibGeoDecomp::Streak<1 >)];
    LibGeoDecomp::Streak<1 > *obj = (LibGeoDecomp::Streak<1 >*)fakeObject;

    const int count = 2;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->endX), lookup<int >(), 1),
        MemberSpec(getAddress(&obj->origin), lookup<Coord<1 > >(), 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_Streak_2_() {
    char fakeObject[sizeof(LibGeoDecomp::Streak<2 >)];
    LibGeoDecomp::Streak<2 > *obj = (LibGeoDecomp::Streak<2 >*)fakeObject;

    const int count = 2;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->endX), lookup<int >(), 1),
        MemberSpec(getAddress(&obj->origin), lookup<Coord<2 > >(), 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_Streak_3_() {
    char fakeObject[sizeof(LibGeoDecomp::Streak<3 >)];
    LibGeoDecomp::Streak<3 > *obj = (LibGeoDecomp::Streak<3 >*)fakeObject;

    const int count = 2;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->endX), lookup<int >(), 1),
        MemberSpec(getAddress(&obj->origin), lookup<Coord<3 > >(), 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_StreakMPIDatatypeHelper() {
    char fakeObject[sizeof(LibGeoDecomp::StreakMPIDatatypeHelper)];
    LibGeoDecomp::StreakMPIDatatypeHelper *obj = (LibGeoDecomp::StreakMPIDatatypeHelper*)fakeObject;

    const int count = 3;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->a), lookup<Streak<1 > >(), 1),
        MemberSpec(getAddress(&obj->b), lookup<Streak<2 > >(), 1),
        MemberSpec(getAddress(&obj->c), lookup<Streak<3 > >(), 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_TestCell_1_() {
    char fakeObject[sizeof(LibGeoDecomp::TestCell<1 >)];
    LibGeoDecomp::TestCell<1 > *obj = (LibGeoDecomp::TestCell<1 >*)fakeObject;

    const int count = 6;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->cycleCounter), lookup<unsigned >(), 1),
        MemberSpec(getAddress(&obj->dimensions), lookup<CoordBox<1 > >(), 1),
        MemberSpec(getAddress(&obj->isEdgeCell), lookup<bool >(), 1),
        MemberSpec(getAddress(&obj->isValid), lookup<bool >(), 1),
        MemberSpec(getAddress(&obj->pos), lookup<Coord<1 > >(), 1),
        MemberSpec(getAddress(&obj->testValue), lookup<double >(), 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_TestCell_2_() {
    char fakeObject[sizeof(LibGeoDecomp::TestCell<2 >)];
    LibGeoDecomp::TestCell<2 > *obj = (LibGeoDecomp::TestCell<2 >*)fakeObject;

    const int count = 6;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->cycleCounter), lookup<unsigned >(), 1),
        MemberSpec(getAddress(&obj->dimensions), lookup<CoordBox<2 > >(), 1),
        MemberSpec(getAddress(&obj->isEdgeCell), lookup<bool >(), 1),
        MemberSpec(getAddress(&obj->isValid), lookup<bool >(), 1),
        MemberSpec(getAddress(&obj->pos), lookup<Coord<2 > >(), 1),
        MemberSpec(getAddress(&obj->testValue), lookup<double >(), 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_TestCell_3_() {
    char fakeObject[sizeof(LibGeoDecomp::TestCell<3 >)];
    LibGeoDecomp::TestCell<3 > *obj = (LibGeoDecomp::TestCell<3 >*)fakeObject;

    const int count = 6;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->cycleCounter), lookup<unsigned >(), 1),
        MemberSpec(getAddress(&obj->dimensions), lookup<CoordBox<3 > >(), 1),
        MemberSpec(getAddress(&obj->isEdgeCell), lookup<bool >(), 1),
        MemberSpec(getAddress(&obj->isValid), lookup<bool >(), 1),
        MemberSpec(getAddress(&obj->pos), lookup<Coord<3 > >(), 1),
        MemberSpec(getAddress(&obj->testValue), lookup<double >(), 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_TestCellMPIDatatypeHelper() {
    char fakeObject[sizeof(LibGeoDecomp::TestCellMPIDatatypeHelper)];
    LibGeoDecomp::TestCellMPIDatatypeHelper *obj = (LibGeoDecomp::TestCellMPIDatatypeHelper*)fakeObject;

    const int count = 3;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->a), lookup<TestCell<1 > >(), 1),
        MemberSpec(getAddress(&obj->b), lookup<TestCell<2 > >(), 1),
        MemberSpec(getAddress(&obj->c), lookup<TestCell<3 > >(), 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_UnstructuredTestCell_UnstructuredTestCellHelpers_EmptyAPI_() {
    char fakeObject[sizeof(LibGeoDecomp::UnstructuredTestCell<UnstructuredTestCellHelpers::EmptyAPI >)];
    LibGeoDecomp::UnstructuredTestCell<UnstructuredTestCellHelpers::EmptyAPI > *obj = (LibGeoDecomp::UnstructuredTestCell<UnstructuredTestCellHelpers::EmptyAPI >*)fakeObject;

    const int count = 5;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->cycleCounter), lookup<unsigned >(), 1),
        MemberSpec(getAddress(&obj->expectedNeighborWeights), lookup<FixedArray<double,100 > >(), 1),
        MemberSpec(getAddress(&obj->id), lookup<int >(), 1),
        MemberSpec(getAddress(&obj->isEdgeCell), lookup<bool >(), 1),
        MemberSpec(getAddress(&obj->isValid), lookup<bool >(), 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_UnstructuredTestCellMPIDatatypeHelper() {
    char fakeObject[sizeof(LibGeoDecomp::UnstructuredTestCellMPIDatatypeHelper)];
    LibGeoDecomp::UnstructuredTestCellMPIDatatypeHelper *obj = (LibGeoDecomp::UnstructuredTestCellMPIDatatypeHelper*)fakeObject;

    const int count = 1;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->a), lookup<UnstructuredTestCell<UnstructuredTestCellHelpers::EmptyAPI > >(), 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}

MPI_Datatype
Typemaps::generateMapLibGeoDecomp_Chronometer() {
    char fakeObject[sizeof(LibGeoDecomp::Chronometer)];
    LibGeoDecomp::Chronometer *obj = (LibGeoDecomp::Chronometer*)fakeObject;

    const int count = 1;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->totalTimes), lookup<FixedArray<double,Chronometer::NUM_INTERVALS > >(), 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI_Aint displacements[count];
    MPI_Datatype memberTypes[count];
    for (int i = 0; i < count; i++) {
        displacements[i] = rawSpecs[i].address;
        memberTypes[i] = rawSpecs[i].type;
        lengths[i] = rawSpecs[i].length;
    }

    // transform absolute addresses into offsets
    for (int i = count-1; i > 0; i--) {
        displacements[i] -= displacements[0];
    }
    displacements[0] = 0;

    // create datatype
    MPI_Datatype objType;
    MPI_Type_create_struct(count, lengths, displacements, memberTypes, &objType);
    MPI_Type_commit(&objType);

    return objType;
}


void Typemaps::initializeMaps()
{
    if (mapsCreated) {
        throw std::logic_error("Typemaps already initialized, duplicate initialization would leak memory");
    }

    if (sizeof(std::size_t) != sizeof(unsigned long)) {
        throw std::logic_error("MPI_UNSIGNED_LONG not suited for communication of std::size_t, needs to be redefined");
    }

    int mpiInitState = 0;
    MPI_Initialized(&mpiInitState);
    if (!mpiInitState) {
        throw std::logic_error("MPI needs to be initialized prior to setting up Typemaps");
    }

    MPI_LIBGEODECOMP_COORD_1_ = generateMapLibGeoDecomp_Coord_1_();
    MPI_LIBGEODECOMP_COORD_2_ = generateMapLibGeoDecomp_Coord_2_();
    MPI_LIBGEODECOMP_COORD_3_ = generateMapLibGeoDecomp_Coord_3_();
    MPI_LIBGEODECOMP_COORDBOX_1_ = generateMapLibGeoDecomp_CoordBox_1_();
    MPI_LIBGEODECOMP_COORDBOX_2_ = generateMapLibGeoDecomp_CoordBox_2_();
    MPI_LIBGEODECOMP_COORDBOX_3_ = generateMapLibGeoDecomp_CoordBox_3_();
    MPI_LIBGEODECOMP_COORDBOXMPIDATATYPEHELPER = generateMapLibGeoDecomp_CoordBoxMPIDatatypeHelper();
    MPI_LIBGEODECOMP_FIXEDARRAY_DOUBLE_100_ = generateMapLibGeoDecomp_FixedArray_double_100_();
    MPI_LIBGEODECOMP_FIXEDARRAY_DOUBLE_CHRONOMETER_NUM_INTERVALS_ = generateMapLibGeoDecomp_FixedArray_double_Chronometer_NUM_INTERVALS_();
    MPI_LIBGEODECOMP_FIXEDARRAY_INT_100_ = generateMapLibGeoDecomp_FixedArray_int_100_();
    MPI_LIBGEODECOMP_FLOATCOORD_1_ = generateMapLibGeoDecomp_FloatCoord_1_();
    MPI_LIBGEODECOMP_FLOATCOORD_2_ = generateMapLibGeoDecomp_FloatCoord_2_();
    MPI_LIBGEODECOMP_FLOATCOORD_3_ = generateMapLibGeoDecomp_FloatCoord_3_();
    MPI_LIBGEODECOMP_FLOATCOORDMPIDATATYPEHELPER = generateMapLibGeoDecomp_FloatCoordMPIDatatypeHelper();
    MPI_LIBGEODECOMP_STREAK_1_ = generateMapLibGeoDecomp_Streak_1_();
    MPI_LIBGEODECOMP_STREAK_2_ = generateMapLibGeoDecomp_Streak_2_();
    MPI_LIBGEODECOMP_STREAK_3_ = generateMapLibGeoDecomp_Streak_3_();
    MPI_LIBGEODECOMP_STREAKMPIDATATYPEHELPER = generateMapLibGeoDecomp_StreakMPIDatatypeHelper();
    MPI_LIBGEODECOMP_TESTCELL_1_ = generateMapLibGeoDecomp_TestCell_1_();
    MPI_LIBGEODECOMP_TESTCELL_2_ = generateMapLibGeoDecomp_TestCell_2_();
    MPI_LIBGEODECOMP_TESTCELL_3_ = generateMapLibGeoDecomp_TestCell_3_();
    MPI_LIBGEODECOMP_TESTCELLMPIDATATYPEHELPER = generateMapLibGeoDecomp_TestCellMPIDatatypeHelper();
    MPI_LIBGEODECOMP_UNSTRUCTUREDTESTCELL_UNSTRUCTUREDTESTCELLHELPERS_EMPTYAPI_ = generateMapLibGeoDecomp_UnstructuredTestCell_UnstructuredTestCellHelpers_EmptyAPI_();
    MPI_LIBGEODECOMP_UNSTRUCTUREDTESTCELLMPIDATATYPEHELPER = generateMapLibGeoDecomp_UnstructuredTestCellMPIDatatypeHelper();
    MPI_LIBGEODECOMP_CHRONOMETER = generateMapLibGeoDecomp_Chronometer();

    mapsCreated = true;
}

bool Typemaps::mapsCreated = false;

}

#endif
