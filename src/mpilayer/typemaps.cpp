#include<libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#include "typemaps.h"
#include <algorithm>

MPI_Datatype MPI_LIBGEODECOMP_COORD_1_;
MPI_Datatype MPI_LIBGEODECOMP_COORD_2_;
MPI_Datatype MPI_LIBGEODECOMP_COORD_3_;
MPI_Datatype MPI_LIBGEODECOMP_COORDBOX_1_;
MPI_Datatype MPI_LIBGEODECOMP_COORDBOX_2_;
MPI_Datatype MPI_LIBGEODECOMP_COORDBOX_3_;
MPI_Datatype MPI_LIBGEODECOMP_COORDBOXMPIDATATYPEHELPER;
MPI_Datatype MPI_LIBGEODECOMP_FIXEDARRAY_DOUBLE_CHRONOMETER_NUM_INTERVALS_;
MPI_Datatype MPI_LIBGEODECOMP_FLOATCOORDBASE_1_;
MPI_Datatype MPI_LIBGEODECOMP_FLOATCOORDBASE_2_;
MPI_Datatype MPI_LIBGEODECOMP_FLOATCOORDBASE_3_;
MPI_Datatype MPI_LIBGEODECOMP_FLOATCOORDBASEMPIDATATYPEHELPER;
MPI_Datatype MPI_LIBGEODECOMP_MYSIMPLECELL;
MPI_Datatype MPI_LIBGEODECOMP_STATISTICS;
MPI_Datatype MPI_LIBGEODECOMP_STREAK_1_;
MPI_Datatype MPI_LIBGEODECOMP_STREAK_2_;
MPI_Datatype MPI_LIBGEODECOMP_STREAK_3_;
MPI_Datatype MPI_LIBGEODECOMP_STREAKMPIDATATYPEHELPER;
MPI_Datatype MPI_LIBGEODECOMP_TESTCELL_1_;
MPI_Datatype MPI_LIBGEODECOMP_TESTCELL_2_;
MPI_Datatype MPI_LIBGEODECOMP_TESTCELL_3_;
MPI_Datatype MPI_LIBGEODECOMP_TESTCELLMPIDATATYPEHELPER;
MPI_Datatype MPI_LIBGEODECOMP_CHRONOMETER;
MPI_Datatype MPI_LIBGEODECOMP_FLOATCOORD_1_;
MPI_Datatype MPI_LIBGEODECOMP_FLOATCOORD_2_;
MPI_Datatype MPI_LIBGEODECOMP_FLOATCOORD_3_;
MPI_Datatype MPI_LIBGEODECOMP_FLOATCOORDMPIDATATYPEHELPER;

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
        MemberSpec(getAddress(&obj->c), MPI_INT, 1)
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
        MemberSpec(getAddress(&obj->c), MPI_INT, 2)
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
        MemberSpec(getAddress(&obj->c), MPI_INT, 3)
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
        MemberSpec(getAddress(&obj->dimensions), MPI_LIBGEODECOMP_COORD_1_, 1),
        MemberSpec(getAddress(&obj->origin), MPI_LIBGEODECOMP_COORD_1_, 1)
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
        MemberSpec(getAddress(&obj->dimensions), MPI_LIBGEODECOMP_COORD_2_, 1),
        MemberSpec(getAddress(&obj->origin), MPI_LIBGEODECOMP_COORD_2_, 1)
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
        MemberSpec(getAddress(&obj->dimensions), MPI_LIBGEODECOMP_COORD_3_, 1),
        MemberSpec(getAddress(&obj->origin), MPI_LIBGEODECOMP_COORD_3_, 1)
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
        MemberSpec(getAddress(&obj->a), MPI_LIBGEODECOMP_COORDBOX_1_, 1),
        MemberSpec(getAddress(&obj->b), MPI_LIBGEODECOMP_COORDBOX_2_, 1),
        MemberSpec(getAddress(&obj->c), MPI_LIBGEODECOMP_COORDBOX_3_, 1)
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
        MemberSpec(getAddress(&obj->elements), MPI_INT, 1),
        MemberSpec(getAddress(&obj->store), MPI_DOUBLE, Chronometer::NUM_INTERVALS)
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
Typemaps::generateMapLibGeoDecomp_FloatCoordBase_1_() {
    char fakeObject[sizeof(LibGeoDecomp::FloatCoordBase<1 >)];
    LibGeoDecomp::FloatCoordBase<1 > *obj = (LibGeoDecomp::FloatCoordBase<1 >*)fakeObject;

    const int count = 1;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->c), MPI_DOUBLE, 1)
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
Typemaps::generateMapLibGeoDecomp_FloatCoordBase_2_() {
    char fakeObject[sizeof(LibGeoDecomp::FloatCoordBase<2 >)];
    LibGeoDecomp::FloatCoordBase<2 > *obj = (LibGeoDecomp::FloatCoordBase<2 >*)fakeObject;

    const int count = 1;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->c), MPI_DOUBLE, 2)
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
Typemaps::generateMapLibGeoDecomp_FloatCoordBase_3_() {
    char fakeObject[sizeof(LibGeoDecomp::FloatCoordBase<3 >)];
    LibGeoDecomp::FloatCoordBase<3 > *obj = (LibGeoDecomp::FloatCoordBase<3 >*)fakeObject;

    const int count = 1;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->c), MPI_DOUBLE, 3)
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
Typemaps::generateMapLibGeoDecomp_FloatCoordBaseMPIDatatypeHelper() {
    char fakeObject[sizeof(LibGeoDecomp::FloatCoordBaseMPIDatatypeHelper)];
    LibGeoDecomp::FloatCoordBaseMPIDatatypeHelper *obj = (LibGeoDecomp::FloatCoordBaseMPIDatatypeHelper*)fakeObject;

    const int count = 3;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->a), MPI_LIBGEODECOMP_FLOATCOORDBASE_1_, 1),
        MemberSpec(getAddress(&obj->b), MPI_LIBGEODECOMP_FLOATCOORDBASE_2_, 1),
        MemberSpec(getAddress(&obj->c), MPI_LIBGEODECOMP_FLOATCOORDBASE_3_, 1)
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
Typemaps::generateMapLibGeoDecomp_MySimpleCell() {
    char fakeObject[sizeof(LibGeoDecomp::MySimpleCell)];
    LibGeoDecomp::MySimpleCell *obj = (LibGeoDecomp::MySimpleCell*)fakeObject;

    const int count = 1;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->temp), MPI_DOUBLE, 1)
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
Typemaps::generateMapLibGeoDecomp_Statistics() {
    char fakeObject[sizeof(LibGeoDecomp::Statistics)];
    LibGeoDecomp::Statistics *obj = (LibGeoDecomp::Statistics*)fakeObject;

    const int count = 5;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(getAddress(&obj->computeTimeGhost), MPI_DOUBLE, 1),
        MemberSpec(getAddress(&obj->computeTimeInner), MPI_DOUBLE, 1),
        MemberSpec(getAddress(&obj->patchAcceptersTime), MPI_DOUBLE, 1),
        MemberSpec(getAddress(&obj->patchProvidersTime), MPI_DOUBLE, 1),
        MemberSpec(getAddress(&obj->totalTime), MPI_DOUBLE, 1)
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
        MemberSpec(getAddress(&obj->endX), MPI_INT, 1),
        MemberSpec(getAddress(&obj->origin), MPI_LIBGEODECOMP_COORD_1_, 1)
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
        MemberSpec(getAddress(&obj->endX), MPI_INT, 1),
        MemberSpec(getAddress(&obj->origin), MPI_LIBGEODECOMP_COORD_2_, 1)
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
        MemberSpec(getAddress(&obj->endX), MPI_INT, 1),
        MemberSpec(getAddress(&obj->origin), MPI_LIBGEODECOMP_COORD_3_, 1)
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
        MemberSpec(getAddress(&obj->a), MPI_LIBGEODECOMP_STREAK_1_, 1),
        MemberSpec(getAddress(&obj->b), MPI_LIBGEODECOMP_STREAK_2_, 1),
        MemberSpec(getAddress(&obj->c), MPI_LIBGEODECOMP_STREAK_3_, 1)
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
        MemberSpec(getAddress(&obj->cycleCounter), MPI_UNSIGNED, 1),
        MemberSpec(getAddress(&obj->dimensions), MPI_LIBGEODECOMP_COORDBOX_1_, 1),
        MemberSpec(getAddress(&obj->isEdgeCell), MPI_CHAR, 1),
        MemberSpec(getAddress(&obj->isValid), MPI_CHAR, 1),
        MemberSpec(getAddress(&obj->pos), MPI_LIBGEODECOMP_COORD_1_, 1),
        MemberSpec(getAddress(&obj->testValue), MPI_DOUBLE, 1)
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
        MemberSpec(getAddress(&obj->cycleCounter), MPI_UNSIGNED, 1),
        MemberSpec(getAddress(&obj->dimensions), MPI_LIBGEODECOMP_COORDBOX_2_, 1),
        MemberSpec(getAddress(&obj->isEdgeCell), MPI_CHAR, 1),
        MemberSpec(getAddress(&obj->isValid), MPI_CHAR, 1),
        MemberSpec(getAddress(&obj->pos), MPI_LIBGEODECOMP_COORD_2_, 1),
        MemberSpec(getAddress(&obj->testValue), MPI_DOUBLE, 1)
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
        MemberSpec(getAddress(&obj->cycleCounter), MPI_UNSIGNED, 1),
        MemberSpec(getAddress(&obj->dimensions), MPI_LIBGEODECOMP_COORDBOX_3_, 1),
        MemberSpec(getAddress(&obj->isEdgeCell), MPI_CHAR, 1),
        MemberSpec(getAddress(&obj->isValid), MPI_CHAR, 1),
        MemberSpec(getAddress(&obj->pos), MPI_LIBGEODECOMP_COORD_3_, 1),
        MemberSpec(getAddress(&obj->testValue), MPI_DOUBLE, 1)
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
        MemberSpec(getAddress(&obj->a), MPI_LIBGEODECOMP_TESTCELL_1_, 1),
        MemberSpec(getAddress(&obj->b), MPI_LIBGEODECOMP_TESTCELL_2_, 1),
        MemberSpec(getAddress(&obj->c), MPI_LIBGEODECOMP_TESTCELL_3_, 1)
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
        MemberSpec(getAddress(&obj->totalTimes), MPI_LIBGEODECOMP_FIXEDARRAY_DOUBLE_CHRONOMETER_NUM_INTERVALS_, 1)
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
        MemberSpec(getAddress((LibGeoDecomp::FloatCoordBase<1 >*)obj), MPI_LIBGEODECOMP_FLOATCOORDBASE_1_, 1)
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
        MemberSpec(getAddress((LibGeoDecomp::FloatCoordBase<2 >*)obj), MPI_LIBGEODECOMP_FLOATCOORDBASE_2_, 1)
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
        MemberSpec(getAddress((LibGeoDecomp::FloatCoordBase<3 >*)obj), MPI_LIBGEODECOMP_FLOATCOORDBASE_3_, 1)
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
        MemberSpec(getAddress(&obj->a), MPI_LIBGEODECOMP_FLOATCOORD_1_, 1),
        MemberSpec(getAddress(&obj->b), MPI_LIBGEODECOMP_FLOATCOORD_2_, 1),
        MemberSpec(getAddress(&obj->c), MPI_LIBGEODECOMP_FLOATCOORD_3_, 1)
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
    MPI_LIBGEODECOMP_COORD_1_ = generateMapLibGeoDecomp_Coord_1_();
    MPI_LIBGEODECOMP_COORD_2_ = generateMapLibGeoDecomp_Coord_2_();
    MPI_LIBGEODECOMP_COORD_3_ = generateMapLibGeoDecomp_Coord_3_();
    MPI_LIBGEODECOMP_COORDBOX_1_ = generateMapLibGeoDecomp_CoordBox_1_();
    MPI_LIBGEODECOMP_COORDBOX_2_ = generateMapLibGeoDecomp_CoordBox_2_();
    MPI_LIBGEODECOMP_COORDBOX_3_ = generateMapLibGeoDecomp_CoordBox_3_();
    MPI_LIBGEODECOMP_COORDBOXMPIDATATYPEHELPER = generateMapLibGeoDecomp_CoordBoxMPIDatatypeHelper();
    MPI_LIBGEODECOMP_FIXEDARRAY_DOUBLE_CHRONOMETER_NUM_INTERVALS_ = generateMapLibGeoDecomp_FixedArray_double_Chronometer_NUM_INTERVALS_();
    MPI_LIBGEODECOMP_FLOATCOORDBASE_1_ = generateMapLibGeoDecomp_FloatCoordBase_1_();
    MPI_LIBGEODECOMP_FLOATCOORDBASE_2_ = generateMapLibGeoDecomp_FloatCoordBase_2_();
    MPI_LIBGEODECOMP_FLOATCOORDBASE_3_ = generateMapLibGeoDecomp_FloatCoordBase_3_();
    MPI_LIBGEODECOMP_FLOATCOORDBASEMPIDATATYPEHELPER = generateMapLibGeoDecomp_FloatCoordBaseMPIDatatypeHelper();
    MPI_LIBGEODECOMP_MYSIMPLECELL = generateMapLibGeoDecomp_MySimpleCell();
    MPI_LIBGEODECOMP_STATISTICS = generateMapLibGeoDecomp_Statistics();
    MPI_LIBGEODECOMP_STREAK_1_ = generateMapLibGeoDecomp_Streak_1_();
    MPI_LIBGEODECOMP_STREAK_2_ = generateMapLibGeoDecomp_Streak_2_();
    MPI_LIBGEODECOMP_STREAK_3_ = generateMapLibGeoDecomp_Streak_3_();
    MPI_LIBGEODECOMP_STREAKMPIDATATYPEHELPER = generateMapLibGeoDecomp_StreakMPIDatatypeHelper();
    MPI_LIBGEODECOMP_TESTCELL_1_ = generateMapLibGeoDecomp_TestCell_1_();
    MPI_LIBGEODECOMP_TESTCELL_2_ = generateMapLibGeoDecomp_TestCell_2_();
    MPI_LIBGEODECOMP_TESTCELL_3_ = generateMapLibGeoDecomp_TestCell_3_();
    MPI_LIBGEODECOMP_TESTCELLMPIDATATYPEHELPER = generateMapLibGeoDecomp_TestCellMPIDatatypeHelper();
    MPI_LIBGEODECOMP_CHRONOMETER = generateMapLibGeoDecomp_Chronometer();
    MPI_LIBGEODECOMP_FLOATCOORD_1_ = generateMapLibGeoDecomp_FloatCoord_1_();
    MPI_LIBGEODECOMP_FLOATCOORD_2_ = generateMapLibGeoDecomp_FloatCoord_2_();
    MPI_LIBGEODECOMP_FLOATCOORD_3_ = generateMapLibGeoDecomp_FloatCoord_3_();
    MPI_LIBGEODECOMP_FLOATCOORDMPIDATATYPEHELPER = generateMapLibGeoDecomp_FloatCoordMPIDatatypeHelper();
}

}

#endif
