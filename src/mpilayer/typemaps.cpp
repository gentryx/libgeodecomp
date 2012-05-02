#include <algorithm>
#include "typemaps.h"

namespace MPI {
    Datatype LIBGEODECOMP_COORD_1_;
    Datatype LIBGEODECOMP_COORD_3_;
    Datatype LIBGEODECOMP_SIMPLECELL;
    Datatype LIBGEODECOMP_COORD_2_;
    Datatype LIBGEODECOMP_STREAK_1_;
    Datatype LIBGEODECOMP_STREAK_2_;
    Datatype LIBGEODECOMP_STREAK_3_;
    Datatype LIBGEODECOMP_COORDBOX_1_;
    Datatype LIBGEODECOMP_COORDBOX_2_;
    Datatype LIBGEODECOMP_COORDBOX_3_;
    Datatype LIBGEODECOMP_STREAKMPIDATATYPEHELPER;
    Datatype LIBGEODECOMP_COORDBOXMPIDATATYPEHELPER;
    Datatype LIBGEODECOMP_TESTCELL_1_;
    Datatype LIBGEODECOMP_TESTCELL_2_;
    Datatype LIBGEODECOMP_TESTCELL_3_;
    Datatype LIBGEODECOMP_TESTCELLMPIDATATYPEHELPER;
}

namespace LibGeoDecomp {
// Member Specification, holds all relevant information for a given member.
class MemberSpec {
public:
    MemberSpec(MPI::Aint _address, MPI::Datatype _type, int _length) {
        address = _address;
        type = _type;
        length = _length;
    }

    MPI::Aint address;
    MPI::Datatype type;
    int length;
};

bool addressLower(MemberSpec a, MemberSpec b)
{
    return a.address < b.address;
}

MPI::Datatype
Typemaps::generateMapLibGeoDecomp_Coord_1_() {
    char fakeObject[sizeof(LibGeoDecomp::Coord<1 >)];
    LibGeoDecomp::Coord<1 > *obj = (LibGeoDecomp::Coord<1 >*)fakeObject;

    int count = 1;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(MPI::Get_address(&obj->c), MPI::INT, 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI::Aint displacements[count];
    MPI::Datatype memberTypes[count];
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
    MPI::Datatype objType;
    objType = MPI::Datatype::Create_struct(count, lengths, displacements, memberTypes);
    objType.Commit();

    return objType;
}

MPI::Datatype
Typemaps::generateMapLibGeoDecomp_Coord_3_() {
    char fakeObject[sizeof(LibGeoDecomp::Coord<3 >)];
    LibGeoDecomp::Coord<3 > *obj = (LibGeoDecomp::Coord<3 >*)fakeObject;

    int count = 1;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(MPI::Get_address(&obj->c), MPI::INT, 3)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI::Aint displacements[count];
    MPI::Datatype memberTypes[count];
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
    MPI::Datatype objType;
    objType = MPI::Datatype::Create_struct(count, lengths, displacements, memberTypes);
    objType.Commit();

    return objType;
}

MPI::Datatype
Typemaps::generateMapLibGeoDecomp_SimpleCell() {
    char fakeObject[sizeof(LibGeoDecomp::SimpleCell)];
    LibGeoDecomp::SimpleCell *obj = (LibGeoDecomp::SimpleCell*)fakeObject;

    int count = 1;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(MPI::Get_address(&obj->val), MPI::DOUBLE, 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI::Aint displacements[count];
    MPI::Datatype memberTypes[count];
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
    MPI::Datatype objType;
    objType = MPI::Datatype::Create_struct(count, lengths, displacements, memberTypes);
    objType.Commit();

    return objType;
}

MPI::Datatype
Typemaps::generateMapLibGeoDecomp_Coord_2_() {
    char fakeObject[sizeof(LibGeoDecomp::Coord<2 >)];
    LibGeoDecomp::Coord<2 > *obj = (LibGeoDecomp::Coord<2 >*)fakeObject;

    int count = 1;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(MPI::Get_address(&obj->c), MPI::INT, 2)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI::Aint displacements[count];
    MPI::Datatype memberTypes[count];
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
    MPI::Datatype objType;
    objType = MPI::Datatype::Create_struct(count, lengths, displacements, memberTypes);
    objType.Commit();

    return objType;
}

MPI::Datatype
Typemaps::generateMapLibGeoDecomp_Streak_1_() {
    char fakeObject[sizeof(LibGeoDecomp::Streak<1 >)];
    LibGeoDecomp::Streak<1 > *obj = (LibGeoDecomp::Streak<1 >*)fakeObject;

    int count = 2;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(MPI::Get_address(&obj->endX), MPI::INT, 1),
        MemberSpec(MPI::Get_address(&obj->origin), MPI::LIBGEODECOMP_COORD_1_, 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI::Aint displacements[count];
    MPI::Datatype memberTypes[count];
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
    MPI::Datatype objType;
    objType = MPI::Datatype::Create_struct(count, lengths, displacements, memberTypes);
    objType.Commit();

    return objType;
}

MPI::Datatype
Typemaps::generateMapLibGeoDecomp_Streak_2_() {
    char fakeObject[sizeof(LibGeoDecomp::Streak<2 >)];
    LibGeoDecomp::Streak<2 > *obj = (LibGeoDecomp::Streak<2 >*)fakeObject;

    int count = 2;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(MPI::Get_address(&obj->endX), MPI::INT, 1),
        MemberSpec(MPI::Get_address(&obj->origin), MPI::LIBGEODECOMP_COORD_2_, 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI::Aint displacements[count];
    MPI::Datatype memberTypes[count];
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
    MPI::Datatype objType;
    objType = MPI::Datatype::Create_struct(count, lengths, displacements, memberTypes);
    objType.Commit();

    return objType;
}

MPI::Datatype
Typemaps::generateMapLibGeoDecomp_Streak_3_() {
    char fakeObject[sizeof(LibGeoDecomp::Streak<3 >)];
    LibGeoDecomp::Streak<3 > *obj = (LibGeoDecomp::Streak<3 >*)fakeObject;

    int count = 2;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(MPI::Get_address(&obj->endX), MPI::INT, 1),
        MemberSpec(MPI::Get_address(&obj->origin), MPI::LIBGEODECOMP_COORD_3_, 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI::Aint displacements[count];
    MPI::Datatype memberTypes[count];
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
    MPI::Datatype objType;
    objType = MPI::Datatype::Create_struct(count, lengths, displacements, memberTypes);
    objType.Commit();

    return objType;
}

MPI::Datatype
Typemaps::generateMapLibGeoDecomp_CoordBox_1_() {
    char fakeObject[sizeof(LibGeoDecomp::CoordBox<1 >)];
    LibGeoDecomp::CoordBox<1 > *obj = (LibGeoDecomp::CoordBox<1 >*)fakeObject;

    int count = 2;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(MPI::Get_address(&obj->dimensions), MPI::LIBGEODECOMP_COORD_1_, 1),
        MemberSpec(MPI::Get_address(&obj->origin), MPI::LIBGEODECOMP_COORD_1_, 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI::Aint displacements[count];
    MPI::Datatype memberTypes[count];
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
    MPI::Datatype objType;
    objType = MPI::Datatype::Create_struct(count, lengths, displacements, memberTypes);
    objType.Commit();

    return objType;
}

MPI::Datatype
Typemaps::generateMapLibGeoDecomp_CoordBox_2_() {
    char fakeObject[sizeof(LibGeoDecomp::CoordBox<2 >)];
    LibGeoDecomp::CoordBox<2 > *obj = (LibGeoDecomp::CoordBox<2 >*)fakeObject;

    int count = 2;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(MPI::Get_address(&obj->dimensions), MPI::LIBGEODECOMP_COORD_2_, 1),
        MemberSpec(MPI::Get_address(&obj->origin), MPI::LIBGEODECOMP_COORD_2_, 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI::Aint displacements[count];
    MPI::Datatype memberTypes[count];
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
    MPI::Datatype objType;
    objType = MPI::Datatype::Create_struct(count, lengths, displacements, memberTypes);
    objType.Commit();

    return objType;
}

MPI::Datatype
Typemaps::generateMapLibGeoDecomp_CoordBox_3_() {
    char fakeObject[sizeof(LibGeoDecomp::CoordBox<3 >)];
    LibGeoDecomp::CoordBox<3 > *obj = (LibGeoDecomp::CoordBox<3 >*)fakeObject;

    int count = 2;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(MPI::Get_address(&obj->dimensions), MPI::LIBGEODECOMP_COORD_3_, 1),
        MemberSpec(MPI::Get_address(&obj->origin), MPI::LIBGEODECOMP_COORD_3_, 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI::Aint displacements[count];
    MPI::Datatype memberTypes[count];
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
    MPI::Datatype objType;
    objType = MPI::Datatype::Create_struct(count, lengths, displacements, memberTypes);
    objType.Commit();

    return objType;
}

MPI::Datatype
Typemaps::generateMapLibGeoDecomp_StreakMPIDatatypeHelper() {
    char fakeObject[sizeof(LibGeoDecomp::StreakMPIDatatypeHelper)];
    LibGeoDecomp::StreakMPIDatatypeHelper *obj = (LibGeoDecomp::StreakMPIDatatypeHelper*)fakeObject;

    int count = 3;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(MPI::Get_address(&obj->a), MPI::LIBGEODECOMP_STREAK_1_, 1),
        MemberSpec(MPI::Get_address(&obj->b), MPI::LIBGEODECOMP_STREAK_2_, 1),
        MemberSpec(MPI::Get_address(&obj->c), MPI::LIBGEODECOMP_STREAK_3_, 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI::Aint displacements[count];
    MPI::Datatype memberTypes[count];
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
    MPI::Datatype objType;
    objType = MPI::Datatype::Create_struct(count, lengths, displacements, memberTypes);
    objType.Commit();

    return objType;
}

MPI::Datatype
Typemaps::generateMapLibGeoDecomp_CoordBoxMPIDatatypeHelper() {
    char fakeObject[sizeof(LibGeoDecomp::CoordBoxMPIDatatypeHelper)];
    LibGeoDecomp::CoordBoxMPIDatatypeHelper *obj = (LibGeoDecomp::CoordBoxMPIDatatypeHelper*)fakeObject;

    int count = 3;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(MPI::Get_address(&obj->a), MPI::LIBGEODECOMP_COORDBOX_1_, 1),
        MemberSpec(MPI::Get_address(&obj->b), MPI::LIBGEODECOMP_COORDBOX_2_, 1),
        MemberSpec(MPI::Get_address(&obj->c), MPI::LIBGEODECOMP_COORDBOX_3_, 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI::Aint displacements[count];
    MPI::Datatype memberTypes[count];
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
    MPI::Datatype objType;
    objType = MPI::Datatype::Create_struct(count, lengths, displacements, memberTypes);
    objType.Commit();

    return objType;
}

MPI::Datatype
Typemaps::generateMapLibGeoDecomp_TestCell_1_() {
    char fakeObject[sizeof(LibGeoDecomp::TestCell<1 >)];
    LibGeoDecomp::TestCell<1 > *obj = (LibGeoDecomp::TestCell<1 >*)fakeObject;

    int count = 6;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(MPI::Get_address(&obj->cycleCounter), MPI::UNSIGNED, 1),
        MemberSpec(MPI::Get_address(&obj->dimensions), MPI::LIBGEODECOMP_COORDBOX_1_, 1),
        MemberSpec(MPI::Get_address(&obj->isEdgeCell), MPI::BOOL, 1),
        MemberSpec(MPI::Get_address(&obj->isValid), MPI::BOOL, 1),
        MemberSpec(MPI::Get_address(&obj->pos), MPI::LIBGEODECOMP_COORD_1_, 1),
        MemberSpec(MPI::Get_address(&obj->testValue), MPI::DOUBLE, 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI::Aint displacements[count];
    MPI::Datatype memberTypes[count];
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
    MPI::Datatype objType;
    objType = MPI::Datatype::Create_struct(count, lengths, displacements, memberTypes);
    objType.Commit();

    return objType;
}

MPI::Datatype
Typemaps::generateMapLibGeoDecomp_TestCell_2_() {
    char fakeObject[sizeof(LibGeoDecomp::TestCell<2 >)];
    LibGeoDecomp::TestCell<2 > *obj = (LibGeoDecomp::TestCell<2 >*)fakeObject;

    int count = 6;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(MPI::Get_address(&obj->cycleCounter), MPI::UNSIGNED, 1),
        MemberSpec(MPI::Get_address(&obj->dimensions), MPI::LIBGEODECOMP_COORDBOX_2_, 1),
        MemberSpec(MPI::Get_address(&obj->isEdgeCell), MPI::BOOL, 1),
        MemberSpec(MPI::Get_address(&obj->isValid), MPI::BOOL, 1),
        MemberSpec(MPI::Get_address(&obj->pos), MPI::LIBGEODECOMP_COORD_2_, 1),
        MemberSpec(MPI::Get_address(&obj->testValue), MPI::DOUBLE, 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI::Aint displacements[count];
    MPI::Datatype memberTypes[count];
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
    MPI::Datatype objType;
    objType = MPI::Datatype::Create_struct(count, lengths, displacements, memberTypes);
    objType.Commit();

    return objType;
}

MPI::Datatype
Typemaps::generateMapLibGeoDecomp_TestCell_3_() {
    char fakeObject[sizeof(LibGeoDecomp::TestCell<3 >)];
    LibGeoDecomp::TestCell<3 > *obj = (LibGeoDecomp::TestCell<3 >*)fakeObject;

    int count = 6;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(MPI::Get_address(&obj->cycleCounter), MPI::UNSIGNED, 1),
        MemberSpec(MPI::Get_address(&obj->dimensions), MPI::LIBGEODECOMP_COORDBOX_3_, 1),
        MemberSpec(MPI::Get_address(&obj->isEdgeCell), MPI::BOOL, 1),
        MemberSpec(MPI::Get_address(&obj->isValid), MPI::BOOL, 1),
        MemberSpec(MPI::Get_address(&obj->pos), MPI::LIBGEODECOMP_COORD_3_, 1),
        MemberSpec(MPI::Get_address(&obj->testValue), MPI::DOUBLE, 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI::Aint displacements[count];
    MPI::Datatype memberTypes[count];
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
    MPI::Datatype objType;
    objType = MPI::Datatype::Create_struct(count, lengths, displacements, memberTypes);
    objType.Commit();

    return objType;
}

MPI::Datatype
Typemaps::generateMapLibGeoDecomp_TestCellMPIDatatypeHelper() {
    char fakeObject[sizeof(LibGeoDecomp::TestCellMPIDatatypeHelper)];
    LibGeoDecomp::TestCellMPIDatatypeHelper *obj = (LibGeoDecomp::TestCellMPIDatatypeHelper*)fakeObject;

    int count = 3;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(MPI::Get_address(&obj->a), MPI::LIBGEODECOMP_TESTCELL_1_, 1),
        MemberSpec(MPI::Get_address(&obj->b), MPI::LIBGEODECOMP_TESTCELL_2_, 1),
        MemberSpec(MPI::Get_address(&obj->c), MPI::LIBGEODECOMP_TESTCELL_3_, 1)
    };
    std::sort(rawSpecs, rawSpecs + count, addressLower);

    // split addresses from member types
    MPI::Aint displacements[count];
    MPI::Datatype memberTypes[count];
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
    MPI::Datatype objType;
    objType = MPI::Datatype::Create_struct(count, lengths, displacements, memberTypes);
    objType.Commit();

    return objType;
}

void Typemaps::initializeMaps()
{
    MPI::LIBGEODECOMP_COORD_1_ = generateMapLibGeoDecomp_Coord_1_();
    MPI::LIBGEODECOMP_COORD_3_ = generateMapLibGeoDecomp_Coord_3_();
    MPI::LIBGEODECOMP_SIMPLECELL = generateMapLibGeoDecomp_SimpleCell();
    MPI::LIBGEODECOMP_COORD_2_ = generateMapLibGeoDecomp_Coord_2_();
    MPI::LIBGEODECOMP_STREAK_1_ = generateMapLibGeoDecomp_Streak_1_();
    MPI::LIBGEODECOMP_STREAK_2_ = generateMapLibGeoDecomp_Streak_2_();
    MPI::LIBGEODECOMP_STREAK_3_ = generateMapLibGeoDecomp_Streak_3_();
    MPI::LIBGEODECOMP_COORDBOX_1_ = generateMapLibGeoDecomp_CoordBox_1_();
    MPI::LIBGEODECOMP_COORDBOX_2_ = generateMapLibGeoDecomp_CoordBox_2_();
    MPI::LIBGEODECOMP_COORDBOX_3_ = generateMapLibGeoDecomp_CoordBox_3_();
    MPI::LIBGEODECOMP_STREAKMPIDATATYPEHELPER = generateMapLibGeoDecomp_StreakMPIDatatypeHelper();
    MPI::LIBGEODECOMP_COORDBOXMPIDATATYPEHELPER = generateMapLibGeoDecomp_CoordBoxMPIDatatypeHelper();
    MPI::LIBGEODECOMP_TESTCELL_1_ = generateMapLibGeoDecomp_TestCell_1_();
    MPI::LIBGEODECOMP_TESTCELL_2_ = generateMapLibGeoDecomp_TestCell_2_();
    MPI::LIBGEODECOMP_TESTCELL_3_ = generateMapLibGeoDecomp_TestCell_3_();
    MPI::LIBGEODECOMP_TESTCELLMPIDATATYPEHELPER = generateMapLibGeoDecomp_TestCellMPIDatatypeHelper();
}
};
