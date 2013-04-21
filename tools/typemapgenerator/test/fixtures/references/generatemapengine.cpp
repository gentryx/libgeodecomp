MPI::Datatype
Typemaps::generateMapEngine() {
    char fakeObject[sizeof(Engine)];
    Engine *obj = (Engine*)fakeObject;

    const int count = 3;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MemberSpec(MPI::Get_address(&obj->capacity), MPI::DOUBLE, 1),
        MemberSpec(MPI::Get_address(&obj->fuel), MPI::INT, 1),
        MemberSpec(MPI::Get_address(&obj->gearRatios), MPI::DOUBLE, 6)
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
