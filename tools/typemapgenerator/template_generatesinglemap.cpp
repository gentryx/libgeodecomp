MPI_Datatype
Typemaps::generateMapKLASS_NAME() {
    char fakeObject[sizeof(KLASS)];
    KLASS *obj = (KLASS*)fakeObject;

    const int count = NUM_MEMBERS;
    int lengths[count];

    // sort addresses in ascending order
    MemberSpec rawSpecs[] = {
        MEMBERSPECS
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
