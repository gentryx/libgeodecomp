#include <algorithm>
#include "typemaps.h"

namespace MPI \{
    Datatype TIRE;
    Datatype RIM;
    Datatype WHEEL;
\}

// Member Specification, holds all relevant information for a given member.
class MemberSpec \{
public\:
    MemberSpec\(MPI\:\:Aint _address, MPI\:\:Datatype _type, int _length\) \{
        address = _address;
        type = _type;
        length = _length;
    \}

    MPI\:\:Aint address;
    MPI\:\:Datatype type;
    int length;
\};

bool addressLower\(MemberSpec a, MemberSpec b\)
\{
    return a.address < b.address;
\}

MPI\:\:Datatype
Typemaps\:\:generateMapTire\(\) \{.*
\}

MPI\:\:Datatype
Typemaps\:\:generateMapRim\(\) \{.*
\}

MPI\:\:Datatype
Typemaps\:\:generateMapWheel\(\) \{.*
\}

void Typemaps\:\:initializeMaps\(\)
\{
    MPI\:\:TIRE = generateMapTire\(\);
    MPI\:\:RIM = generateMapRim\(\);
    MPI\:\:WHEEL = generateMapWheel\(\);
\}
