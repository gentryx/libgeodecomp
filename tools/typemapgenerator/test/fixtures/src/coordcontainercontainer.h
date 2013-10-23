#ifndef _coordcontainercontainer_
#define _coordcontainercontainer__

include "coordcontainer.h"

class CoordContainerContainer
{
    friend class Typemaps;

    CoordContainer<2> cargo2;
    CoordContainer<3> cargo3;
    CoordContainer<4> cargo4;
};

#endif
