#ifndef _coordcontainer_
#define _coordcontainer_

include "coord.h"

template<int DIMENSIONS>
class CoordContainer
{
    friend class Typemaps;

    Coord<DIMENSIONS> pos;
};

#endif
