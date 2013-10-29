#ifndef LIBGEODECOMP_GEOMETRY_FIXEDCOORD_H
#define LIBGEODECOMP_GEOMETRY_FIXEDCOORD_H

namespace LibGeoDecomp {

/**
 * is not meant to be used with actual instances, but rather to
 * bind template parameters in a convenient way. See the testbed and
 * unit tests for examples of how to use FixedCoord.
 */
template<int DIM_X=0, int DIM_Y=0, int DIM_Z=0>
class FixedCoord
{
public:
    static const int X = DIM_X;
    static const int Y = DIM_Y;
    static const int Z = DIM_Z;
};

}

#endif
