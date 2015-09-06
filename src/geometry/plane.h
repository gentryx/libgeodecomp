#ifndef LIBGEODECOMP_GEOMETRY_PLANE_H
#define LIBGEODECOMP_GEOMETRY_PLANE_H

namespace LibGeoDecomp {

/**
 * This utility class ecapsulates the mathematical concept of a plane.
 * Its normal vector is given by "dir". Its main purpose is to check
 * on which side ponts lie. In that sense it can also be viewed as a
 * half-space.
 */
template<typename COORD, typename ID = int>
class Plane
{
public:
    Plane(const COORD& base, const COORD& dir, ID id = ID()) :
        base(base),
        dir(dir),
        neighborID(id),
        length(-1)
    {}

    /**
     * A Plane divides space into two halves. Points intersecting with
     * the plane are not considered to be on top.
     */
    bool isOnTop(const COORD& point) const
    {
        return (point - base) * dir > 0;
    }

    bool operator==(const Plane& other) const
    {
        // intentionally not including ID and length here, as we're
        // rather interested if both Planes refer to the same set
        // of coordinates:
        return
            (base == other.base) &&
            (dir  == other.dir);
    }

    COORD base;
    COORD dir;
    ID neighborID;
    double length;
};

template<typename _CharT, typename _Traits, typename COORD, typename ID>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& os,
           const Plane<COORD, ID>& e)
{
    os << "Plane(base=" << e.base << ", dir" << e.dir << ")";
    return os;
}

}

#endif
