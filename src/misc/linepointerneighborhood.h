#ifndef _libgeodecomp_misc_linepointerneighborhood_h_
#define _libgeodecomp_misc_linepointerneighborhood_h_

#include <libgeodecomp/misc/fixedcoord.h>
#include <libgeodecomp/misc/stencils.h>

namespace LibGeoDecomp {

class LinePointerNeighborhoodHelper
{
public:
    /**
     * Empty dummy
     */
    template<typename CELL, class STENCIL, bool FLAG>
    class West
    {
    public:
        void access(West) const
        {}
    };

    template<typename CELL, class STENCIL>
    class West<CELL, STENCIL, true>
    {
    public:
        const CELL& access(FixedCoord<-1, -1, -1>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL, -1, -1, -1>::VALUE][0];
        }

        const CELL& access(FixedCoord<-1,  0, -1>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL, -1,  0, -1>::VALUE][0];
        }

        const CELL& access(FixedCoord<-1,  1, -1>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL, -1,  1, -1>::VALUE][0];
        }

        const CELL& access(FixedCoord<-1, -1,  0>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL, -1, -1,  0>::VALUE][0];
        }

        const CELL& access(FixedCoord<-1,  0,  0>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL, -1,  0,  0>::VALUE][0];
        }

        const CELL& access(FixedCoord<-1,  1,  0>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL, -1,  1,  0>::VALUE][0];
        }

        const CELL& access(FixedCoord<-1, -1,  1>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL, -1, -1,  1>::VALUE][0];
        }

        const CELL& access(FixedCoord<-1,  0,  1>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL, -1,  0,  1>::VALUE][0];
        }

        const CELL& access(FixedCoord<-1,  1,  1>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL, -1,  1,  1>::VALUE][0];
        }
    };

    /**
     * Empty dummy
     */
    template<typename CELL, class STENCIL, bool FLAG>
    class East
    {
    public:
        void access(East) const
        {}
    };

    template<typename CELL, class STENCIL>
    class East<CELL, STENCIL, true>
    {
    public:
        const CELL& access(FixedCoord< 1, -1, -1>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL,  1, -1, -1>::VALUE][0];
        }

        const CELL& access(FixedCoord< 1,  0, -1>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL,  1,  0, -1>::VALUE][0];
        }

        const CELL& access(FixedCoord< 1,  1, -1>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL,  1,  1, -1>::VALUE][0];
        }

        const CELL& access(FixedCoord< 1, -1,  0>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL,  1, -1,  0>::VALUE][0];
        }

        const CELL& access(FixedCoord< 1,  0,  0>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL,  1,  0,  0>::VALUE][0];
        }

        const CELL& access(FixedCoord< 1,  1,  0>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL,  1,  1,  0>::VALUE][0];
        }

        const CELL& access(FixedCoord< 1, -1,  1>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL,  1, -1,  1>::VALUE][0];
        }

        const CELL& access(FixedCoord< 1,  0,  1>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL,  1,  0,  1>::VALUE][0];
        }

        const CELL& access(FixedCoord< 1,  1,  1>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL,  1,  1,  1>::VALUE][0];
        }
    };

    /**
     * Empty dummy
     */
    template<typename CELL, class STENCIL, bool FLAG>
    class Top
    {
    public:
        void access(Top) const
        {}
    };

    template<typename CELL, class STENCIL>
    class Top<CELL, STENCIL, true>
    {
    public:
        template<int X>
        const CELL& access(FixedCoord< X, -1, -1>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL,  0, -1, -1>::VALUE][0];
        }

        template<int X>
        const CELL& access(FixedCoord< X, -1,  0>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL,  0, -1,  0>::VALUE][0];
        }

        template<int X>
        const CELL& access(FixedCoord< X, -1,  1>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL,  0, -1,  1>::VALUE][0];
        }
    };

    /**
     * Empty dummy
     */
    template<typename CELL, class STENCIL, bool FLAG>
    class Bottom
    {
    public:
        void access(Bottom) const
        {}
    };

    template<typename CELL, class STENCIL>
    class Bottom<CELL, STENCIL, true>
    {
    public:
        template<int X>
        const CELL& access(FixedCoord< X,  1, -1>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL,  0,  1, -1>::VALUE][0];
        }

        template<int X>
        const CELL& access(FixedCoord< X,  1,  0>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL,  0,  1,  0>::VALUE][0];
        }

        template<int X>
        const CELL& access(FixedCoord< X,  1,  1>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL,  0,  1,  1>::VALUE][0];
        }
    };

    /**
     * Empty dummy
     */
    template<typename CELL, class STENCIL, bool FLAG>
    class North
    {
    public:
        void access(North) const
        {}
    };

    template<typename CELL, class STENCIL>
    class North<CELL, STENCIL, true>
    {
    public:
        template<int X, int Y>
        const CELL& access(FixedCoord< X,  Y,  1>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL,  0,  Y,  1>::VALUE][0];
        }
    };

    /**
     * Empty dummy
     */
    template<typename CELL, class STENCIL, bool FLAG>
    class South
    {
    public:
        void access(South) const
        {}
    };

    template<typename CELL, class STENCIL>
    class South<CELL, STENCIL, true>
    {
    public:
        template<int X, int Y>
        const CELL& access(FixedCoord< X,  Y, -1>, CELL **pointers) const
        {
            return pointers[Stencils::OffsetHelper<STENCIL,  0,  Y, -1>::VALUE][0];
        }
    };
};

/**
 * gives cells access to their neighboring cells in a given stencil
 * shape. It is meant as a low-overhead replacement for CoordMap. The
 * flags are used in template specializations to cover those cases in
 * which accesses to neighboring cells need to be rerouted. In
 * essence, we go through the pain of having such many parameters to
 * resolve runtime conditionals for range checking at compile time.
 *
 * fixme:
 * BOUNDARY_WEST:
 *   X == -1 -> use WEST pointers
 =   X >=  0 -> use center pointers + X
 * 
 * BOUNDARY_EAST:
 *   X == 1 -> use EAST pointers
 *   X <= 0 -> use center pointers + X
 *
 * BOUNDARY_TOP:
 *   Y == -1 -> use TOP pointer (don't set this on a torus, as its not needed!)
 *   Y >=  0 -> use center pointers + X
 * 
 */
template<typename CELL, class STENCIL, bool BOUNDARY_WEST, bool BOUNDARY_EAST, bool BOUNDARY_TOP, bool BOUNDARY_BOTTOM, bool BOUNDARY_SOUTH, bool BOUNDARY_NORTH>
class LinePointerNeighborhood :
    public LinePointerNeighborhoodHelper::West<  CELL, STENCIL, BOUNDARY_WEST>,
    public LinePointerNeighborhoodHelper::East<  CELL, STENCIL, BOUNDARY_EAST>,
    public LinePointerNeighborhoodHelper::Top<   CELL, STENCIL, BOUNDARY_TOP>,
    public LinePointerNeighborhoodHelper::Bottom<CELL, STENCIL, BOUNDARY_BOTTOM>,
    public LinePointerNeighborhoodHelper::North< CELL, STENCIL, BOUNDARY_NORTH>,
    public LinePointerNeighborhoodHelper::South< CELL, STENCIL, BOUNDARY_SOUTH>
{
public:
    using LinePointerNeighborhoodHelper::West<  CELL, STENCIL, BOUNDARY_WEST  >::access;
    using LinePointerNeighborhoodHelper::East<  CELL, STENCIL, BOUNDARY_EAST  >::access;
    using LinePointerNeighborhoodHelper::Top<   CELL, STENCIL, BOUNDARY_TOP   >::access;
    using LinePointerNeighborhoodHelper::Bottom<CELL, STENCIL, BOUNDARY_BOTTOM>::access;
    using LinePointerNeighborhoodHelper::North< CELL, STENCIL, BOUNDARY_NORTH >::access;
    using LinePointerNeighborhoodHelper::South< CELL, STENCIL, BOUNDARY_SOUTH >::access;

    LinePointerNeighborhood(CELL **_lines, long *_offset) :
        lines(_lines),
        offset(_offset)
    {}

    template<int X, int Y, int Z>
    const CELL& access(FixedCoord<X, Y, Z>, CELL **lines) const
    {
        return lines[Stencils::OffsetHelper<STENCIL, 0,  Y,  Z>::VALUE][X + *offset];
    }

    template<int X, int Y, int Z>
    const CELL& operator[](FixedCoord<X, Y, Z>) const
    {
        return access(FixedCoord<X, Y, Z>(), lines);
    }

private:
    CELL **lines;
    long *offset;
};

}

#endif
