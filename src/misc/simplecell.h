#ifndef _libgeodecomp_misc_simplecell_h_
#define _libgeodecomp_misc_simplecell_h_

#include <libgeodecomp/misc/coordmap.h>
#include <libgeodecomp/misc/typetraits.h>

namespace LibGeoDecomp {

// fixme: remove
class SimpleCell
{
    friend class Typemaps;
public:
    typedef Topologies::Cube<2>::Topology Topology;

    static inline unsigned nanoSteps() { return 27; }

    inline SimpleCell(const float& _val = 0) :
        val(_val)
    {}

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned&) 
    {
        val = (neighborhood[Coord<2>( 0, -1)].val +
               neighborhood[Coord<2>(-1,  0)].val + 
               neighborhood[Coord<2>( 0,  0)].val + 
               neighborhood[Coord<2>(+1,  0)].val +
               neighborhood[Coord<2>( 0, +1)].val) * 0.2;
    }

    inline void update(const SimpleCell& up, const SimpleCell& left, const SimpleCell& self, const SimpleCell& right, const SimpleCell& down, const unsigned&)
    {
        val = (up.val + left.val + self.val + right.val + down.val) * 0.2;
    }

    inline void update(const SimpleCell& /*ul*/, const SimpleCell& up, const SimpleCell& /*ur*/, const SimpleCell& left, const SimpleCell& self, const SimpleCell& right, const SimpleCell& /*ll*/, const SimpleCell& down, const SimpleCell& /*lr*/, const unsigned&)
    {
        val = (up.val + left.val + self.val + right.val + down.val) * 0.2;
    }

    double val;
};

template<>
class ProvidesDirectUpdate<SimpleCell> : public boost::true_type {};

}

#endif
