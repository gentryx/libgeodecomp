#ifndef LIBGEODECOMP_STORAGE_BOXCELL_H
#define LIBGEODECOMP_STORAGE_BOXCELL_H

#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/storage/neighborhooditerator.h>
#include <libgeodecomp/storage/fixedarray.h>

namespace LibGeoDecomp {

/**
 * This class is an adapter for implementing n-body codes and
 * molecular dynamics (MD) applications with LibGeoDecomp. A BoxCell
 * represents a fixed volume of the simulation space. It stores those
 * particles (of type Cargo) which reside in its area in the given
 * CONTAINER type (e.g. LibGeoDecomp::FixedArray or std::vector). Particles can
 * access neighboring particles in a given distance during update().
 */
template<typename CONTAINER>
class BoxCell
{
public:
    friend class BoxCellTest;

    typedef CONTAINER Container;
    typedef typename Container::value_type Cargo;
    typedef typename Container::value_type value_type;
    typedef typename Container::const_iterator const_iterator;
    typedef typename Container::iterator iterator;
    typedef typename APITraits::SelectTopology<Cargo>::Value Topology;

    class API :
        public APITraits::SelectAPI<Cargo>::Value,
        public APITraits::HasStencil<Stencils::Moore<Topology::DIM, 1> >
    {};

    const static int DIM = Topology::DIM;

    inline BoxCell(
        const FloatCoord<DIM>& origin = Coord<DIM>(),
        const FloatCoord<DIM>& dimension = Coord<DIM>()) :
        origin(origin),
        dimension(dimension)
    {}

    inline const_iterator begin() const
    {
        return particles.begin();
    }

    inline iterator begin()
    {
        return particles.begin();
    }

    inline const_iterator end() const
    {
        return particles.end();
    }

    inline iterator end()
    {
        return particles.end();
    }

    inline void insert(const Cargo& particle)
    {
        particles << particle;
    }

    inline std::size_t size() const
    {
        return particles.size();
    }

    inline const Container& container() const
    {
        return particles;
    }

    inline Container& container()
    {
        return particles;
    }

    template<class NEIGHBORHOOD>
    inline void update(const NEIGHBORHOOD& hood, const int nanoStep)
    {
        typedef NeighborhoodIterator<NEIGHBORHOOD, Cargo, DIM> HoodIterator;
        HoodIterator begin = HoodIterator::begin(hood);
        HoodIterator end = HoodIterator::end(hood);

        if (nanoStep == 0) {
            origin    = hood[Coord<DIM>()].origin;
            dimension = hood[Coord<DIM>()].dimension;
            FloatCoord<DIM> oppositeCorner = origin + dimension;

            particles.clear();

            for (HoodIterator i = begin; i != end; ++i) {
                FloatCoord<DIM> particlePos = i->getPos();

                // a particle is withing our cell iff its position is
                // contained in the rectangle/cube spanned by origin
                // and dimension:
                if (origin.dominates(particlePos) && particlePos.strictlyDominates(oppositeCorner)) {
                    particles << *i;
                }
            }
        } else {
            *this = hood[Coord<DIM>()];
        }

        for (typename Container::iterator i = particles.begin(); i != particles.end(); ++i) {
            i->update(begin, end, nanoStep);
        }

    }

private:
    FloatCoord<DIM> origin;
    FloatCoord<DIM> dimension;
    Container particles;
};

}

#endif
