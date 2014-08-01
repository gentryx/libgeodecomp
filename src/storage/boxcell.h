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

    template<typename NEIGHBORHOOD>
    class NeighborhoodAdapter
    {
    public:
        typedef NeighborhoodIterator<NEIGHBORHOOD, DIM> Iterator;

        inline
        NeighborhoodAdapter(const NEIGHBORHOOD& hood) :
            myBegin(Iterator::begin(hood)),
            myEnd(Iterator::end(hood))
        {}

        inline
        const Iterator& begin() const
        {
            return myBegin;
        }

        inline
        const Iterator& end() const
        {
            return myEnd;
        }

    private:
        Iterator myBegin;
        Iterator myEnd;
    };

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

    inline
    const Cargo& operator[](const std::size_t i) const
    {
        return particles[i];
    }

    inline
    Cargo& operator[](const std::size_t i)
    {
        return particles[i];
    }

    inline
    BoxCell& operator<<(const Cargo& cargo)
    {
        particles << cargo;
        return *this;
    }

    template<class HOOD>
    inline void update(const HOOD& hood, const int nanoStep)
    {
        NeighborhoodAdapter<HOOD> adapter(hood);

        if (nanoStep == 0) {
            origin    = hood[Coord<DIM>()].origin;
            dimension = hood[Coord<DIM>()].dimension;
            FloatCoord<DIM> oppositeCorner = origin + dimension;
            particles.clear();

            typedef typename NeighborhoodAdapter<HOOD>::Iterator Iterator;

            for (Iterator i = adapter.begin(); i != adapter.end(); ++i) {
                FloatCoord<DIM> particlePos = i->getPos();

                // a particle is withing our cell iff its position is
                // contained in the rectangle/cube spanned by origin
                // and dimension:
                if (origin.dominates(particlePos) &&
                    particlePos.strictlyDominates(oppositeCorner)) {
                    particles << *i;
                }
            }

        } else {
            *this = hood[Coord<DIM>()];
        }

        updateCargo(adapter, nanoStep);
    }

    template<class NEIGHBORHOOD_ADAPTER>
    inline void updateCargo(NEIGHBORHOOD_ADAPTER& neighbors, const int nanoStep)
    {
        for (typename Container::iterator i = particles.begin(); i != particles.end(); ++i) {
            i->update(neighbors, nanoStep);
        }
    }

private:
    FloatCoord<DIM> origin;
    FloatCoord<DIM> dimension;
    Container particles;
};

}

#endif
