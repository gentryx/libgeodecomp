#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/misc/apitraits.h>

namespace LibGeoDecomp {

/**
 * This container can be used with the multi-container cells generated
 * by DECLARE_MULTI_CONTAINER_CELL() to add members which are neither
 * a BoxCell nor a ContainerCell, but just a single value. See the
 * unit test for a use case.
 */
template<typename CELL>
class PassThroughContainer
{
private:
    template<typename WRITE_CONTAINER, typename NEIGHBORHOOD, typename COLLECTION_INTERFACE>
    class ValueWrapper;

public:
    friend class PassThroughContainerTest;

    template<
        typename WRITE_CONTAINER,
        typename NEIGHBORHOOD,
        typename COLLECTION_INTERFACE>
    class NeighborhoodAdapter
    {
    public:
        typedef ValueWrapper<WRITE_CONTAINER, NEIGHBORHOOD, COLLECTION_INTERFACE> Value;
    };

    inline PassThroughContainer& operator=(const CELL& other)
    {
        cell = other;
        return *this;
    }

    template<class HOOD>
    inline void copyOver(const PassThroughContainer& oldSelf, const HOOD& hood, const int nanoStep)
    {
        *this = oldSelf;
    }

    template<class NEIGHBORHOOD_ADAPTER_ALL>
    inline void updateCargo(
        NEIGHBORHOOD_ADAPTER_ALL& allNeighbors,
        int nanoStep)
    {
        cell.update(allNeighbors, nanoStep);
    }

private:
    template<typename WRITE_CONTAINER, typename NEIGHBORHOOD, typename COLLECTION_INTERFACE>
    class ValueWrapper
    {
    public:
        typedef typename APITraits::SelectTopology<CELL>::Value Topology;
        const static int DIM = Topology::DIM;

        ValueWrapper(
            WRITE_CONTAINER *unused,
            const NEIGHBORHOOD *hood) :
            hood(hood)
        {}

        const CELL& operator[](const Coord<DIM>& coord) const
        {
            return COLLECTION_INTERFACE()((*hood)[coord]).cell;
        }

    private:
        const NEIGHBORHOOD *hood;
    };

    CELL cell;
};

}
