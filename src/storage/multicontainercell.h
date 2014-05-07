#ifndef LIBGEODECOMP_STORAGE_MULTICONTAINERCELL_H
#define LIBGEODECOMP_STORAGE_MULTICONTAINERCELL_H

#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/storage/containercell.h>
#include <boost/preprocessor/seq.hpp>

#define DECLARE_MULTI_NEIGHBORHOOD_ADAPTER_ADAPTER(INDEX, CELL, MEMBER) \
    class AdapterHelper ## INDEX                                        \
    {                                                                   \
    public:                                                             \
        AdapterHelper ## INDEX(                                         \
            const NEIGHBORHOOD *hood) :                                 \
            hood(hood)                                                  \
        {}                                                              \
                                                                        \
        const BOOST_PP_SEQ_ELEM(0, MEMBER)& operator[](                 \
            const int& id) const                                        \
        {                                                               \
            const BOOST_PP_SEQ_ELEM(0, MEMBER) *res =                   \
                (*hood)[Coord<DIM>()].BOOST_PP_SEQ_ELEM(2, MEMBER)[id]; \
                                                                        \
            if (res) {                                                  \
                return *res;                                            \
            }                                                           \
                                                                        \
            CoordBox<DIM> box(Coord<DIM>::diagonal(-1),                 \
                              Coord<DIM>::diagonal(3));                 \
            for (CoordBox<DIM>::Iterator i = box.begin();               \
                 i != box.end();                                        \
                 ++i) {                                                 \
                                                                        \
                if (*i != Coord<DIM>()) {                               \
                    res = (*hood)[*i].BOOST_PP_SEQ_ELEM(2, MEMBER)[id]; \
                    if (res) {                                          \
                        return *res;                                    \
                    }                                                   \
                }                                                       \
            }                                                           \
                                                                        \
            LOG(ERROR, "could not find id " << id                       \
                << " in neighborhood AdapterHelper" << INDEX);          \
            throw std::logic_error("id not found");                     \
        }                                                               \
                                                                        \
    private:                                                            \
        const NEIGHBORHOOD *hood;                                       \
                                                                        \
    };

#define DECLARE_MULTI_NEIGHBORHOOD_ADAPTER_MEMBER(INDEX, CELL, MEMBER)  \
    AdapterHelper ## INDEX                                              \
        BOOST_PP_SEQ_ELEM(2, MEMBER);

#define DECLARE_MULTI_NEIGHBORHOOD_ADAPTER_INIT(INDEX, CELL, MEMBER)    \
    BOOST_PP_SEQ_ELEM(2, MEMBER)(hood),

#define DECLARE_MULTI_CONTAINER_CELL_MEMBER(INDEX, CELL, MEMBER)        \
    LibGeoDecomp::ContainerCell<BOOST_PP_SEQ_ELEM(0, MEMBER),           \
                                BOOST_PP_SEQ_ELEM(1, MEMBER)>           \
        BOOST_PP_SEQ_ELEM(2, MEMBER);

#define DECLARE_MULTI_CONTAINER_CELL_UPDATE(INDEX, CELL, MEMBER)        \
    BOOST_PP_SEQ_ELEM(2, MEMBER).updateCargo(multiHood, nanoStep);

/**
 * This cell is a wrapper around ContainerCell to allow user code to
 * compose containers with different element types. It expects MEMBERS
 * to be a sequence of member specifications (adhering to the format
 * expected by the Boost Preprocessor library), where each spec is
 * again a sequence of member type, number of elements, and name.
 *
 * See the unit tests for examples of how to use this class.
 */
#define DECLARE_MULTI_CONTAINER_CELL(NAME, MEMBERS)                     \
    class NAME                                                          \
    {                                                                   \
    public:                                                             \
        friend class MultiContainerCellTest;                            \
                                                                        \
        typedef APITraits::SelectTopology<NAME>::Value Topology;        \
        const static int DIM = Topology::DIM;                           \
                                                                        \
        template<typename NEIGHBORHOOD>                                 \
        class MultiNeighborhoodAdapter                                  \
        {                                                               \
        public:                                                         \
                                                                        \
            BOOST_PP_SEQ_FOR_EACH(                                      \
                DECLARE_MULTI_NEIGHBORHOOD_ADAPTER_ADAPTER,             \
                NAME,                                                   \
                MEMBERS)                                                \
                                                                        \
            MultiNeighborhoodAdapter(const NEIGHBORHOOD *hood) :        \
                BOOST_PP_SEQ_FOR_EACH(                                  \
                    DECLARE_MULTI_NEIGHBORHOOD_ADAPTER_INIT,            \
                    NAME,                                               \
                    MEMBERS)                                            \
                hood(hood)                                              \
            {}                                                          \
                                                                        \
            BOOST_PP_SEQ_FOR_EACH(                                      \
                DECLARE_MULTI_NEIGHBORHOOD_ADAPTER_MEMBER,              \
                NAME,                                                   \
                MEMBERS)                                                \
                                                                        \
            const NEIGHBORHOOD *hood;                                   \
        };                                                              \
                                                                        \
        BOOST_PP_SEQ_FOR_EACH(                                          \
            DECLARE_MULTI_CONTAINER_CELL_MEMBER,                        \
            NAME,                                                       \
            MEMBERS)                                                    \
                                                                        \
        template<class NEIGHBORHOOD>                                    \
        inline void update(const NEIGHBORHOOD& hood,                    \
                           const int& nanoStep)                         \
        {                                                               \
            *this = hood[Coord<DIM>()];                                 \
            MultiNeighborhoodAdapter<NEIGHBORHOOD> multiHood(&hood);    \
                                                                        \
            BOOST_PP_SEQ_FOR_EACH(                                      \
                DECLARE_MULTI_CONTAINER_CELL_UPDATE,                    \
                NAME,                                                   \
                MEMBERS)                                                \
                                                                        \
        }                                                               \
    };

#endif
