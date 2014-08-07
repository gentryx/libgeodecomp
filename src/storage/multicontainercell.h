#ifndef LIBGEODECOMP_STORAGE_MULTICONTAINERCELL_H
#define LIBGEODECOMP_STORAGE_MULTICONTAINERCELL_H

#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/storage/containercell.h>
#include <boost/preprocessor/seq.hpp>

#define DECLARE_MULTI_NEIGHBORHOOD_COLLECTION_INTERFACE(INDEX, CELL, MEMBER) \
    class CollectionInterface ## INDEX                                  \
    {                                                                   \
    public:                                                             \
        typedef BOOST_PP_SEQ_ELEM(0, MEMBER) Container;                 \
                                                                        \
        inline                                                          \
        const BOOST_PP_SEQ_ELEM(0, MEMBER)& operator()(                 \
            const CELL& cell)                                           \
        {                                                               \
            return cell.BOOST_PP_SEQ_ELEM(1, MEMBER);                   \
        }                                                               \
                                                                        \
    };

#define DECLARE_MULTI_NEIGHBORHOOD_ADAPTER_ADAPTER(INDEX, CELL, MEMBER) \
    typedef typename BOOST_PP_SEQ_ELEM(0, MEMBER)::NeighborhoodAdapter<NEIGHBORHOOD, CollectionInterface ## INDEX>::Value AdapterHelper ## INDEX;

#define DECLARE_MULTI_NEIGHBORHOOD_ADAPTER_MEMBER(INDEX, CELL, MEMBER)  \
    AdapterHelper ## INDEX                                              \
        BOOST_PP_SEQ_ELEM(1, MEMBER);

#define DECLARE_MULTI_NEIGHBORHOOD_ADAPTER_INIT(INDEX, CELL, MEMBER)    \
    BOOST_PP_SEQ_ELEM(1, MEMBER)(hood),

#define DECLARE_MULTI_CONTAINER_CELL_MEMBER(INDEX, UNUSED, MEMBER)      \
    BOOST_PP_SEQ_ELEM(0, MEMBER) BOOST_PP_SEQ_ELEM(1, MEMBER);

#define DECLARE_MULTI_CONTAINER_CELL_UPDATE(INDEX, CELL, MEMBER)        \
    BOOST_PP_SEQ_ELEM(1, MEMBER).updateCargo(multiHood.BOOST_PP_SEQ_ELEM(1, MEMBER), multiHood, nanoStep);

/**
 * This cell is a wrapper around ContainerCell to allow user code to
 * compose containers with different element types. It expects MEMBERS
 * to be a sequence of member specifications (adhering to the format
 * expected by the Boost Preprocessor library), where each spec is
 * again a sequence of member type and name.
 *
 * See the unit tests for examples of how to use this class.
 */
#define DECLARE_MULTI_CONTAINER_CELL(NAME, API_PROVIDER, MEMBERS)       \
    class NAME                                                          \
    {                                                                   \
    public:                                                             \
        friend class MultiContainerCellTest;                            \
                                                                        \
        typedef LibGeoDecomp::APITraits::SelectAPI<API_PROVIDER>::Value API; \
        typedef LibGeoDecomp::APITraits::SelectTopology<API_PROVIDER>::Value \
            Topology;                                                   \
        const static int DIM = Topology::DIM;                           \
                                                                        \
        template<typename NEIGHBORHOOD>                                 \
        class MultiNeighborhoodAdapter                                  \
        {                                                               \
        public:                                                         \
                                                                        \
            BOOST_PP_SEQ_FOR_EACH(                                      \
                DECLARE_MULTI_NEIGHBORHOOD_COLLECTION_INTERFACE,        \
                NAME,                                                   \
                MEMBERS)                                                \
                                                                        \
            BOOST_PP_SEQ_FOR_EACH(                                      \
                DECLARE_MULTI_NEIGHBORHOOD_ADAPTER_ADAPTER,             \
                NAME,                                                   \
                MEMBERS)                                                \
                                                                        \
            explicit MultiNeighborhoodAdapter(                          \
                const NEIGHBORHOOD *hood) :                             \
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
            *this = hood[LibGeoDecomp::Coord<DIM>()];                   \
            MultiNeighborhoodAdapter<NEIGHBORHOOD> multiHood(&hood);    \
                                                                        \
            BOOST_PP_SEQ_FOR_EACH(                                      \
                DECLARE_MULTI_CONTAINER_CELL_UPDATE,                    \
                NAME,                                                   \
                MEMBERS)                                                \
        }                                                               \
    };

#endif
