#ifndef LIBGEODECOMP_STORAGE_MULTICONTAINERCELL_H
#define LIBGEODECOMP_STORAGE_MULTICONTAINERCELL_H

#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/storage/containercell.h>
#include <boost/preprocessor/seq.hpp>

namespace LibGeoDecomp {

namespace MultiContainerCellHelpers {

/**
 * Below we'll use template instantiations with multiple parameters as
 * parameters for macros. Unfortunately the preprocessor gets confused
 * when arguements to macros contain commas, so we need to enclose
 * those with parentheses. That however turns a datatype into a
 * function type. This helper class can deduce an argument's type from
 * such a function type.
 */
template<typename Type>
class ArgumentType
{};

/**
 * see above
 */
template<typename ReturnType, typename Argument>
class ArgumentType<ReturnType(Argument)>
{
public:
    typedef Argument Value;
};

}

}

#define DECLARE_MULTI_NEIGHBORHOOD_COLLECTION_INTERFACE(INDEX, CELL, MEMBER) \
    class CollectionInterface ## INDEX                                  \
    {                                                                   \
    public:                                                             \
        typedef typename LibGeoDecomp::MultiContainerCellHelpers::ArgumentType<  \
            void (BOOST_PP_SEQ_ELEM(0, MEMBER))>::Value Container;      \
                                                                        \
        inline                                                          \
        const Container& operator()(                                    \
            const CELL& cell)                                           \
        {                                                               \
            return cell.BOOST_PP_SEQ_ELEM(1, MEMBER);                   \
        }                                                               \
                                                                        \
    };

#define DECLARE_MULTI_NEIGHBORHOOD_ADAPTER_ADAPTER(INDEX, CELL, MEMBER) \
    typedef typename LibGeoDecomp::MultiContainerCellHelpers::ArgumentType< \
        void (BOOST_PP_SEQ_ELEM(0, MEMBER))>::Value::NeighborhoodAdapter< \
            CELL,                                                       \
            NEIGHBORHOOD,                                               \
            CollectionInterface ## INDEX>::Value AdapterHelper ## INDEX;

#define DECLARE_MULTI_NEIGHBORHOOD_ADAPTER_MEMBER(INDEX, CELL, MEMBER)  \
    AdapterHelper ## INDEX                                              \
        BOOST_PP_SEQ_ELEM(1, MEMBER);

#define DECLARE_MULTI_NEIGHBORHOOD_ADAPTER_INIT(INDEX, CELL, MEMBER)    \
    BOOST_PP_SEQ_ELEM(1, MEMBER)(writeContainer, hood),

#define DECLARE_MULTI_CONTAINER_CELL_MEMBER(INDEX, UNUSED, MEMBER)      \
    LibGeoDecomp::MultiContainerCellHelpers::ArgumentType<void (BOOST_PP_SEQ_ELEM(0, MEMBER))>::Value BOOST_PP_SEQ_ELEM(1, MEMBER);

#define DECLARE_MULTI_CONTAINER_COPY_MEMBER(INDEX, CELL, MEMBER)        \
    BOOST_PP_SEQ_ELEM(1, MEMBER).copyOver(                              \
        hood[Coord<DIM>()].BOOST_PP_SEQ_ELEM(1, MEMBER),                \
        multiHood.BOOST_PP_SEQ_ELEM(1, MEMBER),                         \
        nanoStep);

#define DECLARE_MULTI_CONTAINER_CELL_UPDATE(INDEX, CELL, MEMBER)        \
    BOOST_PP_SEQ_ELEM(1, MEMBER).updateCargo(multiHood, nanoStep);

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
                NAME *writeContainer,                                   \
                const NEIGHBORHOOD *hood) :                             \
                BOOST_PP_SEQ_FOR_EACH(                                  \
                    DECLARE_MULTI_NEIGHBORHOOD_ADAPTER_INIT,            \
                    NAME,                                               \
                    MEMBERS)                                            \
                writeContainer(writeContainer),                         \
                hood(hood)                                              \
            {}                                                          \
                                                                        \
            NAME *operator->()                                          \
            {                                                           \
                return writeContainer;                                  \
            }                                                           \
                                                                        \
            BOOST_PP_SEQ_FOR_EACH(                                      \
                DECLARE_MULTI_NEIGHBORHOOD_ADAPTER_MEMBER,              \
                NAME,                                                   \
                MEMBERS)                                                \
                                                                        \
            NAME *writeContainer;                                       \
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
                           int nanoStep)                                \
        {                                                               \
            *this = hood[LibGeoDecomp::Coord<DIM>()];                   \
            MultiNeighborhoodAdapter<NEIGHBORHOOD> multiHood(           \
                this,                                                   \
                &hood);                                                 \
                                                                        \
            BOOST_PP_SEQ_FOR_EACH(                                      \
                DECLARE_MULTI_CONTAINER_COPY_MEMBER,                    \
                NAME,                                                   \
                MEMBERS)                                                \
                                                                        \
            BOOST_PP_SEQ_FOR_EACH(                                      \
                DECLARE_MULTI_CONTAINER_CELL_UPDATE,                    \
                NAME,                                                   \
                MEMBERS)                                                \
        }                                                               \
    };

#endif
