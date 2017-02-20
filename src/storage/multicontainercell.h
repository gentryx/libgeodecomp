#ifndef LIBGEODECOMP_STORAGE_MULTICONTAINERCELL_H
#define LIBGEODECOMP_STORAGE_MULTICONTAINERCELL_H

#include <libflatarray/preprocessor.hpp>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/storage/containercell.h>

namespace LibGeoDecomp {

namespace MultiContainerCellHelpers {

/**
 * Below we'll use template instantiations with multiple parameters as
 * parameters for macros. Unfortunately the preprocessor gets confused
 * when arguments to macros contain commas, so we need to enclose
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
            void (LIBFLATARRAY_ELEM(0, MEMBER))>::Value Container;      \
                                                                        \
        inline                                                          \
        const Container& operator()(                                    \
            const CELL& cell)                                           \
        {                                                               \
            return cell.LIBFLATARRAY_ELEM(1, MEMBER);                   \
        }                                                               \
                                                                        \
    };

#define DECLARE_MULTI_NEIGHBORHOOD_ADAPTER_ADAPTER(INDEX, CELL, MEMBER) \
    typedef typename LibGeoDecomp::MultiContainerCellHelpers::ArgumentType< \
        void (LIBFLATARRAY_ELEM(0, MEMBER))>::Value::NeighborhoodAdapter< \
            CELL,                                                       \
            NEIGHBORHOOD,                                               \
            CollectionInterface ## INDEX>::Value AdapterHelper ## INDEX;

#define DECLARE_MULTI_NEIGHBORHOOD_ADAPTER_MEMBER(INDEX, CELL, MEMBER)  \
    AdapterHelper ## INDEX                                              \
        LIBFLATARRAY_ELEM(1, MEMBER);

#define DECLARE_MULTI_NEIGHBORHOOD_ADAPTER_INIT(INDEX, CELL, MEMBER)    \
    LIBFLATARRAY_ELEM(1, MEMBER)(writeContainer, hood),

#define DECLARE_MULTI_CONTAINER_CELL_MEMBER(INDEX, UNUSED, MEMBER)      \
    LibGeoDecomp::MultiContainerCellHelpers::ArgumentType<void (LIBFLATARRAY_ELEM(0, MEMBER))>::Value LIBFLATARRAY_ELEM(1, MEMBER);

#define DECLARE_MULTI_CONTAINER_COPY_MEMBER(INDEX, CELL, MEMBER)        \
    LIBFLATARRAY_ELEM(1, MEMBER).copyOver(                              \
        hood[Coord<DIM>()].LIBFLATARRAY_ELEM(1, MEMBER),                \
        multiHood.LIBFLATARRAY_ELEM(1, MEMBER),                         \
        nanoStep);

#define DECLARE_MULTI_CONTAINER_CELL_UPDATE(INDEX, CELL, MEMBER)        \
    LIBFLATARRAY_ELEM(1, MEMBER).updateCargo(multiHood, nanoStep);

/**
 * This cell is a wrapper around ContainerCell and BoxCell to allow
 * user code to compose containers with different element types.
 *
 * Example: we could have freely moving particles stored in a BoxCell
 * and static elements stored in a ContainerCell, with elements and
 * particles interacting with each other. This Macro will generate the
 * necessary glue to stitch both components of the model together.
 *
 * See the unit tests for examples of how to use this class.
 *
 * The macro expects MEMBERS to be a sequence of member specifications
 * (adhering to the format expected by the Boost Preprocessor
 * library), where each spec is again a sequence of member type and
 * name.
 *
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
            LIBFLATARRAY_FOR_EACH(                                      \
                DECLARE_MULTI_NEIGHBORHOOD_COLLECTION_INTERFACE,        \
                NAME,                                                   \
                MEMBERS)                                                \
                                                                        \
            LIBFLATARRAY_FOR_EACH(                                      \
                DECLARE_MULTI_NEIGHBORHOOD_ADAPTER_ADAPTER,             \
                NAME,                                                   \
                MEMBERS)                                                \
                                                                        \
            MultiNeighborhoodAdapter(                                   \
                NAME *writeContainer,                                   \
                const NEIGHBORHOOD *hood) :                             \
                LIBFLATARRAY_FOR_EACH(                                  \
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
            LIBFLATARRAY_FOR_EACH(                                      \
                DECLARE_MULTI_NEIGHBORHOOD_ADAPTER_MEMBER,              \
                NAME,                                                   \
                MEMBERS)                                                \
                                                                        \
            NAME *writeContainer;                                       \
            const NEIGHBORHOOD *hood;                                   \
        };                                                              \
                                                                        \
        LIBFLATARRAY_FOR_EACH(                                          \
            DECLARE_MULTI_CONTAINER_CELL_MEMBER,                        \
            NAME,                                                       \
            MEMBERS)                                                    \
                                                                        \
        template<class NEIGHBORHOOD>                                    \
        inline void update(const NEIGHBORHOOD& hood,                    \
                           int nanoStep)                                \
        {                                                               \
            MultiNeighborhoodAdapter<NEIGHBORHOOD> multiHood(           \
                this,                                                   \
                &hood);                                                 \
                                                                        \
            LIBFLATARRAY_FOR_EACH(                                      \
                DECLARE_MULTI_CONTAINER_COPY_MEMBER,                    \
                NAME,                                                   \
                MEMBERS)                                                \
                                                                        \
            LIBFLATARRAY_FOR_EACH(                                      \
                DECLARE_MULTI_CONTAINER_CELL_UPDATE,                    \
                NAME,                                                   \
                MEMBERS)                                                \
        }                                                               \
    };

#endif
