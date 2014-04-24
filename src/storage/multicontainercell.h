#ifndef LIBGEODECOMP_STORAGE_MULTICONTAINERCELL_H
#define LIBGEODECOMP_STORAGE_MULTICONTAINERCELL_H

#include <libgeodecomp/storage/containercell.h>
#include <boost/preprocessor/seq.hpp>

#define DECLARE_MULTI_CONTAINER_CELL_MEMBER(INDEX, CELL, MEMBER)        \
    LibGeoDecomp::ContainerCell<BOOST_PP_SEQ_ELEM(0, MEMBER),           \
                                BOOST_PP_SEQ_ELEM(1, MEMBER)>           \
        BOOST_PP_SEQ_ELEM(2, MEMBER);

/**
 * This cell is a wrapper around ContainerCell to allow user code to
 * compose containers with different element types. It expects MEMBERS
 * to be a sequence of member specifications (adhering to the format
 * expected by the Boost Preprocessor library), where each spec is
 * again a sequence of member type, number of elements, and name.
 *
 * See the unit tests for examples of how to use this class.
 */
#define DECLARE_MULTI_CONTAINER_CELL(NAME, MEMBERS)     \
    class NAME                                          \
    {                                                   \
    public:                                             \
    friend class MultiContainerCellTest;                \
                                                        \
    BOOST_PP_SEQ_FOR_EACH(                              \
        DECLARE_MULTI_CONTAINER_CELL_MEMBER,            \
        NAME,                                           \
        MEMBERS);                                       \
                                                        \
    };

#endif
