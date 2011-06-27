#ifndef _libgeodecomp_misc_typetraits_h_
#define _libgeodecomp_misc_typetraits_h_

#include <boost/type_traits.hpp>

namespace LibGeoDecomp {

/** 
 * specifies whether the cell type supports updates with a low
 * overhead signature (see stepper.h)
 */
template<class CELL_TYPE>
class ProvidesDirectUpdate : public boost::false_type 
{};


/**
 * Can the marker provide fast Streak wise iterators, or only Coord
 * iterators? (see stepper.h)
 */
template<class MARKER>
class ProvidesStreakIterator : public boost::false_type 
{};

template<class CELL_TYPE>
class ProvidesStreakUpdate : public boost::false_type 
{};

}

#endif
