#ifndef _libgeodecomp_misc_typetraits_h_
#define _libgeodecomp_misc_typetraits_h_

#include <boost/type_traits.hpp>

namespace LibGeoDecomp {

// fixme: remove
template<class MARKER>
class ProvidesStreakIterator : public boost::false_type 
{};

template<class CELL_TYPE>
class ProvidesStreakUpdate : public boost::false_type 
{};

}

#endif
