#ifndef LIBGEODECOMP_IO_MOCKINITIALIZER_H
#define LIBGEODECOMP_IO_MOCKINITIALIZER_H

// #include <libgeodecomp/io/testinitializer.h>

// #include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/config.h>

// #include <libgeodecomp/geometry/adjacencymanufacturer.h>
#include <stdexcept>
#include <libgeodecomp/geometry/adjacency.h>

// #include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/regionstreakiterator.h>
#include <libgeodecomp/geometry/streak.h>
#include <libgeodecomp/geometry/topologies.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>

// #include <libgeodecomp/storage/selector.h>
#include <libgeodecomp/config.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/sharedptr.h>
#include <libflatarray/member_ptr_to_offset.hpp>

// #include <libgeodecomp/storage/defaultfilterfactory.h>
#include <libgeodecomp/misc/sharedptr.h>

// ...fine above

#include <ctime>

// #include <libgeodecomp/storage/defaultarrayfilter.h>
// #include <libgeodecomp/config.h>
// #include <libgeodecomp/io/logger.h>
// #include <libgeodecomp/storage/filterbase.h>
// #include <libgeodecomp/storage/memorylocation.h>
// #include <typeinfo>

// fine below...

// #include <libgeodecomp/storage/defaultcudafilter.h>
// #include <libgeodecomp/storage/defaultcudaarrayfilter.h>
// #include <libgeodecomp/storage/defaultfilter.h>


// #include <libgeodecomp/storage/filterbase.h>
// #include <libgeodecomp/storage/memberfilter.h>
// #include <stdexcept>
// #include <typeinfo>


#include <cstddef>
#include <vector>

// #include <libgeodecomp/geometry/regionbasedadjacency.h>
// #include <libgeodecomp/misc/sharedptr.h>

// #include <libgeodecomp/misc/apitraits.h>
// #include <libgeodecomp/misc/random.h>
// #include <libgeodecomp/storage/gridbase.h>
// #include <libgeodecomp/geometry/regionbasedadjacency.h>
// #include <stdexcept>

// #include <libgeodecomp/misc/testcell.h>

namespace LibGeoDecomp {

// Hardwire this warning to off as MSVC would otherwise complain about
// inline functions not being included in object files:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

/**
 * This Initializer will record basic events.
 */
/*class MockInitializer : public TestInitializer<TestCell<2> >
{
public:
    explicit MockInitializer(const std::string& configString = "")
    {
        events += "created, configString: '" + configString + "'\n";
    }

    ~MockInitializer()
    {
        events += "deleted\n";
    }

    static std::string events;
    };*/

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
