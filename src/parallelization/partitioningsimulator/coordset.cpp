#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#include <iostream>
#include <sstream>
#include <libgeodecomp/parallelization/partitioningsimulator/coordset.h>

namespace LibGeoDecomp {

std::string CoordSet::toString() const
{
    std::ostringstream temp;
    for (CoordSet::Sequence s = sequence(); s.hasNext();) 
        temp << s.next().toString();
    return temp.str();
}


void CoordSet::insert(CoordBox<2> rect)
{
    for (CoordBoxSequence<2> s = rect.sequence(); s.hasNext();) 
        insert(s.next());
}


void CoordSet::insert(Coord<2> coord)
{
    STDCoordSet::insert(coord);
}


CoordSet::Sequence CoordSet::sequence() const 
{
    return Sequence(*this);
}


CoordSet::Sequence::Sequence(
        const STDCoordSet& set) :
    STDCoordSet(set),
    _it(set.begin()),
    _end(set.end())
{}


bool CoordSet::Sequence::hasNext()
{
    return _it != _end;
}


Coord<2> CoordSet::Sequence::next()
{
    if (hasNext()) {
        Coord<2> result = *_it;
        _it++;
        return result;
    } else {
        std::cout << "\nERROR! next() called at end of sequence\n";
        return Coord<2>(0, 0);
    }
}

};
#endif
