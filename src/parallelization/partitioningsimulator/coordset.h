#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_partitioningsimulator_coordset_h_
#define _libgeodecomp_parallelization_partitioningsimulator_coordset_h_

#include <set>
#include <libgeodecomp/misc/coordbox.h>

namespace LibGeoDecomp {

typedef std::set<Coord<2> > STDCoordSet;

class CoordSet : public STDCoordSet
{
public:    
    class Sequence : public STDCoordSet
    {
    public: 
        Sequence(const STDCoordSet& set);
        bool hasNext();
        Coord<2> next();

    private:
        STDCoordSet::const_iterator _it;
        STDCoordSet::const_iterator _end;
    };
    void insert(CoordBox<2> rect);
    void insert(Coord<2> coord);

    std::string toString() const;
    Sequence sequence() const; 
};

};

#endif
#endif
