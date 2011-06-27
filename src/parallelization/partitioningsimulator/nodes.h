#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_partitioningsimulator_nodes_h_
#define _libgeodecomp_parallelization_partitioningsimulator_nodes_h_

#include <libgeodecomp/misc/commontypedefs.h>

namespace LibGeoDecomp {

/**
 * convenience class for managing sets of nodes represented as unsigned
 */
class Nodes : public USet
{
    friend class NodesTest;

public:
    typedef std::pair<Nodes, Nodes> NodesPair;

    Nodes() {};

    NodesPair splitInHalf() const;
    
    static Nodes firstNum(const unsigned& num); 
    static Nodes range(const unsigned& start, const unsigned& size); 
    static Nodes singletonSet(const unsigned& node);


    inline Nodes operator&&(const Nodes& other) const 
    {        
        return Nodes(USet::operator&&(other));
    }


    inline Nodes operator||(const Nodes& other) const 
    {        
        return Nodes(USet::operator||(other));
    }


private:
    Nodes(const USet& uSet): USet(uSet) {}

};

};

#endif
#endif
