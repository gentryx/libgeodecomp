#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#include <libgeodecomp/parallelization/partitioningsimulator/partition.h>

namespace LibGeoDecomp {

Partition::Partition(const CoordBox<2>& rect, const unsigned& numPartitions):
    _rect(rect),
    _numPartitions(numPartitions)
{}


Partition::~Partition()
{}


bool Partition::inBounds(Coord<2> coord) const
{
    return _rect.inBounds(coord);
}


unsigned Partition::getWidth() const
{
    return _rect.dimensions.x();
}


unsigned Partition::getHeight() const
{
    return _rect.dimensions.y();
}


unsigned Partition::getNumPartitions() const
{
    return _numPartitions;
}


std::string Partition::toString() const
{
    std::stringstream s;
    Nodes nodes = getNodes();
    s << "{\n";
    for (Nodes::iterator i = nodes.begin(); i != nodes.end(); i++) 
        s << *i << " => " << coordsForNode(*i);
    s << "}\n";
    return s.str();
}


bool Partition::operator==(const Partition& other) const
{
    return equal(&other);
}


bool Partition::equal(const Partition* other) const
{
    Nodes nodes = getNodes();
    if (nodes == other->getNodes()) {
        for (Nodes::iterator i = nodes.begin(); i != nodes.end(); i++) 
            if (coordsForNode(*i) != other->coordsForNode(*i)) return false;
        return true;
    } else {
        return false;
    }

}


void Partition::copyFromBroadcast(MPILayer& mpilayer, const unsigned& root)
{
    _rect = mpilayer.broadcast(_rect, root);
    _numPartitions = mpilayer.broadcast(_numPartitions, root);
}


bool Partition::compatible(const Partition& other) const
{
    return (getNodes() == other.getNodes()) && 
        (getRect() == other.getRect());
}

};
#endif
