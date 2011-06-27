#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#include <iostream>
#include <libgeodecomp/parallelization/partitioningsimulator/partitionstriping.h>

namespace LibGeoDecomp {

PartitionStriping::PartitionStriping(const CoordBox<2>& rect, const unsigned& numPartitions):
    Partition(rect, numPartitions),
    _stripeWidth(0)
{
    _stripeWidth = getWidth() / getNumPartitions();
    _leftOver = getWidth() % getNumPartitions();
}

        
CoordBox<2> PartitionStriping::coordsForNode(const unsigned& node) const
{
    // Nodes 0.._leftOver get width++
    // bounds [xmin, xmax[ [ymin, ymax]
    int ymin = 0;
    int ymax = getHeight();
    int xmin = node * _stripeWidth + std::min(node, _leftOver);
    int xmax = xmin + _stripeWidth;
    if (node < _leftOver) xmax++;
    CoordBox<2> result(Coord<2>(xmin, ymin), 
                       Coord<2>(xmax - xmin, ymax - ymin));
    return result;
}


CoordBox<2> PartitionStriping::rectForNodes(const Nodes& nodes) const
{
    Nodes::iterator prev = nodes.begin();
    for (Nodes::iterator i = ++nodes.begin(); i != nodes.end(); i++) {
        if (*prev + 1 != *i) {
            throw std::invalid_argument("PartitionStriping::rectForNodes:\
                    nodes coords do not form a rectangle");
        }
        prev++;
    }
    unsigned firstNode = *(nodes.begin());
    unsigned lastNode = *(nodes.rbegin());
    Coord<2> origin = coordsForNode(firstNode).origin;
    Coord<2> originOpposite = coordsForNode(lastNode).originOpposite();
    unsigned width = originOpposite.x() - origin.x() + 1;
    unsigned height = originOpposite.y() - origin.y() + 1;
    return CoordBox<2>(origin, Coord<2>(width, height));
}


unsigned PartitionStriping::nodeForCoord(const Coord<2>& coord) const
{
    failIfCoordOutOfBounds(coord);
    int fatZone = _leftOver * (_stripeWidth + 1); 
    return coord.x() < fatZone ?
        coord.x() / (_stripeWidth + 1) :
        (coord.x() - fatZone) / _stripeWidth + _leftOver;
}


Nodes PartitionStriping::getNodes() const
{
    return Nodes::firstNum(getNumPartitions());
}


void PartitionStriping::copyFromBroadcast(
    MPILayer& mpilayer, const unsigned& root)
{
    Partition::copyFromBroadcast(mpilayer, root);
    _stripeWidth = mpilayer.broadcast(_stripeWidth, root);
    _leftOver = mpilayer.broadcast(_leftOver, root);
}

};
#endif
