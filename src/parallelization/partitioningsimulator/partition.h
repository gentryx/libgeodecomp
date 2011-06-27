#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_partitioningsimulator_partition_h_
#define _libgeodecomp_parallelization_partitioningsimulator_partition_h_

#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/commontypedefs.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/parallelization/partitioningsimulator/nodes.h>

namespace LibGeoDecomp {

/** 
 * This class embodies the partition of a rectangular grind unsigned
 * into numPartitions rectangular regions. The naming of partitions (and
 * nodes respectively) is zero-based.
 */
class Partition
{
public:
    Partition(const CoordBox<2>& rect, const unsigned& numPartitions);

    virtual ~Partition();
    virtual CoordBox<2> coordsForNode(const unsigned& node) const = 0;
    
    /**
     * if the coordsForNode for nodes form a rectangle, return this rectangle.
     * otherwise throw std::invalid_argument
     */
    virtual CoordBox<2> rectForNodes(const Nodes& nodes) const = 0;

    virtual unsigned nodeForCoord(const Coord<2>& coord) const = 0;
    bool inBounds(Coord<2> coord) const;
    unsigned getNumPartitions() const;
    virtual Nodes getNodes() const = 0;
    inline CoordBox<2> getRect() const { return _rect; }

    bool operator==(const Partition& other) const;
    bool equal(const Partition* other) const;

    std::string toString() const;

    /**
     * copy data from a broadcasted source partiotion (which must be of the same
     * type)
     */
    virtual void copyFromBroadcast(MPILayer& mpilayer, const unsigned& root);

    /**
     * true if other has the same underlying rectangle and nodes
     */
    bool compatible(const Partition& other) const;

protected:
    CoordBox<2> _rect;
    unsigned _numPartitions;

    unsigned getWidth() const;
    unsigned getHeight() const;

    inline void failIfCoordOutOfBounds(const Coord<2>& coord) const
    {
        if (not inBounds(coord)) {
            throw std::out_of_range(
                    "Coord " + coord.toString() + 
                    " is outside Partition for " + _rect.toString());
        }
    }

};

};

#endif
#endif
