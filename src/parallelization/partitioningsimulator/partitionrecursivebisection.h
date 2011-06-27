#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_partitioningsimulator_partitionrecursivebisection_h_
#define _libgeodecomp_parallelization_partitioningsimulator_partitionrecursivebisection_h_

#include <libgeodecomp/misc/commontypedefs.h>
#include <libgeodecomp/misc/supermap.h>
#include <libgeodecomp/misc/supervector.h>
#include <libgeodecomp/parallelization/partitioningsimulator/partition.h>
#include <libgeodecomp/parallelization/partitioningsimulator/splitter.h>
#include <libgeodecomp/parallelization/partitioningsimulator/treenode.h>
#include <libgeodecomp/parallelization/partitioningsimulator/nodes.h>
#include <libgeodecomp/parallelization/partitioningsimulator/clustertable.h>

namespace LibGeoDecomp {

/**
 * This class implements a partitioning into rectangles by recursive bisection
 */
class PartitionRecursiveBisection : public Partition
{
    friend class PartitionRecursiveBisectionTest;
    friend class MPILayer;

public:
    /**
     * Constructor guarantees that @a *splitter will not be used after
     * construction finishes
     */
    PartitionRecursiveBisection(
            const CoordBox<2>& rect, 
            const Nodes& nodes,
            const Splitter& splitter);
    
    /**
     * whenever a split occurs that involves nodes from different clusters
     * the split that was used for model will be taken.
     */
    PartitionRecursiveBisection(
            const CoordBox<2>& rect, 
            const Nodes& nodes,
            const Splitter& splitter,
            const PartitionRecursiveBisection& model,
            const ClusterTable& table);

    ~PartitionRecursiveBisection();

    CoordBox<2> coordsForNode(const unsigned& node) const;

    CoordBox<2> rectForNodes(const Nodes& nodes) const;

    unsigned nodeForCoord(const Coord<2>& coord) const;

    inline Nodes getNodes() const { return _nodes; }

    void copyFromBroadcast(MPILayer& mpilayer, const unsigned& root);
    
private:
    static const unsigned NULL_INDEX = 0;
    static const unsigned INVALID_NODE = 0;

    /**
     * _nodes are sorted in ascending order
     */
    Nodes _nodes;

    SuperVector<TreeNode> _tree;
    /**
     * holds _tree indices for respective nodes
     */
    UVec _node2index; 

    /**
     * the index of the trees root
     */
    unsigned _root;

    /**
     * Construct a new instance from preobtained member variables
     */
    PartitionRecursiveBisection(
            const unsigned& numPartitions,
            const SuperVector<TreeNode>& tree,
            const UVec& node2index,
            const unsigned& root);

    /**
     * Constructs a dummy instance that can be used as a NULL argument
     */
    PartitionRecursiveBisection();

    unsigned nodeForCoord(const Coord<2>& coord, const unsigned& index) const;

    CoordBox<2> rectForNodes(const Nodes& nodes, const unsigned& index) const;

    /**
     * returns index of the trees root
     */
    unsigned growTree(
            const CoordBox<2>& rect, 
            const Nodes& nodes,
            const Splitter& splitter,
            const PartitionRecursiveBisection& model,
            const ClusterTable& table);

    unsigned latestIndex() const;

    /**
     * Returns how rect and nodes where split for this
     * PartitionRecursiveBisection. Throws invalid_argument if the argument
     * combination does not occuur for this instance
     */
    Splitter::Result splitRect(
            const CoordBox<2>& rect,
            const Nodes& nodes) const;

    Nodes nodesInSubtree(const TreeNode& treeNode) const;

    TreeNode findTreeNode(const CoordBox<2>&, const unsigned& index) const;
};

};
#endif
#endif
