#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#include <iostream>
#include <sstream>
#include <libgeodecomp/parallelization/partitioningsimulator/partitionrecursivebisection.h>

namespace LibGeoDecomp {

PartitionRecursiveBisection::PartitionRecursiveBisection(
            const CoordBox<2>& rect, 
            const Nodes& nodes,
            const Splitter& splitter):
    Partition(rect, nodes.size()),
    _nodes(nodes),
    _node2index(nodes.size())
{
    _root = growTree(
            rect, 
            nodes, 
            splitter, 
            PartitionRecursiveBisection(), 
            ClusterTable(_numPartitions));
}


PartitionRecursiveBisection::PartitionRecursiveBisection(
        const CoordBox<2>& rect, 
        const Nodes& nodes,
        const Splitter& splitter,
        const PartitionRecursiveBisection& model,
        const ClusterTable& table):
    Partition(rect, nodes.size()),
    _nodes(nodes),
    _node2index(nodes.size())
{
    _root = growTree(rect, nodes, splitter, model, table);
}


PartitionRecursiveBisection::PartitionRecursiveBisection(
        const unsigned& numPartitions,
        const SuperVector<TreeNode>& tree,
        const UVec& node2index,
        const unsigned& root):
    Partition(tree[root].rect, numPartitions),
    _nodes(),
    _tree(tree),
    _node2index(node2index),
    _root(root)
{
    _nodes = nodesInSubtree(_tree[_root]);
    if (_nodes.size() != _numPartitions) {
        throw std::invalid_argument("PartitionRecursiveBisection:\
                Inconsistent number of nodes in Constructor");
    }
}


PartitionRecursiveBisection::PartitionRecursiveBisection():
    Partition(CoordBox<2>(), 0)
{}


PartitionRecursiveBisection::~PartitionRecursiveBisection()
{} 

unsigned PartitionRecursiveBisection::growTree(
            const CoordBox<2>& rect, 
            const Nodes& nodes,
            const Splitter& splitter,
            const PartitionRecursiveBisection& model,
            const ClusterTable& table)
{
    if (nodes.size() > 1) {
        Splitter::Result split; 
        if (table.sameCluster(nodes)) {
            split = splitter.splitRect(rect, nodes);
        } else {
            split = model.splitRect(rect, nodes);
        }
        
        unsigned left = growTree(split.leftRect, split.leftNodes, splitter, model, table);
        unsigned right = growTree(split.rightRect, split.rightNodes, splitter, model, table);
        _tree.push_back(TreeNode(true, INVALID_NODE, left, right, rect));
        return latestIndex();
    
    } else if (nodes.size() == 1) {
        unsigned node = *(nodes.begin());
        _tree.push_back(TreeNode(false, node, NULL_INDEX, NULL_INDEX, rect));
        _node2index[node] = latestIndex();
        return latestIndex();

    } else {
        throw std::invalid_argument("growTree: need at least one node");
    }
}


unsigned PartitionRecursiveBisection::latestIndex() const
{
    return _tree.size() - 1;
}


CoordBox<2> PartitionRecursiveBisection::coordsForNode(
        const unsigned& node) const
{
    return _tree[_node2index[node]].rect;
}


CoordBox<2> PartitionRecursiveBisection::rectForNodes(const Nodes& nodes) const
{
    return rectForNodes(nodes, _root);
}


CoordBox<2> PartitionRecursiveBisection::rectForNodes(
        const Nodes& nodes, const unsigned& index) const
{
    const TreeNode& treeNode = _tree[index];
    if (nodesInSubtree(treeNode) == nodes) {
        return treeNode.rect;
    } else if (treeNode.inner) {
        if (_tree[treeNode.right].rect.contains(
                    coordsForNode(*(nodes.begin())))) {
            return rectForNodes(nodes, treeNode.right);
        } else {
            return rectForNodes(nodes, treeNode.left);
        }
    } else {
        throw std::invalid_argument("PartitionRecursiveBisection::rectForNodes\
                nodes coords do not form a rectangle");
    }
}


unsigned PartitionRecursiveBisection::nodeForCoord(const Coord<2>& coord) const
{
    failIfCoordOutOfBounds(coord);
    return nodeForCoord(coord, _root);
}


unsigned PartitionRecursiveBisection::nodeForCoord(
        const Coord<2>& coord, const unsigned& index) const
{
    const TreeNode& treeNode = _tree[index];
    if (treeNode.inner) {
        if (_tree[treeNode.right].rect.inBounds(coord)) {
            return nodeForCoord(coord, treeNode.right);
        } else {
            return nodeForCoord(coord, treeNode.left);
        }
    } else {
        return treeNode.node;
    }
}


Splitter::Result PartitionRecursiveBisection::splitRect(
            const CoordBox<2>& rect,
            const Nodes& nodes) const
{
    if (nodes.size() == 0) {
        throw std::invalid_argument(
                "PartitionRecursiveBisection::splitRect: \
                nodes.size() may not be 0");
    }

    TreeNode treeNode = findTreeNode(rect, _root);
    Nodes leftNodes = nodesInSubtree(_tree[treeNode.left]);
    Nodes rightNodes = nodesInSubtree(_tree[treeNode.right]);

    Nodes nodesInSubtree = leftNodes || rightNodes;

    if (nodesInSubtree != nodes) {
        throw std::invalid_argument("PartitionRecursiveBisection::splitRect:\
                 this split did not occur (nodes mismatch).");
    }

    Splitter::Result result;
    result.leftRect = _tree[treeNode.left].rect;
    result.rightRect = _tree[treeNode.right].rect;
    result.leftNodes = leftNodes;
    result.rightNodes = rightNodes;
    return result;
}


Nodes PartitionRecursiveBisection::nodesInSubtree(const TreeNode& treeNode) const
{
    if (treeNode.inner) {
        return nodesInSubtree(_tree[treeNode.left]) ||
            nodesInSubtree(_tree[treeNode.right]); 
    } else {
        return Nodes::singletonSet(treeNode.node);
    }
}


TreeNode PartitionRecursiveBisection::findTreeNode(
        const CoordBox<2>& rect, const unsigned& index) const
{
    const TreeNode& treeNode = _tree[index];
    if (treeNode.rect == rect) {
        return treeNode;
    } else if (treeNode.inner) {
        if (_tree[treeNode.right].rect.contains(rect)) {
            return findTreeNode(rect, treeNode.right);
        } else {
            return findTreeNode(rect, treeNode.left);
        }
    } else {
        throw std::invalid_argument("PartitionRecursiveBisection::findTreeNode\
                cannot find Rectangle.");
    }
}


void PartitionRecursiveBisection::copyFromBroadcast(
    MPILayer& mpilayer, const unsigned& root)
{
    Partition::copyFromBroadcast(mpilayer, root);
    _tree = mpilayer.broadcastVector(_tree, root);
    _node2index = mpilayer.broadcastVector(_node2index, root);
    _root = mpilayer.broadcast(_root, root);
    _nodes = nodesInSubtree(_tree[_root]);
}

};
#endif
