#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_partitioningsimulator_treenode_h_
#define _libgeodecomp_parallelization_partitioningsimulator_treenode_h_

namespace LibGeoDecomp {

class TreeNode
{
public:
    friend class Typemaps;

    TreeNode() {}

    TreeNode(bool _inner, unsigned _node, unsigned _left, 
             unsigned _right, CoordBox<2> _rect):
        inner(_inner),
        node(_node),
        left(_left),
        right(_right),
        rect(_rect)
    {}

    bool inner;
    unsigned node;
    unsigned left;
    unsigned right;
    CoordBox<2> rect;
};

}

#endif
#endif
