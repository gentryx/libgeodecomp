#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#include <algorithm>
#include <libgeodecomp/parallelization/partitioningsimulator/nodes.h>

namespace LibGeoDecomp {

Nodes Nodes::firstNum(const unsigned& num)
{
    return Nodes::range(0, num);
}


Nodes Nodes::range(const unsigned& start, const unsigned& size)
{
    Nodes result;
    for (unsigned i = start; i < start + size; i++)
        result.insert(i);
    return result;
}


Nodes Nodes::singletonSet(const unsigned& node)
{
    Nodes result;
    result.insert(node);
    return result;
}


Nodes::NodesPair Nodes::splitInHalf() const
{
    // first half is larger or equal in size to second half
    NodesPair result;
    unsigned secondSize = size() / 2;
    unsigned firstSize = size() - secondSize;
    unsigned index = 0;
    for (const_iterator i = this->begin(); i != this->end(); i++) {
        if (index < firstSize) {
            result.first.insert(*i);
        } else {
            result.second.insert(*i);
        }
        index++;
    }
    return result;
}

};
#endif
