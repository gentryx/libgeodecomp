#ifndef _libgeodecomp_parallelization_hiparsimulator_partitions_recursivebisectionpartition_h_
#define _libgeodecomp_parallelization_hiparsimulator_partitions_recursivebisectionpartition_h_

#include <cmath>
#include <libgeodecomp/misc/floatcoord.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/partition.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<int DIM>
class RecursiveBisectionPartition : public Partition<DIM>
{
    friend class RecursiveBisectionPartitionTest;
public:
    typedef SuperVector<long> LongVec;

    inline RecursiveBisectionPartition(
        const Coord<DIM>& _origin = Coord<DIM>(), 
        const Coord<DIM>& _dimensions = Coord<DIM>(),
        const long& offset = 0,
        const LongVec weights = LongVec(),
        const FloatCoord<DIM>& _dimWeights = Coord<DIM>::diagonal(1)) :
        Partition<DIM>(0, weights),
        origin(_origin),
        dimensions(_dimensions),
        dimWeights(_dimWeights) 
    {}
    
    inline Region<DIM> getRegion(const long& i) const
    {
        CoordBox<DIM> cuboid = searchNodeCuboid(
            startOffsets.begin(), 
            startOffsets.end() - 1,
            startOffsets.begin() + i,
            CoordBox<DIM>(origin, dimensions));

        Region<DIM> r;
        r << cuboid;
        return r;
    }

private:
    using Partition<DIM>::startOffsets;

    Coord<DIM> origin;
    Coord<DIM> dimensions;
    FloatCoord<DIM> dimWeights;

    /**
     * returns the CoordBox which belongs to the node whose weight is
     * being pointed to by the iterator node. We assume that all
     * regions of the nodes from begin to end combined for the
     * CoordBox box.
     */
    CoordBox<DIM> searchNodeCuboid(
        const LongVec::const_iterator& begin,
        const LongVec::const_iterator& end,
        const LongVec::const_iterator& node,
        const CoordBox<DIM>& box) const
    {
        if (std::distance(begin, end) == 1) {
            return box;
        }

        long halfWeight = (*begin + *end) / 2;

        LongVec::const_iterator approxMiddle = std::lower_bound(
            begin, 
            end,
            halfWeight);
        if (*approxMiddle != halfWeight) {
            long delta1 = *approxMiddle - halfWeight;
            LongVec::const_iterator predecessor = approxMiddle - 1;
            long delta2 = halfWeight - *predecessor;
            if (delta2 < delta1) {
                approxMiddle = predecessor;
            }
        }

        double ratio = 1.0 * (*approxMiddle - *begin) / (*end - *begin);
        CoordBox<DIM> newBoxes[2];
        splitBox(newBoxes, box, ratio);

        if (*node < *approxMiddle) {
            return searchNodeCuboid(begin, approxMiddle, node, newBoxes[0]);
        } else {
            return searchNodeCuboid(approxMiddle, end, node, newBoxes[1]);
        }
    }

    inline void splitBox(
        CoordBox<DIM> *newBoxes, 
        const CoordBox<DIM>& oldBox, 
        const double& ratio) const
    {
        newBoxes[0] = oldBox;
        newBoxes[1] = oldBox;

        int longestDim = 0;
        const Coord<DIM>& dim = oldBox.dimensions;
        for (int i = 1; i < DIM; ++i) {
            if (dim[i] > dim[longestDim]) {
                longestDim = i;
            }
        }

        int offset = round(ratio * dim[longestDim]);
        int remainder = dim[longestDim] - offset;
        newBoxes[0].dimensions[longestDim] = offset;
        newBoxes[1].dimensions[longestDim] = remainder;
        newBoxes[1].origin[longestDim] += offset;
    }
};

}
}

template<typename _CharT, typename _Traits, int _Dim>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const typename LibGeoDecomp::HiParSimulator::RecursiveBisectionPartition<_Dim>::Iterator& iter)
{
    __os << iter.toString();
    return __os;
}

#endif
