#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#include <math.h>
#include <stdexcept>
#include <libgeodecomp/parallelization/partitioningsimulator/splitter.h>

namespace LibGeoDecomp {

Splitter::Splitter(const DVec& powers, const SplitDirection& direction): 
    _powers(powers),
    _table(powers.size()),
    _direction(direction)
{}


Splitter::Splitter(
    const DVec& powers, 
    const ClusterTable& table, 
    const SplitDirection& direction): 
    _powers(powers),
    _table(table),
    _direction(direction)
{
    if (_powers.size() != _table.numNodes()) {
        throw std::invalid_argument("Splitter:\
            powers.size() and table.numNodes() do not match");
    }
}


Splitter::Result Splitter::splitRect(
        const CoordBox<2>& rect,
        const Nodes& nodes) const
{
    if (nodes.size() == 0) {
        throw std::invalid_argument(
                "Splitter::splitRect: nodes.size() may not be 0");
    }

    Result result;
    Nodes::NodesPair nodesPair = splitNodes(_table, nodes);
    result.leftNodes = nodesPair.first;
    result.rightNodes = nodesPair.second;
    double leftWeight = 0;
    double rightWeight = 0;

    for (Nodes::iterator i = result.leftNodes.begin(); 
            i != result.leftNodes.end(); i++) {
        leftWeight += powers()[*i];
    }
    for (Nodes::iterator i = result.rightNodes.begin(); 
            i != result.rightNodes.end(); i++) {
        rightWeight += powers()[*i];
    }
    double weightRatio = leftWeight / (leftWeight + rightWeight);

    if ((_direction == VERTICAL) || 
        ((_direction == LONGEST) && (rect.dimensions.x() > rect.dimensions.y()))) {
        UPair widthPair = splitVec(
            weightsInDirection(rect, Coord<2>(1, 0)), 
            weightRatio);
        result.leftRect = CoordBox<2>(
            rect.origin, 
            Coord<2>(widthPair.first, rect.dimensions.y()));
        result.rightRect = CoordBox<2>(
            rect.origin + Coord<2>(widthPair.first, 0),
            Coord<2>(widthPair.second, rect.dimensions.y()));
    } else {
        UPair heightPair = splitVec(
            weightsInDirection(rect, Coord<2>(0, 1)), 
            weightRatio);
        result.leftRect = CoordBox<2>(
                rect.origin, 
                Coord<2>(rect.dimensions.x(), heightPair.first));
        result.rightRect = CoordBox<2>(
            rect.origin + Coord<2>(0, heightPair.first),
            Coord<2>(rect.dimensions.x(), heightPair.second));
    }
    return result;
}


UPair Splitter::splitUnsigned(const unsigned& n, const double& ratio)
{
    if (ratio < 0 || ratio > 1) {
        throw std::invalid_argument(
                "Splitter::splitUnsigned: ratio must be in [0, 1]");
    }

    unsigned left = (unsigned)round(n * ratio);
    unsigned right = n - left;
    return UPair(left, right);
}


DVec Splitter::powers() const
{
    return _powers;
}


double Splitter::weight(const CoordBox<2>& rect) const
{
    return rect.size();
}


UPair Splitter::newGuess(
        const UPair& bestGuess, 
        const double& bestError, 
        const unsigned& size) 
{
    unsigned first = bestGuess.first;
    unsigned second = bestGuess.second;
    
    if (bestError < 0) { // first is to small
        first += 1;
        second -= 1;
    } else if (bestError > 0) { // first is to big
        first -= 1;
        second += 1;
        return UPair(bestGuess.first - 1, bestGuess.second + 1);
    }

    if (first > size || second > size) {
        std::ostringstream error;
        error << "Splitter::newGuess: invalid guess ("
            << first << ", " << second << "). for a split of " << size
            << "\nthis may happen when negative weights are involved";
        throw std::logic_error(error.str());
    }
    return UPair(first, second);
}


DVec Splitter::weightsInDirection(const CoordBox<2>& rect, const Coord<2>& dir) const
{
    unsigned count;
    unsigned width;
    unsigned height;
    Coord<2> inc;
    if (dir.x() == 0) {
        count = rect.dimensions.y();
        width = rect.dimensions.x();
        height = 1;
        inc = Coord<2>(0, 1);
    } else {
        count = rect.dimensions.x();;
        width = 1;
        height = rect.dimensions.y();
        inc = Coord<2>(1, 0);
    }

    DVec result(count);
    for (unsigned i = 0; i < count; i++)
        result[i] = weight(CoordBox<2>(
                               rect.origin + (inc * i), 
                               Coord<2>(width, height)));
    return result;
}


Nodes::NodesPair Splitter::splitNodes(
    const ClusterTable& table, const Nodes& nodes) 
{
    ClusterTable subSet = table.subSet(nodes);
    unsigned numNodes = subSet.numNodes();
    if (numNodes < nodes.size()) {
        throw std::invalid_argument("ClusterTable::splitNodes:\
            table doesn't know all of the argument nodes");
    }

    if (subSet.size() > 1) {

        Nodes first;
        Nodes second;
        subSet.sort();
        UPair split = Splitter::splitVec(subSet.sizes(), 0.5);
        for (unsigned i = 0; i < subSet.size(); i++) {
            if (i < split.first) first = first || subSet[i];
            else second = second || subSet[i];
        }
        return Nodes::NodesPair(first, second);
    } else {
        return nodes.splitInHalf();
    }
}

};
#endif
