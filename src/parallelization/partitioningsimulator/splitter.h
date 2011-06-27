#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_partitioningsimulator_splitter_h_
#define _libgeodecomp_parallelization_partitioningsimulator_splitter_h_

#include <cmath>
#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/commontypedefs.h>
#include <libgeodecomp/parallelization/partitioningsimulator/nodes.h>
#include <libgeodecomp/parallelization/partitioningsimulator/clustertable.h>

namespace LibGeoDecomp {

/** 
 * This class and its descendants encapsulate algorithms for splitting a
 * CoordBox<2> among nodes
 */
class Splitter
{
    friend class SplitterTest;

public:
    struct Result 
    {
        Nodes leftNodes;
        Nodes rightNodes;
        CoordBox<2> leftRect;
        CoordBox<2> rightRect;

        inline bool operator==(const Result& other) const
        {
            return 
                leftNodes == other.leftNodes &&
                rightNodes == other.rightNodes &&
                leftRect == other.leftRect &&
                rightRect == other.rightRect;
        }


        std::string toString() const
        {
            std::ostringstream tmp;
            tmp << "leftNodes: " << leftNodes << "\n"
                << "rightNodes: " << rightNodes << "\n"
                << "leftRect: " << leftRect << "\n"
                << "rightRect: " << rightRect << "\n";
            return tmp.str();
        }
    };

    enum SplitDirection {HORIZONTAL, VERTICAL, LONGEST};


    Splitter(const DVec& powers, const SplitDirection& direction = LONGEST);

    Splitter(
        const DVec& powers, 
        const ClusterTable& table, 
        const SplitDirection& direction = LONGEST);

    virtual ~Splitter() {}
    virtual Result splitRect(
            const CoordBox<2>& rect,
            const Nodes& nodes) const;

protected:
    static UPair splitUnsigned(const unsigned& n, const double& ratio = 0.5);

    static UPair newGuess(
            const UPair& bestGuess, 
            const double& bestError,
            const unsigned& size);

    double weightError(
        const UPair& guess, 
        const CoordBox<2>& rect, 
        const double& targetWeight,
        const Coord<2>& dir) const;

    virtual DVec powers() const;
    virtual double weight(const CoordBox<2>& rect) const;

    DVec weightsInDirection(
        const CoordBox<2>& rect,
        const Coord<2>& dir) const;


    template<typename T>
    static T firstSum(const UPair& guess, const SuperVector<T>& vec)
    {
        if (guess.first > vec.size()) {
            std::ostringstream error;
            error << "ModelSplitter::firstWeight: guess ("  
                << guess.first << ", " << guess.second << ")"
                << "is invalid for a split of " << vec.size() << "\n";
            throw std::invalid_argument(error.str());
        }
        T result = 0;
        for (unsigned i = 0; i < guess.first; i++)
            result += vec[i];
        return result;
    }


    /**
     * splits @a nodes in two groups along cluster lines. attempts to make groups
     * close in size
     */
    static Nodes::NodesPair splitNodes(
        const ClusterTable& table, const Nodes& nodes);


    /**
     * tries to split @a vec in 2 consecutive parts with sums as close to @a
     * ratio as possible. @return a pair of the lengths of either half. vector
     * elements must be non-negative
     */
    template<typename T>
    static UPair splitVec(const SuperVector<T>& vec, const double& ratio)
    {
        double targetSum = vec.sum() * ratio;
        UPair bestGuess = splitUnsigned(vec.size(), ratio); 
        double bestError = firstSum(bestGuess, vec) - targetSum;

        UPair guess = newGuess(bestGuess, bestError, vec.size());
        double error = firstSum(guess, vec) - targetSum;

        while (fabs(error) < fabs(bestError)) {
            bestGuess = guess;
            bestError = error;
            guess = newGuess(bestGuess, bestError, vec.size());
            error = firstSum(guess, vec) - targetSum;
        } 
        return bestGuess;
    }

private:
    DVec _powers;
    ClusterTable _table;
    SplitDirection _direction;

};

};

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const LibGeoDecomp::Splitter::Result& result)
{
    __os << result.toString();
    return __os;
}

#endif
#endif
