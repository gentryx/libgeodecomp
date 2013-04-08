#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_PARTITIONS_ZCURVEPARTITION_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_PARTITIONS_ZCURVEPARTITION_H

#include <bitset>
#include <boost/multi_array.hpp>
#include <boost/shared_ptr.hpp>
#include <sstream>
#include <stdexcept>
#include <list>
#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/topologies.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/spacefillingcurve.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<int DIMENSIONS>
class ZCurvePartition : public SpaceFillingCurve<DIMENSIONS>
{
    friend class ZCurvePartitionTest;
public:
    const static int DIM = DIMENSIONS;

    typedef SuperVector<Coord<DIM> > CoordVector;
    typedef boost::shared_ptr<boost::multi_array<CoordVector, DIM> > Cache;
    typedef typename Topologies::Cube<DIM>::Topology Topology;
    typedef typename Topology::template LocateHelper<DIM, CoordVector> LocateHelper;
  
    class Square
    {
    public:
        inline Square(
            const Coord<DIM>& _origin, 
            const Coord<DIM> _dimensions, 
            const unsigned& _quadrant) :
            origin(_origin),
            dimensions(_dimensions),
            quadrant(_quadrant)
        {}

        inline std::string toString() const
        {
            std::stringstream s;
            s << "Square(origin:" << origin << ", dimensions:" << dimensions << ", quadrant: " << quadrant << ")";
            return s.str();
        }

        Coord<DIM> origin;
        Coord<DIM> dimensions;
        unsigned quadrant;
    };

    class Iterator : public SpaceFillingCurve<DIM>::Iterator
    {
    public:
        friend class ZCurvePartitionTest;

        using SpaceFillingCurve<DIM>::Iterator::cursor;
        using SpaceFillingCurve<DIM>::Iterator::endReached;
        using SpaceFillingCurve<DIM>::Iterator::hasTrivialDimensions;
        using SpaceFillingCurve<DIM>::Iterator::origin;
        using SpaceFillingCurve<DIM>::Iterator::sublevelState;

        static const int NUM_QUADRANTS = 1 << DIM;

        inline Iterator(
            const Coord<DIM>& origin, 
            const Coord<DIM>& dimensions, 
            const unsigned& pos=0) :
            SpaceFillingCurve<DIM>::Iterator(origin, false)
        {
            squareStack.push_back(Square(origin, dimensions, 0));
            digDown(pos);
        }

        inline Iterator(const Coord<DIM>& origin) :
            SpaceFillingCurve<DIM>::Iterator(origin, true)
        {}

        inline Iterator& operator++()
        {
            // std::cout << "operator++\n";
            if (endReached)
                return *this;
            if (sublevelState == TRIVIAL) {
                operatorIncTrivial();
            } else {
                operatorIncCached();
            }
            return *this;
        }
    private:
        std::list<Square> squareStack;
        unsigned trivialSquareDirDim;
        unsigned trivialSquareCounter;
        Coord<DIM> cachedSquareOrigin;
        Coord<DIM> *cachedSquareCoordsIterator;
        Coord<DIM> *cachedSquareCoordsEnd;
        
        inline void operatorIncTrivial()
        {
            // std::cout << "operatorIncTrivial, " << trivialSquareCounter << "\n"
            //           << cursor << "\n";
            if (--trivialSquareCounter > 0) {
                cursor[trivialSquareDirDim]++;
            } else {
                digUpDown();
            }
        }

        inline void operatorIncCached()
        {
            // std::cout << "operatorIncCached\n";
            cachedSquareCoordsIterator++;
            if (cachedSquareCoordsIterator != cachedSquareCoordsEnd) {
                cursor = cachedSquareOrigin + *cachedSquareCoordsIterator;
            } else {
                digUpDown();
            }
        }

        inline void digDown(const unsigned& offset)
        {
            // std::cout << "digDown(" << offset << ")\n";
            if (squareStack.empty())
                throw std::logic_error("cannot descend from empty squares stack");
            Square currentSquare = squareStack.back();
            squareStack.pop_back();
            const Coord<DIM>& origin = currentSquare.origin;
            const Coord<DIM>& dimensions = currentSquare.dimensions; 
            // std::cout << "digging1\n";

            if (offset >= dimensions.prod()) {
                // std::cout << "digging2\n";
                endReached = true;
                cursor = origin;
                return;
            }
            if (hasTrivialDimensions(dimensions)) {
                // std::cout << "digging3\n";
                digDownTrivial(origin, dimensions, offset);
            } else if (isCached(dimensions)) {
                // std::cout << "digging4\n";
                digDownCached(origin, dimensions, offset);
            } else {
                // std::cout << "digging5\n";
                digDownRecursion(offset, currentSquare);
            }
            // std::cout << "digging6\n";
        }

        inline void digDownTrivial(
            const Coord<DIM>& origin, 
            const Coord<DIM>& dimensions, 
            const unsigned& offset)
        {
            sublevelState = TRIVIAL;
            cursor = origin;

            // std::cout << "digDownTrivial()\n"
            //           << origin << "\n"
            //           << dimensions << "\n"
            //           << offset << "\n"
            //           << "cursor: " << cursor << "\n\n";

            trivialSquareDirDim = 0;
            for (int i = 1; i < DIM; ++i) 
                if (dimensions[i] > 1) 
                    trivialSquareDirDim = i;
                
            trivialSquareCounter = dimensions[trivialSquareDirDim] - offset;
            cursor[trivialSquareDirDim] += offset;

        }

        inline void digDownCached(
            const Coord<DIM>& origin, 
            const Coord<DIM>& dimensions, 
            const unsigned& offset)
        {
            sublevelState = CACHED;
            CoordVector& coords = 
                LocateHelper()(
                    *ZCurvePartition<DIM>::coordsCache,
                    dimensions,
                    maxCachedDimensions);

            cachedSquareOrigin = origin;
            cachedSquareCoordsIterator = &coords[offset];
            cachedSquareCoordsEnd      = &coords[0] + coords.size();
            cursor = cachedSquareOrigin + *cachedSquareCoordsIterator;
        }

        inline void digDownRecursion(const unsigned& offset, Square currentSquare)
        {
            const Coord<DIM>& dimensions = currentSquare.dimensions; 
            Coord<DIM> halfDimensions = dimensions / 2;
            Coord<DIM> remainingDimensions = dimensions - halfDimensions;

            const int numQuadrants = ZCurvePartition<DIM>::Iterator::NUM_QUADRANTS;
            Coord<DIM> quadrantDims[numQuadrants];

            for (int i = 0; i < numQuadrants; ++i) {
                // high bits denote that the quadrant is in the upper
                // half in respect to that dimension, e.g. quadrant 2
                // would be (for 2D) in the lower half for dimension 0
                // but in the higher half for dimension 1.
                std::bitset<DIM> quadrantShift(i);
                Coord<DIM> quadrantDim;

                for (int d = 0; d < DIM; ++d) {
                    quadrantDim[d] = quadrantShift[d]?
                        remainingDimensions[d] :
                        halfDimensions[d];
                }

                quadrantDims[i] = quadrantDim;
            }

            unsigned accuSizes[numQuadrants];
            accuSizes[0] = 0;
            for (int i = 1; i < numQuadrants; ++i) 
                accuSizes[i] = accuSizes[i - 1] + quadrantDims[i - 1].prod();

            unsigned pos = offset + accuSizes[currentSquare.quadrant];
            unsigned index = std::upper_bound(
                accuSizes, 
                accuSizes + numQuadrants,
                pos) - accuSizes - 1;

            if (index >= (1 << DIM)) 
                throw std::logic_error("offset too large?");

            unsigned newOffset = pos - accuSizes[index];
            Coord<DIM> newDimensions = quadrantDims[index];
            Coord<DIM> newOrigin;

            std::bitset<DIM> quadrantShift(index);
            for (int d = 0; d < DIM; ++d) {
                newOrigin[d] = quadrantShift[d]? 
                    halfDimensions[d] :
                    0;
            }
            newOrigin += currentSquare.origin;

            // std::cout << "digDownRecursion(" << offset
            //           << ",\n " << currentSquare.origin 
            //           << ",\n " << currentSquare.dimensions
            //           << ",\n " << currentSquare.quadrant << ")\n"
            //           << "newOffset: " << newOffset << "\n"
            //           << "newOrigin: " << newOrigin << "\n"
            //           << "newDimensions: " << newDimensions << "\n"
            //           << "index: " << index << "\n"
            //           << "pos: " << pos << "\n"
            //           << "accuSizes[0] " << accuSizes[0] << "\n"
            //           << "accuSizes[1] " << accuSizes[1] << "\n"
            //           << "\n";

            currentSquare.quadrant = index;
            squareStack.push_back(currentSquare);
                
            Square newSquare(newOrigin, newDimensions, 0);
            squareStack.push_back(newSquare);

            digDown(newOffset);
        }

        inline void digUp()
        {
            // std::cout << "digUpDown()\n";
            while (!squareStack.empty() && 
                   (squareStack.back().quadrant == 
                    (ZCurvePartition<DIM>::Iterator::NUM_QUADRANTS - 1))) {
                // std::cout << " " << squareStack.back().origin << "\n"
                //           << " " << squareStack.back().dimensions << "\n"
                //           << " " << squareStack.back().quadrant << "\n\n";
                squareStack.pop_back();
            }

            if (squareStack.empty()) {
                // std::cout << "  empty\n";
                endReached = true;
                cursor = origin;
            } else {
                // std::cout << "  quadrant++\n";
                squareStack.back().quadrant++;
            }
        }

        inline void digUpDown()
        {
            digUp();
            if (endReached) {
                return;
            }
            digDown(0);
        }

        inline bool isCached(const Coord<DIM>& dimensions) const
        {
            bool ret = true;
            for (int i = 0; i < DIM; ++i)
                ret &= dimensions[i] < maxCachedDimensions[i];
            return ret;
        }
    };

    inline ZCurvePartition(
        const Coord<DIM>& _origin=Coord<DIM>(), 
        const Coord<DIM>& _dimensions=Coord<DIM>(),
        const long& offset=0,
        const SuperVector<long>& weights=SuperVector<long>(2)) :
        SpaceFillingCurve<DIM>(offset, weights),
        origin(_origin),              
        dimensions(_dimensions)
    {}

    inline Iterator operator[](const unsigned& i) const
    {
        return Iterator(origin, dimensions, i);
    }

    inline Iterator begin() const
    {
        return (*this)[0];
    }

    inline Iterator end() const
    {
        return Iterator(origin);
    }

    inline Region<DIM> getRegion(const long& node) const 
    {
        return Region<DIM>(
            (*this)[startOffsets[node + 0]], 
            (*this)[startOffsets[node + 1]]);
    }

    static inline bool fillCaches() 
    {
        // store squares of at most maxDim in size. the division by
        // DIM^2 is a trick to keep the cache small if DIM is large.
        Coord<DIM> maxDim = Coord<DIM>::diagonal(68 / DIM / DIM);
        ZCurvePartition<DIM>::coordsCache.reset(
            new boost::multi_array<CoordVector, DIM>(
                maxDim.toExtents()));

        CoordBox<DIM> box(Coord<DIM>(), maxDim);
        for (typename CoordBox<DIM>::Iterator iter = box.begin(); iter != box.end(); ++iter) {
            Coord<DIM> dim = *iter;
            if (!hasTrivialDimensions(dim)) {
                CoordVector coords;
                Iterator end(Coord<DIM>());
                for (Iterator i(Coord<DIM>(), dim, 0); i != end; ++i)
                    coords.push_back(*i);

                LocateHelper()(
                    *ZCurvePartition<DIM>::coordsCache,
                    dim,
                    maxDim) = coords;
            }
        }

        ZCurvePartition<DIM>::maxCachedDimensions = maxDim;
        return true;
    }

private:
    using SpaceFillingCurve<DIM>::startOffsets;

    static Cache coordsCache;
    static Coord<DIMENSIONS> maxCachedDimensions;
    static bool cachesInitialized;

    Coord<DIM> origin;
    Coord<DIM> dimensions;
};

template<int DIM>
typename ZCurvePartition<DIM>::Cache ZCurvePartition<DIM>::coordsCache;

template<int DIM>
Coord<DIM> ZCurvePartition<DIM>::maxCachedDimensions;

template<int DIM>
bool ZCurvePartition<DIM>::cachesInitialized = 
    ZCurvePartition<DIM>::fillCaches();

}
}

#endif
