#ifndef LIBGEODECOMP_GEOMETRY_PARTITIONS_HILBERTPARTITION_H
#define LIBGEODECOMP_GEOMETRY_PARTITIONS_HILBERTPARTITION_H

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/partitions/spacefillingcurve.h>
#include <libgeodecomp/misc/sharedptr.h>
#include <libgeodecomp/storage/grid.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace LibGeoDecomp {

/**
 * An implementation of Hilbert's space-filling curve (SFC). It's
 * limited to 2D and we suggest the use of the ZCurvePartition anyway
 * as it typically yields better results.
 */
class HilbertPartition : public SpaceFillingCurve<2>
{
    friend class HilbertPartitionTest;
private:
    enum Form {LL_TO_LR=0, LL_TO_UL=1, UR_TO_LR=2, UR_TO_UL=3};

public:
    typedef Grid<std::vector<Coord<2> >, Topologies::Cube<3>::Topology> CacheType;

    using Partition<2>::AdjacencyPtr;

    static SharedPtr<CacheType>::Type squareCoordsCache;
    static Form squareFormTransitions[4][4];
    static int squareSectorTransitions[4][4];
    static Coord<2> maxCachedDimensions;
    static bool cachesInitialized;

    class Square
    {
    public:
        inline Square(const Coord<2>& origin, const Coord<2> dimensions, unsigned quadrant, const Form& form) :
            origin(origin),
            dimensions(dimensions),
            quadrant(quadrant),
            form(form)
        {}

        inline std::string toString() const
        {
            std::stringstream s;
            s << "Square(origin:" << origin << ", dimensions:" << dimensions << ", quadrant: " << quadrant << ", form: " << int(form) << ")";
            return s.str();
        }

        Coord<2> origin;
        Coord<2> dimensions;
        unsigned quadrant;
        Form form;
    };

    class Iterator : public SpaceFillingCurve<2>::Iterator
    {
    public:
        using SpaceFillingCurve<2>::Iterator::cursor;
        using SpaceFillingCurve<2>::Iterator::endReached;
        using SpaceFillingCurve<2>::Iterator::hasTrivialDimensions;
        using SpaceFillingCurve<2>::Iterator::sublevelState;

        inline Iterator(
            const Coord<2>& origin,
            const Coord<2>& dimensions,
            unsigned pos=0,
            const Form& form=LL_TO_LR) :
            SpaceFillingCurve<2>::Iterator(origin, false)
        {
            squareStack.push_back(Square(origin, dimensions, 0, form));
            digDown(pos);
        }

        inline explicit Iterator(const Coord<2>& origin) :
            SpaceFillingCurve<2>::Iterator(origin, true)
        {}

        inline Iterator& operator++()
        {
            if (endReached) {
                return *this;
            }
            if (sublevelState == TRIVIAL) {
                operatorIncTrivial();
            } else {
                operatorIncCached();
            }
            return *this;
        }

    private:
        std::vector<Square> squareStack;
        unsigned trivialSquareHorizontal;
        unsigned trivialSquareCounter;
        Coord<2> cachedSquareOrigin;
        Coord<2> *cachedSquareCoordsIterator;
        Coord<2> *cachedSquareCoordsEnd;

        inline void operatorIncTrivial()
        {
            if (--trivialSquareCounter > 0) {
                if (trivialSquareHorizontal) {
                    cursor.x()++;
                } else {
                    cursor.y()++;
                }
            } else {
                digUpDown();
            }
        }

        inline void operatorIncCached()
        {
            cachedSquareCoordsIterator++;
            if (cachedSquareCoordsIterator != cachedSquareCoordsEnd) {
                cursor = cachedSquareOrigin + *cachedSquareCoordsIterator;
            } else {
                digUpDown();
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

        inline void digDown(unsigned offset)
        {
            if (squareStack.empty()) {
                throw std::logic_error("cannot descend from empty squares stack");
            }

            Square currentSquare = pop(squareStack);
            const Coord<2>& origin = currentSquare.origin;
            const Coord<2>& dimensions = currentSquare.dimensions;
            const Form& form = currentSquare.form;

            if ((int)offset >= dimensions.x() * dimensions.y()) {
                endReached = true;
                cursor = origin;
                return;
            }
            if (hasTrivialDimensions(dimensions)) {
                digDownTrivial(origin, dimensions, offset);
            } else if (isCached(dimensions)) {
                digDownCached(origin, dimensions, offset, form);
            } else {
                digDownRecursion(offset, currentSquare);
            }
        }

        inline void digDownTrivial(
            const Coord<2>& origin,
            const Coord<2>& dimensions,
            unsigned offset)
        {
            sublevelState = TRIVIAL;
            cursor = origin;
            if (dimensions.x() > 1) {
                trivialSquareCounter = dimensions.x() - offset;
                trivialSquareHorizontal = true;
                cursor.x() += offset;
            } else {
                trivialSquareCounter = dimensions.y() - offset;
                trivialSquareHorizontal = false;
                cursor.y() += offset;
            }
        }

        inline void digDownCached(
            const Coord<2>& origin,
            const Coord<2>& dimensions,
            unsigned offset,
            const Form& form)
        {
            sublevelState = CACHED;
            Coord<3> c(dimensions.x(),
                       dimensions.y(),
                       form);
            std::vector<Coord<2> >& coords = (*squareCoordsCache)[c];
            cachedSquareOrigin = origin;
            cachedSquareCoordsIterator = &coords[offset];
            cachedSquareCoordsEnd      = &coords[0] + coords.size();
            cursor = cachedSquareOrigin + *cachedSquareCoordsIterator;
        }

        inline void digDownRecursion(
            unsigned offset,
            Square currentSquare)
        {
            const Coord<2>& origin = currentSquare.origin;
            const Coord<2>& dimensions = currentSquare.dimensions;
            const Form& form = currentSquare.form;
            Coord<2> halfDimensions = dimensions / 2;
            Coord<2> restDimensions = dimensions - halfDimensions;
            unsigned totalSize = dimensions.x() * dimensions.y();
            unsigned leftHalfSize = halfDimensions.x() * dimensions.y();
            unsigned rightHalfSize = totalSize - leftHalfSize;
            unsigned upperLeftQuarterSize = halfDimensions.x() * halfDimensions.y();
            unsigned lowerLeftQuarterSize = halfDimensions.x() * restDimensions.y();

            unsigned upperRightQuarterSize = restDimensions.x() * halfDimensions.y();
            unsigned lowerRightQuarterSize = restDimensions.x() * restDimensions.y();
            // accumulated quarter sizes, e.g. accuSizes[0] is the
            // sum of the first 0 quarters, ergo 0, accuSizes[3]
            // is the accumulated size of the first three quarters.
            unsigned accuSizes[4];
            accuSizes[0] = 0;
            switch (form) {
            case LL_TO_LR:
                accuSizes[1] = lowerLeftQuarterSize;
                accuSizes[2] = leftHalfSize;
                accuSizes[3] = leftHalfSize + upperRightQuarterSize;
                break;
            case LL_TO_UL:
                accuSizes[1] = lowerLeftQuarterSize;
                accuSizes[2] = lowerLeftQuarterSize + lowerRightQuarterSize;
                accuSizes[3] = lowerLeftQuarterSize + rightHalfSize;
                break;
            case UR_TO_LR:
                accuSizes[1] = upperRightQuarterSize;
                accuSizes[2] = upperRightQuarterSize + upperLeftQuarterSize;
                accuSizes[3] = upperRightQuarterSize + leftHalfSize;
                break;
            case UR_TO_UL:
                accuSizes[1] = upperRightQuarterSize;
                accuSizes[2] = rightHalfSize;
                accuSizes[3] = rightHalfSize + lowerLeftQuarterSize;
                break;
            default:
                throw std::invalid_argument("illegal form");
            };

            unsigned newQuarter;
            unsigned pos = offset + accuSizes[currentSquare.quadrant];
            if (pos < accuSizes[2]) {
                newQuarter = (pos < accuSizes[1]) ? 0 : 1;
            } else {
                newQuarter = (pos < accuSizes[3]) ? 2 : 3;
            }
            currentSquare.quadrant = newQuarter;
            squareStack.push_back(currentSquare);

            unsigned newOffset = pos - accuSizes[newQuarter];
            Coord<2> newOrigin;
            Coord<2> newDimensions;

            switch (squareSectorTransitions[form][newQuarter]) {
            case 0:
                newOrigin = origin;
                newDimensions = halfDimensions;
                break;
            case 1:
                newOrigin.x() = origin.x() + halfDimensions.x();
                newOrigin.y() = origin.y();
                newDimensions.x() = dimensions.x() - halfDimensions.x();
                newDimensions.y() = halfDimensions.y();
                break;
            case 2:
                newOrigin.x() = origin.x();
                newOrigin.y() = origin.y() + halfDimensions.y();
                newDimensions.x() = halfDimensions.x();
                newDimensions.y() = dimensions.y() - halfDimensions.y();
                break;
            case 3:
                newOrigin = origin + halfDimensions;
                newDimensions = dimensions - halfDimensions;
                break;
            };

            Form newForm = squareFormTransitions[form][newQuarter];
            Square newSquare(newOrigin, newDimensions, 0, newForm);
            squareStack.push_back(newSquare);

            digDown(newOffset);
        }

        inline void digUp()
        {
            while (!squareStack.empty()) {
                if (++squareStack.back().quadrant == 4) {
                    squareStack.pop_back();
                } else {
                    return;
                }
            }
            endReached = true;
            cursor = origin;
        }

        inline bool isCached(const Coord<2>& dimensions) const
        {
            return (dimensions.x() < maxCachedDimensions.x() &&
                    dimensions.y() < maxCachedDimensions.y());
        }
    };

    inline explicit HilbertPartition(
        const Coord<2>& origin = Coord<2>(0, 0),
        const Coord<2>& dimensions = Coord<2>(0, 0),
        const long& offset = 0,
        const std::vector<std::size_t>& weights = std::vector<std::size_t>(2),
        const AdjacencyPtr& /* unused: adjacency */ = AdjacencyPtr()) :
        SpaceFillingCurve<2>(offset, weights),
        origin(origin),
        dimensions(dimensions)
    {}

    inline Iterator operator[](unsigned i) const
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

    inline Region<2> getRegion(const std::size_t node) const
    {
        return Region<2>(
            (*this)[startOffsets[node + 0]],
            (*this)[startOffsets[node + 1]]);
    }

private:
    using SpaceFillingCurve<2>::startOffsets;

    Coord<2> origin;
    Coord<2> dimensions;

    static inline bool fillCaches()
    {
        Coord<2> maxDim(17, 17);
        squareCoordsCache.reset(new CacheType(Coord<3>(maxDim.x(), maxDim.y(), 4)));

        for (int y = 2; y < maxDim.y(); ++y) {
            maxCachedDimensions = Coord<2>(y, y);
            for (int x = 2; x < maxDim.x(); ++x) {
                Coord<2> dimensions(x, y);
                for (int f = 0; f < 4; ++f) {
                    std::vector<Coord<2> > coords;
                    Iterator end(Coord<2>(0, 0));
                    for (Iterator i(Coord<2>(0, 0), dimensions, 0, (Form)f); i != end; ++i) {
                        coords.push_back(*i);
                    }

                    Coord<3> c(dimensions.x(), dimensions.y(), f);
                    (*squareCoordsCache)[c] = coords;
                }
            }
        }

        maxCachedDimensions = maxDim;
        return true;
    }
};

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const typename HilbertPartition::Square& square)
{
    __os << square.toString();
    return __os;
}

}

#endif
