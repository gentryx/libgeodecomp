#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDGRID_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDGRID_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/streak.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/storage/gridbase.h>
#include <libgeodecomp/storage/selector.h>
#include <libgeodecomp/storage/sellcsigmasparsematrixcontainer.h>

#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <cassert>

namespace LibGeoDecomp {

/**
 * A grid type for irregular structures
 */
template<typename ELEMENT_TYPE, std::size_t MATRICES = 1, typename WEIGHT_TYPE = double, int C = 64, int SIGMA = 1>
class UnstructuredGrid : public GridBase<ELEMENT_TYPE, 1, WEIGHT_TYPE>
{
public:
    typedef std::vector<std::pair<ELEMENT_TYPE, WEIGHT_TYPE> > NeighborList;
    typedef typename std::vector<std::pair<ELEMENT_TYPE, WEIGHT_TYPE> >::iterator NeighborListIterator;
    const static int DIM = 1;

    explicit UnstructuredGrid(
        const Coord<DIM>& dim = Coord<DIM>(),
        const ELEMENT_TYPE& defaultElement = ELEMENT_TYPE(),
        const ELEMENT_TYPE& edgeElement = ELEMENT_TYPE(),
        const Coord<DIM>& /* topological dimension is irrelevant here */ = Coord<DIM>()) :
        elements(dim.x(), defaultElement),
        edgeElement(edgeElement),
        dimension(dim)
    {
        for (std::size_t i = 0; i < MATRICES; ++i) {
            matrices[i] =
                SellCSigmaSparseMatrixContainer<WEIGHT_TYPE, C ,SIGMA>(dim.x());
        }
    }

    explicit
    UnstructuredGrid(
        const CoordBox<DIM> box,
        const ELEMENT_TYPE& defaultElement = ELEMENT_TYPE(),
        const ELEMENT_TYPE& edgeElement = ELEMENT_TYPE(),
        const Coord<DIM>& /* topological dimension is irrelevant here */ = Coord<DIM>()) :
        elements(box.dimensions.x(), defaultElement),
        edgeElement(edgeElement),
        dimension(box.dimensions)
    {
        if (box.origin != Coord<DIM>()) {
            throw std::logic_error("UnstructuredGrid can't handle origin in resize(CoordBox)");
        }

        for (std::size_t i = 0; i < MATRICES; ++i) {
            matrices[i] =
                SellCSigmaSparseMatrixContainer<WEIGHT_TYPE, C, SIGMA>(dimension.x());
        }
    }

    UnstructuredGrid& operator=(const UnstructuredGrid& other)
    {
        elements = other.elements;
        edgeElement = other.edgeElement;
        dimension = other.dimension;

        for (std::size_t i = 0; i < MATRICES; ++i) {
            matrices[i] = other.matrices[i];
        }

        return *this;
    }

    void setWeights(std::size_t matrixID, const std::map<Coord<2>, WEIGHT_TYPE>& matrix)
    {
        assert(matrixID < MATRICES);
        matrices[matrixID].initFromMatrix(matrix);
    }

    inline
    const SellCSigmaSparseMatrixContainer<WEIGHT_TYPE, C, SIGMA>& getWeights(const std::size_t matrixID) const
    {
        assert(matrixID < MATRICES);
        return matrices[matrixID];
    }

    inline
    SellCSigmaSparseMatrixContainer<WEIGHT_TYPE, C, SIGMA>& getWeights(const std::size_t matrixID)
    {
        assert(matrixID < MATRICES);
        return matrices[matrixID];
    }

    /**
     * Returns a list of pairs representing the neighborhood of center element.
     * The first element of the pair is the ELEMENT_TYPE
     * and the secound the ADJACENCY_TYPE.
     * The first Element of the list is the center element it self.
     * If center element does not exist the EdggeElement is returned.
     * In both cases ADJACENCY_TYPE = -1
     */
    inline NeighborList getNeighborhood(const Coord<DIM>& center) const
    {
        NeighborList neighborhood;

        if (boundingBox().inBounds(center)) {
            neighborhood.push_back(std::make_pair(*this[center], -1));
            std::vector<std::pair<int, WEIGHT_TYPE> > neighbor =
                matrices[0].getRow(center.x());

            for (NeighborListIterator it = neighbor.begin();
                 it != neighbor.end();
                 ++it) {
                neighborhood.push_back(std::make_pair((*this)[it->first], it->second));
            }
        } else {
            neighborhood.push_back(std::make_pair(getEdgeElement(), -1));
        }

        return neighborhood;
    }

    inline const Coord<DIM>& getDimensions() const
    {
        return dimension;
    }

    inline const ELEMENT_TYPE& operator[](const int y) const
    {
        if (y < 0 || y >= dimension.x()) {
            return getEdgeElement();
        } else {
            return elements[y];
        }
    }

    inline ELEMENT_TYPE& operator[](const int y)
    {
        if (y < 0 || y >= dimension.x()) {
            return getEdgeElement();
        } else {
            return elements[y];
        }
    }

    inline ELEMENT_TYPE& operator[](const Coord<DIM>& coord)
    {
        return (*this)[coord.x()];
    }

    inline const ELEMENT_TYPE& operator[](const Coord<DIM>& coord) const
    {
        return (*this)[coord.x()];
    }

    inline bool operator==(const UnstructuredGrid& other) const
    {
        if (boundingBox() == CoordBox<DIM>() &&
            other.boundingBox() == CoordBox<DIM>()) {
            return true;
        }

        if ((edgeElement != other.edgeElement) ||
            (elements    != other.elements)) {
            return false;
        }

        for (std::size_t i = 0; i < MATRICES; ++i) {
            if (matrices[i] != other.matrices[i]) {
                return false;
            }
        }

        return true;
    }

    inline bool operator==(const GridBase<ELEMENT_TYPE, DIM>& other) const
    {
        if (boundingBox() != other.boundingBox()) {
            return false;
        }

        if (edgeElement != other.getEdge()) {
            return false;
        }

        CoordBox<DIM> box = boundingBox();
        for (typename CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            if ((*this)[*i] != other.get(*i)) {
                return false;
            }
        }

        // fixme: check weights
        return true;
    }

    inline bool operator!=(const UnstructuredGrid& other) const
    {
        return !(*this == other);
    }

    inline bool operator!=(const GridBase<ELEMENT_TYPE, DIM>& other) const
    {
        return !(*this == other);
    }

    // fixme: needs test
    inline void resize(const CoordBox<DIM>& newDim)
    {
        *this = UnstructuredGrid(
            newDim.dimensions,
            elements[0],
            edgeElement);
    }

    inline std::string toString() const
    {
        std::ostringstream message;
        message << "Unstructured Grid <" << DIM << ">(" << dimension.x() << ")\n"
                << "boundingBox: " << boundingBox()  << "\n"
                << "edgeElement: " << edgeElement;

        CoordBox<DIM> box = boundingBox();
        int index = 0;
        for (typename CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            message << "\nCoord " << *i << ":\n"
                    << (*this)[*i] << "\n"
                    << "neighbor: ";

            std::vector<std::pair<int, WEIGHT_TYPE> > neighbor = matrices[0].getRow(index++);
            message << neighbor;
        }

        message << "\n";
        return message.str();
    }

    void set(const Coord<DIM>& coord, const ELEMENT_TYPE& element)
    {
        (*this)[coord] = element;
    }

    void set(const Streak<DIM>& streak, const ELEMENT_TYPE *element)
    {
        for (Coord<DIM> cursor = streak.origin; cursor.x() < streak.endX; ++cursor.x()) {
            (*this)[cursor] = *element;
            ++element;
        }
    }

    ELEMENT_TYPE get(const Coord<DIM>& coord) const
    {
        return (*this)[coord];
    }

    void get(const Streak<DIM>& streak, ELEMENT_TYPE *element) const
    {
        Coord<DIM> cursor = streak.origin;
        for (; cursor.x() < streak.endX; ++cursor.x()) {
            *element = (*this)[cursor];
            ++element;
        }
    }

    inline ELEMENT_TYPE& getEdgeElement()
    {
        return edgeElement;
    }

    inline const ELEMENT_TYPE& getEdgeElement() const
    {
        return edgeElement;
    }

    void setEdge(const ELEMENT_TYPE& element)
    {
        getEdgeElement() = element;
    }

    const ELEMENT_TYPE& getEdge() const
    {
        return getEdgeElement();
    }

    CoordBox<DIM> boundingBox() const
    {
        return CoordBox<DIM>(Coord<DIM>(), dimension);
    }

    inline void saveRegion(std::vector<ELEMENT_TYPE> *buffer, const Region<DIM>& region, const Coord<1>& offset = Coord<DIM>()) const
    {
        ELEMENT_TYPE *target = buffer->data();

        typename Region<DIM>::StreakIterator end = region.endStreak(offset);
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(offset); i != end; ++i) {
            get(*i, target);
            target += i->length();
        }
    }

    inline void loadRegion(const std::vector<ELEMENT_TYPE> buffer, const Region<DIM>& region, const Coord<1>& offset = Coord<DIM>())
    {
        const ELEMENT_TYPE *source = buffer.data();

        typename Region<DIM>::StreakIterator end = region.endStreak(offset);
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(offset); i != end; ++i) {
            set(*i, source);
            source += i->length();
        }
    }

protected:
    void saveMemberImplementation(
        char *target,
        MemoryLocation::Location targetLocation,
        const Selector<ELEMENT_TYPE>& selector,
        const Region<DIM>& region) const
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            selector.copyMemberOut(&(*this)[i->origin], MemoryLocation::HOST, target, targetLocation, i->length());
            target += selector.sizeOfExternal() * i->length();
        }
    }

    void loadMemberImplementation(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<ELEMENT_TYPE>& selector,
        const Region<DIM>& region)
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            selector.copyMemberIn(source, sourceLocation, &(*this)[i->origin], MemoryLocation::HOST, i->length());
            source += selector.sizeOfExternal() * i->length();
        }
    }

private:
    std::vector<ELEMENT_TYPE> elements;
    // TODO wrapper for different types of sell c sigma containers
    SellCSigmaSparseMatrixContainer<WEIGHT_TYPE, C, SIGMA> matrices[MATRICES];
    ELEMENT_TYPE edgeElement;
    Coord<DIM> dimension;
};

template<typename _CharT, typename _Traits, typename ELEMENT_TYPE, std::size_t MATRICES, typename WEIGHT_TYPE, int C, int SIGMA>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const LibGeoDecomp::UnstructuredGrid<ELEMENT_TYPE, MATRICES, WEIGHT_TYPE, C, SIGMA>& grid)
{
    __os << grid.toString();
    return __os;
}

}

#endif
#endif
