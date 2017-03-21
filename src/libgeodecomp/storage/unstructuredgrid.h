#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDGRID_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDGRID_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/streak.h>
#include <libgeodecomp/misc/stringops.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/storage/gridbase.h>
#include <libgeodecomp/storage/selector.h>
#include <libgeodecomp/storage/sellcsigmasparsematrixcontainer.h>

#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <cassert>
#include <stdexcept>

namespace LibGeoDecomp {

/**
 * A grid type for irregular structures
 */
template<typename ELEMENT_TYPE, std::size_t MATRICES = 1, typename WEIGHT_TYPE = double, int MY_C = 64, int MY_SIGMA = 1>
class UnstructuredGrid : public GridBase<ELEMENT_TYPE, 1, WEIGHT_TYPE>
{
public:
    friend class ReorderingUnstructuredGridTest;

    typedef WEIGHT_TYPE WeightType;
    typedef typename GridBase<ELEMENT_TYPE, 1, WEIGHT_TYPE>::SparseMatrix SparseMatrix;
    typedef std::vector<std::pair<ELEMENT_TYPE, WEIGHT_TYPE> > NeighborList;
    typedef typename std::vector<std::pair<ELEMENT_TYPE, WEIGHT_TYPE> >::iterator NeighborListIterator;
    typedef ELEMENT_TYPE StorageType;

    const static int DIM = 1;
    const static int SIGMA = MY_SIGMA;
    const static int C = MY_C;

    explicit UnstructuredGrid(
        const Coord<DIM>& dim = Coord<DIM>(),
        const ELEMENT_TYPE& defaultElement = ELEMENT_TYPE(),
        const ELEMENT_TYPE& edgeElement = ELEMENT_TYPE(),
        const Coord<DIM>& /* topological dimension is irrelevant here */ = Coord<DIM>()) :
        elements(static_cast<unsigned>(dim.x()), defaultElement),
        origin(0),
        edgeElement(edgeElement),
        dimension(dim)
    {
        for (std::size_t i = 0; i < MATRICES; ++i) {
            std::size_t size = static_cast<std::size_t>(dim.x());
            matrices[i] = SellCSigmaSparseMatrixContainer<WEIGHT_TYPE, C, SIGMA>(size);
        }
    }

    explicit
    UnstructuredGrid(
        const CoordBox<DIM> box,
        const ELEMENT_TYPE& defaultElement = ELEMENT_TYPE(),
        const ELEMENT_TYPE& edgeElement = ELEMENT_TYPE(),
        const Coord<DIM>& /* topological dimension is irrelevant here */ = Coord<DIM>()) :
        elements(static_cast<std::size_t>(box.dimensions.x()), defaultElement),
        origin(box.origin.x()),
        edgeElement(edgeElement),
        dimension(box.dimensions)
    {
        for (std::size_t i = 0; i < MATRICES; ++i) {
            std::size_t size = static_cast<std::size_t>(box.origin.x() + dimension.x());
            matrices[i] = SellCSigmaSparseMatrixContainer<WEIGHT_TYPE, C, SIGMA>(size);
        }
    }

    /**
     * Return a pointer to the underlying data storage. Use with care!
     */
    inline
    ELEMENT_TYPE *data()
    {
        return elements.data();
    }

    /**
     * Return a const pointer to the underlying data storage. Use with
     * care!
     */
    inline
    const ELEMENT_TYPE *data() const
    {
        return elements.data();
    }

    void setWeights(std::size_t matrixID, const SparseMatrix& matrix)
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

    inline const Coord<DIM>& getDimensions() const
    {
        return dimension;
    }

    inline const ELEMENT_TYPE& operator[](const int i) const
    {
        int y = i - origin;

        if (y < 0 || y >= dimension.x()) {
            return getEdgeElement();
        }

        return elements[static_cast<unsigned>(y)];
    }

    inline ELEMENT_TYPE& operator[](const int i)
    {
        int y = i - origin;

        if (y < 0 || y >= dimension.x()) {
            return getEdgeElement();
        }

        return elements[static_cast<unsigned>(y)];
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

        if (matrices != other.matrices) {
            return false;
        }

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

    inline void resize(const CoordBox<DIM>& newDim)
    {
        const ELEMENT_TYPE defaultElement = elements.size() ? elements[0] : edgeElement;

        *this = UnstructuredGrid(
            newDim.dimensions,
            defaultElement,
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
        return CoordBox<DIM>(Coord<DIM>(origin), dimension);
    }

    inline void saveRegion(std::vector<ELEMENT_TYPE> *buffer, const Region<DIM>& region, const Coord<1>& offset = Coord<DIM>()) const
    {
        saveRegion(
            buffer,
            region.beginStreak(offset),
            region.endStreak(offset),
            region.size());
    }

    template<typename ITER1, typename ITER2>
    inline void saveRegion(std::vector<ELEMENT_TYPE> *buffer, const ITER1& start, const ITER2& end, std::size_t size) const
    {
        if (size > buffer->size()) {
            throw std::invalid_argument(
                "insufficient buffer size for UnstructuredGrid::saveRegion (have: " +
                StringOps::itoa(buffer->size()) +
                ", need at least: " +
                StringOps::itoa(size) +
                ")");
        }
        ELEMENT_TYPE *target = buffer->data();

        for (ITER1 i = start; i != end; ++i) {
            get(*i, target);
            target += i->length();
        }
    }

    inline void loadRegion(const std::vector<ELEMENT_TYPE>& buffer, const Region<DIM>& region, const Coord<1>& offset = Coord<DIM>())
    {
        loadRegion(
            buffer,
            region.beginStreak(offset),
            region.endStreak(offset),
            region.size());
    }

    template<typename ITER1, typename ITER2>
    inline void loadRegion(const std::vector<ELEMENT_TYPE>& buffer, const ITER1& start, const ITER2& end, std::size_t size)
    {
        if (size > buffer.size()) {
            throw std::invalid_argument(
                "insufficient buffer size for UnstructuredGrid::saveRegion (have: " +
                StringOps::itoa(buffer.size()) +
                ", need at least: " +
                StringOps::itoa(size) +
                ")");
        }

        const ELEMENT_TYPE *source = buffer.data();

        for (ITER1 i = start; i != end; ++i) {
            set(*i, source);
            source += i->length();
        }
    }

    template<typename ITER1, typename ITER2>
    void saveMemberImplementation(
        char *target,
        MemoryLocation::Location targetLocation,
        const Selector<ELEMENT_TYPE>& selector,
        const ITER1& start,
        const ITER2& end) const
    {
        for (ITER1 i = start; i != end; ++i) {
            selector.copyMemberOut(
                &(*this)[i->origin],
                MemoryLocation::HOST,
                target,
                targetLocation,
                static_cast<std::size_t>(i->length()));
            target += selector.sizeOfExternal() * i->length();
        }
    }

    template<typename ITER1, typename ITER2>
    void loadMemberImplementation(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<ELEMENT_TYPE>& selector,
        const ITER1& start,
        const ITER2& end)
    {
        for (ITER1 i = start; i != end; ++i) {
            selector.copyMemberIn(
                source,
                sourceLocation,
                &(*this)[i->origin],
                MemoryLocation::HOST,
                static_cast<std::size_t>(i->length()));
            source += selector.sizeOfExternal() * i->length();
        }
    }

protected:
    void saveMemberImplementation(
        char *target,
        MemoryLocation::Location targetLocation,
        const Selector<ELEMENT_TYPE>& selector,
        const Region<DIM>& region) const
    {
        saveMemberImplementation(target, targetLocation, selector, region.beginStreak(), region.endStreak());
    }

    void loadMemberImplementation(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<ELEMENT_TYPE>& selector,
        const Region<DIM>& region)
    {
        loadMemberImplementation(source, sourceLocation, selector, region.beginStreak(), region.endStreak());
    }

private:
    std::vector<ELEMENT_TYPE> elements;
    int origin;
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
