#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDSOAGRID_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDSOAGRID_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

#include <libflatarray/flat_array.hpp>

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
#include <utility>
#include <cassert>

namespace LibGeoDecomp {

namespace UnstructuredSoAGridHelpers {

/**
 * Internal helper class to save a region of a cell member.
 */
template<typename CELL, int DIM>
class SaveMember
{
public:
    SaveMember(
        char *target,
        const Selector<CELL>& selector,
        const Region<DIM>& region) :
        target(target),
        selector(selector),
        region(region)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor) const
    {
        char *currentTarget = target;

        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            accessor.index = i->origin.x();
            const char *data = accessor.access_member(selector.sizeOfMember(), selector.offset());
            selector.copyStreakOut(data, currentTarget, i->length(), DIM_X);
            currentTarget += selector.sizeOfExternal() * i->length();
        }
    }

private:
    char *target;
    const Selector<CELL>& selector;
    const Region<DIM>& region;
};

/**
 * Internal helper class to load a region of a cell member.
 */
template<typename CELL, int DIM>
class LoadMember
{
public:
    LoadMember(
        const char *source,
        const Selector<CELL>& selector,
        const Region<DIM>& region) :
        source(source),
        selector(selector),
        region(region)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor) const
    {
        const char *currentSource = source;

        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            accessor.index = i->origin.x();
            char *currentTarget = accessor.access_member(selector.sizeOfMember(), selector.offset());
            selector.copyStreakIn(currentSource, currentTarget, i->length(), DIM_X);
            currentSource += selector.sizeOfExternal() * i->length();
        }
    }

private:
    const char *source;
    const Selector<CELL>& selector;
    const Region<DIM>& region;
};

}

/**
 * A unstructured grid for irregular structures using SoA memory layout.
 */
template<typename ELEMENT_TYPE, std::size_t MATRICES = 1,
         typename VALUE_TYPE = double, int C = 64, int SIGMA = 1>
class UnstructuredSoAGrid : public GridBase<ELEMENT_TYPE, 1>
{
public:
    const static int DIM = 1;

    explicit UnstructuredSoAGrid(
        const Coord<DIM>& dim = Coord<DIM>(),
        const ELEMENT_TYPE& defaultElement = ELEMENT_TYPE(),
        const ELEMENT_TYPE& edgeElement = ELEMENT_TYPE()) :
        elements(dim.x(), 1, 1),
        edgeElement(edgeElement),
        dimension(dim)
    {
        // init matrices
        for (std::size_t i = 0; i < MATRICES; ++i) {
            matrices[i] =
                SellCSigmaSparseMatrixContainer<VALUE_TYPE,C,SIGMA>(dim.x());
        }

        // the grid size should be padded to the total number of chunks
        // -> no border cases for vectorization
        const std::size_t rowsPadded = ((dim.x() - 1) / C + 1) * C;
        elements.resize(rowsPadded, 1, 1);

        // init soa_grid
        for (std::size_t i = 0; i < rowsPadded; ++i) {
            set(i, defaultElement);
        }
    }

    explicit
    UnstructuredSoAGrid(const CoordBox<DIM> box,
                        const ELEMENT_TYPE& defaultElement = ELEMENT_TYPE(),
                        const ELEMENT_TYPE& edgeElement = ELEMENT_TYPE()) :
        elements(box.dimensions.x(), 1, 1),
        edgeElement(edgeElement),
        dimension(box.dimensions)
    {
        // init matrices
        for (std::size_t i = 0; i < MATRICES; ++i) {
            matrices[i] =
                SellCSigmaSparseMatrixContainer<VALUE_TYPE,C,SIGMA>(dimension.x());
        }

        // the grid size should be padded to the total number of chunks
        // -> no border cases for vectorization
        const std::size_t rowsPadded = ((dimension.x() - 1) / C + 1) * C;
        elements.resize(rowsPadded, 1, 1);

        // init soa_grid
        for (std::size_t i = 0; i < rowsPadded; ++i) {
            set(i, defaultElement);
        }
    }

    template<typename O_ELEMENT_TYPE>
    UnstructuredSoAGrid<ELEMENT_TYPE, MATRICES, VALUE_TYPE, C, SIGMA>&
    operator=(const UnstructuredSoAGrid<O_ELEMENT_TYPE, MATRICES, VALUE_TYPE, C, SIGMA>& other)
    {
        elements    = other.elements;
        edgeElement = other.edgeElement;
        dimension   = other.dimension;

        for (std::size_t i = 0; i < MATRICES; ++i) {
            matrices[i] = other.matrices[i];
        }

        return *this;
    }

    inline
    void setCompleteAdjacency(std::size_t matrixID, std::size_t matrixSize,
                              const std::map<Coord<2>, VALUE_TYPE>& matrix)
    {
        assert(matrixID < MATRICES);
        matrices[matrixID].initFromMatrix(matrixSize, matrix);
    }

    /**
     * iterator musst be an interator over pair< Coord<2>, VALUE_TYPE >
     */
    template<typename ITERATOR>
    void setAdjacency(std::size_t const  matrixID, const ITERATOR& start,
                      const ITERATOR& end)
    {
        assert(matrixID < MATRICES);
        for (ITERATOR i = start; i != end; ++i) {
            Coord<2> c = i->first;
            matrices[matrixID].addPoint(c.x(), c.y(), i->second);
        }
    }

    inline
    const SellCSigmaSparseMatrixContainer<VALUE_TYPE, C, SIGMA>&
    getAdjacency(std::size_t const matrixID) const
    {
        assert(matrixID < MATRICES);
        return matrices[matrixID];
    }

    inline
    SellCSigmaSparseMatrixContainer<VALUE_TYPE, C, SIGMA>&
    getAdjacency(std::size_t const matrixID)
    {
        assert(matrixID < MATRICES);
        return matrices[matrixID];
    }

    inline const Coord<DIM>& getDimensions() const
    {
        return dimension;
    }

    inline const ELEMENT_TYPE operator[](const int y) const
    {
        if (y < 0 || y >= dimension.x()) {
            return getEdgeElement();
        } else {
            return get(y);
        }
    }

    inline ELEMENT_TYPE operator[](const Coord<DIM>& coord) const
    {
        return (*this)[coord.x()];
    }

    inline bool operator==(const UnstructuredSoAGrid& other) const
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
        for (typename CoordBox<DIM>::Iterator i = box.begin();
             i != box.end(); ++i) {
            if ((*this)[*i] != other.get(*i)) {
                return false;
            }
        }

        return true;
    }

    inline bool operator!=(const UnstructuredSoAGrid& other) const
    {
        return !(*this == other);
    }

    inline bool operator!=(const GridBase<ELEMENT_TYPE, DIM>& other) const
    {
        return !(*this == other);
    }

    inline std::string toString() const
    {
        std::ostringstream message;
        message << "Unstructured Grid SoA<" << DIM << ">(" << dimension.x() << ")\n"
                << "boundingBox: " << boundingBox()  << "\n"
                << "edgeElement: " << edgeElement;

        CoordBox<DIM> box = boundingBox();
        int index = 0;
        for (typename CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            message << "\nCoord " << *i << ":\n"
                    << (*this)[*i] << "\n"
                    << "neighbor: ";

            std::vector<std::pair<int, VALUE_TYPE> > neighbor =
                matrices[0].getRow(index++);
            message << neighbor;
        }

        message << "\n";
        return message.str();
    }

    virtual void set(const Coord<DIM>& coord, const ELEMENT_TYPE& element)
    {
        set(coord.x(), element);
    }

    virtual void set(const Streak<DIM>& streak, const ELEMENT_TYPE *cells)
    {
        elements.set(streak.origin.x(), 0, 0, cells, streak.length());
    }

    virtual ELEMENT_TYPE get(const Coord<DIM>& coord) const
    {
        return get(coord.x());
    }

    virtual void get(const Streak<DIM>& streak, ELEMENT_TYPE *cells) const
    {
        elements.get(streak.origin.x(), 0, 0, cells, streak.length());
    }

    inline ELEMENT_TYPE& getEdgeElement()
    {
        return edgeElement;
    }

    inline const ELEMENT_TYPE& getEdgeElement() const
    {
        return edgeElement;
    }

    virtual void setEdge(const ELEMENT_TYPE& element)
    {
        getEdgeElement() = element;
    }

    virtual const ELEMENT_TYPE& getEdge() const
    {
        return getEdgeElement();
    }

    virtual CoordBox<DIM> boundingBox() const
    {
        return CoordBox<DIM>(Coord<DIM>(), dimension);
    }

    template<typename FUNCTOR>
    void callback(FUNCTOR functor) const
    {
        elements.callback(functor);
    }

protected:
    void saveMemberImplementation(
        char *target, const Selector<ELEMENT_TYPE>& selector, const Region<DIM>& region) const
    {
        elements.callback(
            UnstructuredSoAGridHelpers::SaveMember<ELEMENT_TYPE, DIM>(
                target, selector, region));
    }

    void loadMemberImplementation(
        const char *source, const Selector<ELEMENT_TYPE>& selector,
        const Region<DIM>& region)
    {
        elements.callback(
            UnstructuredSoAGridHelpers::LoadMember<ELEMENT_TYPE, DIM>(
                source, selector, region));
    }

private:
    inline ELEMENT_TYPE get(int x) const
    {
        assert(x >= 0);
        return elements.get(x, 0, 0);
    }

    inline void set(int x, const ELEMENT_TYPE& cell)
    {
        assert(x >= 0);
        elements.set(x, 0, 0, cell);
    }

    LibFlatArray::soa_grid<ELEMENT_TYPE> elements;
    // TODO wrapper for different types of sell c sigma containers
    SellCSigmaSparseMatrixContainer<VALUE_TYPE, C, SIGMA> matrices[MATRICES];
    ELEMENT_TYPE edgeElement;
    Coord<DIM> dimension;
};

template<typename ELEMENT_TYPE, std::size_t MATRICES, typename VALUE_TYPE, int C, int SIGMA>
inline
std::ostream& operator<<(std::ostream& os,
                         const UnstructuredSoAGrid<ELEMENT_TYPE, MATRICES, VALUE_TYPE, C, SIGMA>& grid)
{
    os << grid.toString();
    return os;
}

}

#endif
#endif
