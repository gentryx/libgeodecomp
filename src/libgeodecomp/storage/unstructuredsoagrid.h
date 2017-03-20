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
#include <libgeodecomp/storage/soagrid.h>

#include <iostream>
#include <vector>
#include <utility>
#include <cassert>

namespace LibGeoDecomp {

namespace UnstructuredSoAGridHelpers {

/**
 * Internal helper class to save a region of a cell member.
 */
template<typename CELL, int DIM, typename ITER1, typename ITER2>
class SaveMember
{
public:
    SaveMember(
        char *target,
        MemoryLocation::Location targetLocation,
        const Selector<CELL>& selector,
        const ITER1& start,
        const ITER2& end) :
        target(target),
        targetLocation(targetLocation),
        selector(selector),
        start(start),
        end(end)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor) const
    {
        char *currentTarget = target;

        for (auto i = start; i != end; ++i) {
            accessor.index() = i->origin.x();
            const char *data = accessor.access_member(selector.sizeOfMember(), selector.offset());
            selector.copyStreakOut(
                data,
                MemoryLocation::HOST,
                currentTarget,
                targetLocation,
                static_cast<std::size_t>(i->length()),
                DIM_X);
            currentTarget += selector.sizeOfExternal() * i->length();
        }
    }

private:
    char *target;
    MemoryLocation::Location targetLocation;
    const Selector<CELL>& selector;
    const ITER1& start;
    const ITER2& end;
};

/**
 * Internal helper class to load a region of a cell member.
 */
template<typename CELL, int DIM, typename ITER1, typename ITER2>
class LoadMember
{
public:
    LoadMember(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<CELL>& selector,
        const ITER1& start,
        const ITER2& end) :
        source(source),
        sourceLocation(sourceLocation),
        selector(selector),
        start(start),
        end(end)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor) const
    {
        const char *currentSource = source;

        for (auto i = start; i != end; ++i) {
            accessor.index() = i->origin.x();
            char *currentTarget = accessor.access_member(selector.sizeOfMember(), selector.offset());
            selector.copyStreakIn(
                currentSource,
                sourceLocation,
                currentTarget,
                MemoryLocation::HOST,
                static_cast<std::size_t>(i->length()),
                DIM_X);
            currentSource += selector.sizeOfExternal() * i->length();
        }
    }

private:
    const char *source;
    MemoryLocation::Location sourceLocation;
    const Selector<CELL>& selector;
    const ITER1& start;
    const ITER2& end;
};

}

/**
 * A unstructured grid for irregular structures using SoA memory layout.
 */
template<
    typename ELEMENT_TYPE,
    std::size_t MATRICES = 1,
    typename WEIGHT_TYPE = double,
    int MY_C = 64,
    int MY_SIGMA = 1>
class UnstructuredSoAGrid : public GridBase<ELEMENT_TYPE, 1>
{
public:
    friend class ReorderingUnstructuredGridTest;

    using GridBase<ELEMENT_TYPE, 1>::saveRegion;
    using GridBase<ELEMENT_TYPE, 1>::loadRegion;

    typedef typename GridBase<ELEMENT_TYPE, 1>::SparseMatrix SparseMatrix;
    typedef WEIGHT_TYPE WeightType;
    typedef char StorageType;
    const static int DIM = 1;
    const static int SIGMA = MY_SIGMA;
    const static int C = MY_C;

    static const int AGGREGATED_MEMBER_SIZE =  LibFlatArray::aggregated_member_size<ELEMENT_TYPE>::VALUE;

    explicit
    UnstructuredSoAGrid(
        const CoordBox<DIM> box = CoordBox<DIM>(Coord<DIM>(), Coord<DIM>(1)),
        const ELEMENT_TYPE& defaultElement = ELEMENT_TYPE(),
        const ELEMENT_TYPE& edgeElement = ELEMENT_TYPE(),
        const Coord<DIM>& topologicalDimensionIsIrrelevantHere = Coord<DIM>()) :
        elements(
            static_cast<std::size_t>(box.dimensions.x()),
            static_cast<std::size_t>(1),
            static_cast<std::size_t>(1)),
        origin(box.origin.x()),
        edgeElement(edgeElement),
        dimension(box.dimensions)
    {
        // init matrices
        for (std::size_t i = 0; i < MATRICES; ++i) {
            matrices[i] =
                SellCSigmaSparseMatrixContainer<WEIGHT_TYPE, C, SIGMA>(dimension.x());
        }

        // the grid size should be padded to the total number of chunks
        // -> no border cases for vectorization
        const std::size_t rowsPadded = static_cast<std::size_t>(((dimension.x() - 1) / C + 1) * C);
        elements.resize(rowsPadded, 1, 1);

        // init soa_grid
        for (std::size_t i = 0; i < rowsPadded; ++i) {
            set(i, defaultElement);
        }
    }

    /**
     * Return a pointer to the underlying data storage. Use with care!
     */
    inline
    char *data()
    {
        return elements.data();
    }

    /**
     * Return a const pointer to the underlying data storage. Use with
     * care!
     */
    inline
    const char *data() const
    {
        return elements.data();
    }

    inline
    void setWeights(std::size_t matrixID, const SparseMatrix& matrix)
    {
        assert(matrixID < MATRICES);
        matrices[matrixID].initFromMatrix(matrix);
    }

    inline
    const SellCSigmaSparseMatrixContainer<WEIGHT_TYPE, C, SIGMA>& getWeights(std::size_t const matrixID) const
    {
        assert(matrixID < MATRICES);
        return matrices[matrixID];
    }

    inline
    SellCSigmaSparseMatrixContainer<WEIGHT_TYPE, C, SIGMA>& getWeights(std::size_t const matrixID)
    {
        assert(matrixID < MATRICES);
        return matrices[matrixID];
    }

    inline const Coord<DIM>& getDimensions() const
    {
        return dimension;
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

    inline void resize(const CoordBox<DIM>& newDim)
    {
        *this = UnstructuredSoAGrid(
            newDim,
            elements.get(0, 0, 0),
            edgeElement);
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

            std::vector<std::pair<int, WEIGHT_TYPE> > neighbor =
                matrices[0].getRow(index++);
            message << neighbor;
        }

        message << "\n";
        return message.str();
    }

    inline void set(const Coord<DIM>& coord, const ELEMENT_TYPE& element)
    {
        int index = coord.x() - origin;

        if ((index < 0) || (index >= dimension.x())) {
            edgeElement = element;
            return;
        }

        set(index, element);
    }

    inline void set(const Streak<DIM>& streak, const ELEMENT_TYPE *cells)
    {
        elements.set(streak.origin.x() - origin, 0, 0, cells, streak.length());
    }

    inline ELEMENT_TYPE get(const Coord<DIM>& coord) const
    {
        int index = coord.x() - origin;

        if ((index < 0) || (index >= dimension.x())) {
            return edgeElement;
        }

        return get(index);
    }

    inline void get(const Streak<DIM>& streak, ELEMENT_TYPE *cells) const
    {
        elements.get(streak.origin.x() - origin, 0, 0, cells, streak.length());
    }

    inline ELEMENT_TYPE& getEdgeElement()
    {
        return edgeElement;
    }

    inline const ELEMENT_TYPE& getEdgeElement() const
    {
        return edgeElement;
    }

    inline void setEdge(const ELEMENT_TYPE& element)
    {
        getEdgeElement() = element;
    }

    inline const ELEMENT_TYPE& getEdge() const
    {
        return getEdgeElement();
    }

    inline CoordBox<DIM> boundingBox() const
    {
        return CoordBox<DIM>(Coord<DIM>(origin), dimension);
    }

    template<typename FUNCTOR>
    void callback(FUNCTOR functor) const
    {
        elements.callback(functor);
    }

    template<typename FUNCTOR>
    void callback(UnstructuredSoAGrid *newGrid, FUNCTOR functor) const
    {
        elements.callback(&newGrid->elements, functor);
    }

    inline void saveRegion(std::vector<char> *target, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>()) const
    {
        typedef SoAGridHelpers::OffsetStreakIterator<typename Region<DIM>::StreakIterator, DIM> StreakIteratorType;
        StreakIteratorType start(region.beginStreak(), Coord<3>(offset.x(), 0, 0));
        StreakIteratorType end(  region.endStreak(),   Coord<3>(offset.x(), 0, 0));

        saveRegion(target, start, end, region.size());
    }

    template<typename ITER1, typename ITER2>
    inline void saveRegion(std::vector<char> *target, const ITER1& start, const ITER2& end, std::size_t size) const
    {
        elements.save(start, end, target->data(), size);
    }

    inline void loadRegion(const std::vector<char>& source, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>())
    {
        typedef SoAGridHelpers::OffsetStreakIterator<typename Region<DIM>::StreakIterator, DIM> StreakIteratorType;
        StreakIteratorType start(region.beginStreak(), Coord<3>(offset.x(), 0, 0));
        StreakIteratorType end(  region.endStreak(),   Coord<3>(offset.x(), 0, 0));

        loadRegion(source, start, end, region.size());
    }

    template<typename ITER1, typename ITER2>
    inline void loadRegion(const std::vector<char>& source, const ITER1& start, const ITER2& end, std::size_t size)
    {
        elements.load(start, end, source.data(), size);
    }

    template<typename ITER1, typename ITER2>
    void saveMemberImplementation(
        char *target,
        MemoryLocation::Location targetLocation,
        const Selector<ELEMENT_TYPE>& selector,
        const ITER1& start,
        const ITER2& end) const
    {
        elements.callback(
            UnstructuredSoAGridHelpers::SaveMember<ELEMENT_TYPE, DIM, ITER1, ITER2>(
                target, targetLocation, selector, start, end));
    }

    template<typename ITER1, typename ITER2>
    void loadMemberImplementation(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<ELEMENT_TYPE>& selector,
        const ITER1& start,
        const ITER2& end)
    {
        elements.callback(
            UnstructuredSoAGridHelpers::LoadMember<ELEMENT_TYPE, DIM, ITER1, ITER2>(
                source, sourceLocation, selector, start, end));
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
    LibFlatArray::soa_grid<ELEMENT_TYPE> elements;
    int origin;
    // TODO wrapper for different types of sell c sigma containers
    SellCSigmaSparseMatrixContainer<WEIGHT_TYPE, C, SIGMA> matrices[MATRICES];
    ELEMENT_TYPE edgeElement;
    Coord<DIM> dimension;

    inline
    ELEMENT_TYPE get(int x) const
    {
        return elements.get(x, 0, 0);
    }

    inline
    void set(int x, const ELEMENT_TYPE& cell)
    {
        elements.set(x, 0, 0, cell);
    }
};

template<typename ELEMENT_TYPE, std::size_t MATRICES, typename WEIGHT_TYPE, int C, int SIGMA>
inline
std::ostream& operator<<(std::ostream& os,
                         const UnstructuredSoAGrid<ELEMENT_TYPE, MATRICES, WEIGHT_TYPE, C, SIGMA>& grid)
{
    os << grid.toString();
    return os;
}

}

#endif
#endif
