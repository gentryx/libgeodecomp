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

    static CoordBox<DIM> paddedBoundingBox(const Coord<DIM>& dim)
    {
        // the grid size should be padded to the total number of chunks
        // -> no border cases for vectorization
        const std::size_t rowsPadded = ((dim.x() - 1) / C + 1) * C;

        return CoordBox<DIM>(Coord<DIM>(), Coord<1>(rowsPadded));
    }

    explicit
    UnstructuredSoAGrid(
        const Coord<DIM> dim = Coord<DIM>(11),
        const ELEMENT_TYPE& defaultElement = ELEMENT_TYPE(),
        const ELEMENT_TYPE& edgeElement = ELEMENT_TYPE(),
        const Coord<DIM>& topologicalDimensionIsIrrelevantHere = Coord<DIM>()) :
        elements(paddedBoundingBox(dim), defaultElement, edgeElement),
        dimension(dim)
    {
        // init matrices
        for (std::size_t i = 0; i < MATRICES; ++i) {
            matrices[i] =
                SellCSigmaSparseMatrixContainer<WEIGHT_TYPE, C, SIGMA>(dim.x());
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

        if (elements != other.elements) {
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

        if (getEdge() != other.getEdge()) {
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
        if (newDim.origin.x() > 0) {
            throw std::logic_error("UnstructuredSoAGrid::resize() called with non-zero offset");
        }

        *this = UnstructuredSoAGrid(
            newDim.dimensions,
            elements.get(Coord<1>()),
            elements.getEdge());
    }

    inline std::string toString() const
    {
        std::ostringstream message;
        message << "Unstructured Grid SoA<" << DIM << ">(" << boundingBox().dimensions.x() << ")\n"
                << "boundingBox: " << boundingBox()  << "\n"
                << "edgeElement: " << getEdge();

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
        elements.set(coord, element);
    }

    inline void set(const Streak<DIM>& streak, const ELEMENT_TYPE *cells)
    {
        elements.set(streak, cells);
    }

    inline ELEMENT_TYPE get(const Coord<DIM>& coord) const
    {
        return elements.get(coord);
    }

    inline void get(const Streak<DIM>& streak, ELEMENT_TYPE *cells) const
    {
        elements.get(streak, cells);
    }

    inline void setEdge(const ELEMENT_TYPE& element)
    {
        elements.setEdge(element);
    }

    inline const ELEMENT_TYPE& getEdge() const
    {
        return elements.getEdge();
    }

    inline CoordBox<DIM> boundingBox() const
    {
        return CoordBox<DIM>(Coord<DIM>(), dimension);
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
        saveRegionImplementation(target, region.beginStreak(offset), region.endStreak(offset), region.size());
    }

    template<typename ITER1, typename ITER2>
    inline void saveRegionImplementation(std::vector<char> *target, const ITER1& start, const ITER2& end, int size) const
    {
        elements.saveRegionImplementation(target, start, end, size);
    }

    inline void loadRegion(const std::vector<char>& source, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>())
    {
        loadRegionImplementation(source, region.beginStreak(offset), region.endStreak(offset), region.size());
    }

    template<typename ITER1, typename ITER2>
    inline void loadRegionImplementation(const std::vector<char>& source, const ITER1& start, const ITER2& end, int size)
    {
        elements.loadRegionImplementation(source, start, end, size);
    }

    template<typename ITER1, typename ITER2>
    void saveMemberImplementationGeneric(
        char *target,
        MemoryLocation::Location targetLocation,
        const Selector<ELEMENT_TYPE>& selector,
        const ITER1& start,
        const ITER2& end) const
    {
        elements.saveMemberImplementationGeneric(
            target,
            targetLocation,
            selector,
            start,
            end);
    }

    template<typename ITER1, typename ITER2>
    void loadMemberImplementationGeneric(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<ELEMENT_TYPE>& selector,
        const ITER1& start,
        const ITER2& end)
    {
        elements.loadMemberImplementationGeneric(
            source,
            sourceLocation,
            selector,
            start,
            end);
    }

protected:
    void saveMemberImplementation(
        char *target,
        MemoryLocation::Location targetLocation,
        const Selector<ELEMENT_TYPE>& selector,
        const typename Region<DIM>::StreakIterator& begin,
        const typename Region<DIM>::StreakIterator& end) const
    {
        saveMemberImplementationGeneric(target, targetLocation, selector, begin, end);
    }

    void loadMemberImplementation(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<ELEMENT_TYPE>& selector,
        const typename Region<DIM>::StreakIterator& begin,
        const typename Region<DIM>::StreakIterator& end)
    {
        loadMemberImplementationGeneric(source, sourceLocation, selector, begin, end);
    }

private:
    SoAGrid<ELEMENT_TYPE, Topologies::Torus<1>::Topology> elements;
    SellCSigmaSparseMatrixContainer<WEIGHT_TYPE, C, SIGMA> matrices[MATRICES];
    Coord<DIM> dimension;

    inline
    ELEMENT_TYPE get(int x) const
    {
        return elements.get(Coord<1>(x));
    }

    inline
    void set(int x, const ELEMENT_TYPE& cell)
    {
        elements.set(Coord<1>(x), cell);
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
