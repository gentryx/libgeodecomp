#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDGRID_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDGRID_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/streak.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/storage/displacedgrid.h>
#include <libgeodecomp/storage/gridbase.h>
#include <libgeodecomp/storage/selector.h>
#include <libgeodecomp/storage/sellcsigmasparsematrixcontainer.h>

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
#include <libgeodecomp/misc/cudaboostworkaround.h>
#include <libgeodecomp/communication/boostserialization.h>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#endif

#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <cassert>

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
        elements(CoordBox<1>(Coord<1>(), dim), defaultElement, edgeElement)
    {
        for (std::size_t i = 0; i < MATRICES; ++i) {
            matrices[i] = SellCSigmaSparseMatrixContainer<WEIGHT_TYPE, C ,SIGMA>(dim.x());
        }
    }

    explicit
    UnstructuredGrid(
        const CoordBox<DIM> box,
        const ELEMENT_TYPE& defaultElement = ELEMENT_TYPE(),
        const ELEMENT_TYPE& edgeElement = ELEMENT_TYPE(),
        const Coord<DIM>& /* topological dimension is irrelevant here */ = Coord<DIM>()) :
        elements(box, defaultElement, edgeElement)
    {
        // fixme: allocation too large here? (this is allocating an array of size "box.origin.x() + dimension.x()")
        for (std::size_t i = 0; i < MATRICES; ++i) {
            matrices[i] = SellCSigmaSparseMatrixContainer<WEIGHT_TYPE, C, SIGMA>(box.origin.x() + box.dimensions.x());
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
        return elements.getDimensions();
    }

    inline const ELEMENT_TYPE& operator[](const int i) const
    {
        return elements[Coord<1>(i)];
    }

    inline ELEMENT_TYPE& operator[](const int i)
    {
        return elements[Coord<1>(i)];
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

        if (elements.getEdge() != other.getEdge()) {
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
        elements.resize(newDim);

        // fixme: allocation too large here? (this is allocating an array of size "box.origin.x() + dimension.x()")
        for (std::size_t i = 0; i < MATRICES; ++i) {
            matrices[i] = SellCSigmaSparseMatrixContainer<WEIGHT_TYPE, C, SIGMA>(boundingBox().origin.x() + getDimensions().x());
        }
    }

    inline std::string toString() const
    {
        std::ostringstream message;
        message << "Unstructured Grid <" << DIM << ">(" << getDimensions().x() << ")\n"
                << "boundingBox: " << boundingBox()  << "\n"
                << "edgeElement: " << getEdge();

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

    void setEdge(const ELEMENT_TYPE& element)
    {
        elements.setEdge(element);
    }

    const ELEMENT_TYPE& getEdge() const
    {
        return elements.getEdge();
    }

    CoordBox<DIM> boundingBox() const
    {
        return elements.boundingBox();
    }

    inline void saveRegion(std::vector<ELEMENT_TYPE> *buffer, const Region<DIM>& region, const Coord<1>& offset = Coord<DIM>()) const
    {
        elements.saveRegion(buffer, region, offset);
    }

#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
    void saveRegion(
        std::vector<char> *buffer,
        const Region<DIM>& region,
        const Coord<DIM>& offset = Coord<DIM>()) const
    {
        elements.saveRegion(buffer, region, offset);
    }
#endif

    inline void loadRegion(const std::vector<ELEMENT_TYPE>& buffer, const Region<DIM>& region, const Coord<1>& offset = Coord<DIM>())
    {
        elements.loadRegion(buffer, region, offset);
    }

#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
    void loadRegion(
        const std::vector<char>& buffer,
        const Region<DIM>& region,
        const Coord<DIM>& offset = Coord<DIM>())
    {
        elements.loadRegion(buffer, region, offset);
    }
#endif

    template<typename ITER1, typename ITER2>
    void saveRegionImplementation(
        std::vector<ELEMENT_TYPE> *buffer,
        const ITER1& begin,
        const ITER2& end,
        int /* unused: size */ = 0) const
    {
        elements.saveRegionImplementation(buffer, begin, end);
    }

    template<typename ITER1, typename ITER2>
    void loadRegionImplementation(
        const std::vector<ELEMENT_TYPE>& buffer,
        const ITER1& begin,
        const ITER2& end,
        int /* unused: size */ = 0)
    {
        elements.loadRegionImplementation(buffer, begin, end);
    }

    template<typename ITER1, typename ITER2>
    void saveMemberImplementationGeneric(
        char *target,
        MemoryLocation::Location targetLocation,
        const Selector<ELEMENT_TYPE>& selector,
        const ITER1& start,
        const ITER2& end) const
    {
        // fixme: delegate here?
        for (ITER1 i = start; i != end; ++i) {
            selector.copyMemberOut(&(*this)[i->origin], MemoryLocation::HOST, target, targetLocation, i->length());
            target += selector.sizeOfExternal() * i->length();
        }
    }

    template<typename ITER1, typename ITER2>
    void loadMemberImplementationGeneric(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<ELEMENT_TYPE>& selector,
        const ITER1& start,
        const ITER2& end)
    {
        // fixme: delegate here?
        for (ITER1 i = start; i != end; ++i) {
            selector.copyMemberIn(source, sourceLocation, &(*this)[i->origin], MemoryLocation::HOST, i->length());
            source += selector.sizeOfExternal() * i->length();
        }
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
    DisplacedGrid<ELEMENT_TYPE, Topologies::Cube<1>::Topology> elements;
    SellCSigmaSparseMatrixContainer<WEIGHT_TYPE, C, SIGMA> matrices[MATRICES];
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
