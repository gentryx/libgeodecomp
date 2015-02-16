#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDGRID_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDGRID_H

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
#include <list>
#include <utility>


namespace LibGeoDecomp {

/**
 * A unstructuredgrid for irregular structures
 */
template<typename ELEMENT_TYPE, size_t MATRICES=1,
         typename VALUE_TYPE=double, int C=64, int SIGMA=1>
class UnstructuredGrid : public GridBase<ELEMENT_TYPE, 1>
{
public:
    typedef std::vector<std::pair<ELEMENT_TYPE, VALUE_TYPE> > NeighborList;
    typedef typename std::vector<std::pair<ELEMENT_TYPE, VALUE_TYPE> >::iterator NeighborListIterator;
    const static int DIM = 1;

    explicit UnstructuredGrid(
        const Coord<DIM>& dim = Coord<DIM>(),
        const ELEMENT_TYPE& defaultElement = ELEMENT_TYPE(),
        const ELEMENT_TYPE& edgeElement = ELEMENT_TYPE()) :
        elements(dim.x(), defaultElement),
        edgeElement(edgeElement),
        dimension(dim)
    {
        for (size_t i=0; i < MATRICES; ++i) {
            matrices[i] =
                SellCSigmaSparseMatrixContainer<VALUE_TYPE,C,SIGMA> (dim.x());
        }
    }

    template<typename O_ELEMENT_TYPE>
    UnstructuredGrid<ELEMENT_TYPE, MATRICES, VALUE_TYPE, C, SIGMA>&
    operator=(const UnstructuredGrid<O_ELEMENT_TYPE, MATRICES, VALUE_TYPE,
                                     C, SIGMA> & other)
    {
        elements = other.elements;
        edgeElement = other.edgeElement;
        dimension = other.dimension;

        for (size_t i=0; i<MATRICES; ++i){
            matrices[i] = other.matrices[i];
        }

        return *this;
    }

    /**
     * iterator musst be an interator over pair< Coord<2>, VALUE_TYPE >
     */
    template<typename ITERATOR>
    void setAdjacency(size_t const  matrixID, ITERATOR start,
                      const ITERATOR end){
        if (matrixID >= MATRICES){
            throw std::invalid_argument("matrixID not available");
        }
        for (ITERATOR i = start; i != end; ++i) {

            Coord<2> c = i->first;
            matrices[matrixID].addPoint(c.x(), c.y(), i->second);
        }
    }

    const SellCSigmaSparseMatrixContainer<VALUE_TYPE,C,SIGMA> &
    getAdjacency(size_t const matrixID) const {
        if (matrixID >= MATRICES){
            throw std::invalid_argument("matrixID not available");
        }

        return matrices[matrixID];
    }

    SellCSigmaSparseMatrixContainer<VALUE_TYPE,C,SIGMA> &
    getAdjacency(size_t const matrixID){
        if (matrixID >= MATRICES){
            throw std::invalid_argument("matrixID not available");
        }

        return matrices[matrixID];
    }
    /**
     * returns a list of pairs representing the neighborhood of center element.
     * the first element of the pair is the ELEMENT_TYPE
     * and the secound the ADJACENCY_TYPE.
     * The first Element of the list is the center element it selfe.
     * if center element does not exist the EdggeElement ist reurnt.
     * In bothe cases ADJACENCY_TYPE = -1
     */
    inline NeighborList getNeighborhood(const Coord<DIM>& center) const
    {
        NeighborList neighborhood;

        if (boundingBox().inBounds(center)) {
            neighborhood.push_back( std::make_pair( *this[center], -1 ) );
            std::vector< std::pair<int, VALUE_TYPE> > neighbor =
                matrices[0].getRow(center.x());

            for (NeighborListIterator it = neighbor.begin();
                 it != neighbor.end();
                 ++it) {
                neighborhood.push_back(std::make_pair((*this)[it->first], it->second));
            }

        } else {
            neighborhood.push_back( std::make_pair( getEdgeElement(), -1) );
        }

        return neighborhood;
    }

    inline const Coord<DIM>& getDimensions() const
    {
        return dimension;
    }

    inline const ELEMENT_TYPE& operator[](const int y) const
    {
        if ( y < 0 || y >= dimension.x()){
            return getEdgeElement();
        } else {
            return elements[y];
        }
    }

    inline ELEMENT_TYPE& operator[](const int y)
    {
        if ( y < 0 || y >= dimension.x()){
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

        for (int i = 0; i < MATRICES; ++i) {
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

    inline bool operator!=(const UnstructuredGrid& other) const
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
        message << "Unstructured Grid <" << DIM << ">(" << dimension.x() << ")\n"
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
        (*this)[coord] = element;
    }

    virtual void set(const Streak<DIM>& streak, const ELEMENT_TYPE *element)
    {
        for (Coord<DIM> cursor = streak.origin; cursor.x() < streak.endX; ++cursor.x()) {
            (*this)[cursor] = *element;
            ++element;
        }
    }

    virtual ELEMENT_TYPE get(const Coord<DIM>& coord) const
    {
        return (*this)[coord];
    }

    virtual void get(const Streak<DIM>& streak, ELEMENT_TYPE *element) const
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
        return CoordBox<DIM>( Coord<DIM>(), dimension );
    }

protected:
    void saveMemberImplementation(
        char *target, const Selector<ELEMENT_TYPE>& selector, const Region<DIM>& region) const
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak();
                i != region.endStreak(); ++i) {
            selector.copyMemberOut(&(*this)[i->origin], target, i->length());
            target += selector.sizeOfExternal() * i->length();
        }
    }

    void loadMemberImplementation(
        const char *source, const Selector<ELEMENT_TYPE>& selector,
        const Region<DIM>& region)
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak();
                i != region.endStreak(); ++i) {
            selector.copyMemberIn(source, &(*this)[i->origin], i->length());
            source += selector.sizeOfExternal() * i->length();
        }
    }


private:
    std::vector<ELEMENT_TYPE> elements;
    // TODO wrapper for different types of sell c sigma containers
    SellCSigmaSparseMatrixContainer<VALUE_TYPE, C, SIGMA>  matrices[MATRICES];
    ELEMENT_TYPE edgeElement;
    Coord<DIM> dimension;

};

template<typename _CharT, typename _Traits, typename ELEMENT_TYPE, size_t MATRICES, typename VALUE_TYPE, int C, int SIGMA>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const LibGeoDecomp::UnstructuredGrid<ELEMENT_TYPE, MATRICES, VALUE_TYPE, C, SIGMA>& grid)
{
    __os << grid.toString();
    return __os;
}

}

#endif
