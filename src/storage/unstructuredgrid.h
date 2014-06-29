#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDGRID_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDGRID_H

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/streak.h>
#include <libgeodecomp/storage/gridbase.h>
#include <libgeodecomp/storage/selector.h>
#include <libgeodecomp/storage/sellcsigmasparsematrixcontainer.h>

#include <iostream>
#include <vector>


namespace LibGeoDecomp {

/**
 * A unstructuredgrid for irregular structures
 */
template<typename ELEMENT_TYPE>
class UnstructuredGrid : public GridBase<ELEMENT_TYPE, 1>
{
public:
    explicit UnstructuredGrid(/*TODO*/{}

    typedef Topologies::Cube<1>::Topology Topology;

    inline ELEMENT_TYPE& operator[](const Coord<DIM>& coord)
    {
        return Topology::locate(*this, coord);
    }

    inline const ELEMENT_TYPE& operator[](const Coord<DIM>& coord) const
    {
        return (const_cast<Grid&>(*this))[coord];
    }

    inline const std::vector<ELEMENT_TYPE> operator[](const int y) const
    {
        return elements[y];
    }

    inline std::vector<ELEMENT_TYPE> operator[](const int y)
    {
        return elements[y];
    }





    virtual void set(const Coord<DIM>& coord, const ELEMENT_TYPE& element)
    {
        (*this)[coord] = element;
    }

    virtual void set(const Streak<DIM>& streak, const ELEMENT_TYPE *element)
    {
        Coord<DIM> cursor = streak.origin;
        for (; cursor.x() < streak.endX; ++cursor.x()) {
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

    virtual void setEdge(const ELEMENT_TYPE& cell)
    {
        getEdgeElement() = cell;
    }

    virtual const ELEMENT_TYPE& getEdge() const
    {
        return getEdgeElement();
    }

    virtual CoordBox<DIM> boundingBox() const
    {
        return CoordBox<DIM>( Coord<DIM>(), Coord<DIM>(elements.size()) ); //TODO dimension inclusive or exclusive?
    }

private:
    std::vector<ELEMENT_TYPE> elements;
    std::vector< SellCSigmaSparseMatrixContainer<64,1> > adjazenzMatrices  // TODO wie rausfinden auf welche hardware es läuft
    CELL_TYPE edgeElement;

};

}

#endif
