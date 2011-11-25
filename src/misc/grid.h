#ifndef _libgeodecomp_misc_grid_h_
#define _libgeodecomp_misc_grid_h_

// CodeGear's C++ compiler isn't compatible with boost::multi_array
// (at least the version that ships with C++ Builder 2009)
#ifndef __CODEGEARC__
#include <boost/multi_array.hpp>
#else
#include <libgeodecomp/misc/supervector.h>
#endif

#include <iostream>

#include <libgeodecomp/misc/alignedallocator.h>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/coordmap.h>
#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/gridbase.h>
#include <libgeodecomp/misc/topologies.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE, typename GRID_TYPE>
class CoordMap;

template<int DIM, typename MATRIX_TYPE, typename CELL_TYPE>
class GridFiller;

/**
 * A multi-dimensional regular grid
 */
template<typename CELL_TYPE, typename TOPOLOGY=Topologies::Cube<2>::Topology>
class Grid : public GridBase<CELL_TYPE, TOPOLOGY::DIMENSIONS>
{
    friend class GridTest;
    friend class ParallelStripingSimulatorTest;
    
public:
    const static int DIM = TOPOLOGY::DIMENSIONS;

#ifndef __CODEGEARC__
    typedef typename boost::detail::multi_array::sub_array<CELL_TYPE, DIM - 1> SliceRef;
    typedef typename boost::detail::multi_array::const_sub_array<CELL_TYPE, 1> ConstSliceRef;
    typedef typename boost::multi_array<
        // always align on cache line boundaries
        CELL_TYPE, DIM, AlignedAllocator<CELL_TYPE, 64> > CellMatrix;
    typedef typename CellMatrix::index Index;
#else
    typedef SuperVector<CELL_TYPE>& SliceRef;
    typedef const SuperVector<CELL_TYPE>& ConstSliceRef;
    typedef int Index;
#endif

    typedef TOPOLOGY Topology;
    typedef CELL_TYPE CellType;
    typedef CoordMap<CELL_TYPE, Grid<CELL_TYPE, TOPOLOGY> > MyCoordMap;

    explicit Grid(const Coord<DIM>& dim=Coord<DIM>(),
         const CELL_TYPE& _defaultCell=CELL_TYPE(),
         const CELL_TYPE& _edgeCell=CELL_TYPE()) :
        dimensions(dim),
        cellMatrix(dim.toExtents()),
        edgeCell(_edgeCell)
    {
        GridFiller<DIM, CellMatrix, CELL_TYPE>()(
            &cellMatrix, _defaultCell);
    }

    ~Grid() {}

    Grid& operator=(const Grid& other)
    {
        resize(other.getDimensions());
        std::copy(other.cellMatrix.begin(), other.cellMatrix.end(), cellMatrix.begin());
        edgeCell = other.edgeCell;
        return *this;
    }

    inline void resize(const Coord<DIM>& newDim)
    {
        // temporarly resize to 0-sized array to avoid having two
        // large arrays simultaneously allocated. somehow I feel
        // boost::multi_array::resize should be responsible for
        // this...
        Coord<DIM> tempDim;
        cellMatrix.resize(tempDim.toExtents());
        dimensions = newDim;
        cellMatrix.resize(newDim.toExtents());
    }
    
    /**
     * returns a map that is referenced by relative coordinates from the
     * originating coordinate coord.
     */
    inline MyCoordMap getNeighborhood(const Coord<DIM>& center)
    {
        return MyCoordMap(center, this);
    }

    inline CELL_TYPE& getEdgeCell()
    {
        return edgeCell;
    }

    inline CELL_TYPE *baseAddress() 
    {
        return &(*this)[Coord<DIM>()];
    }

    inline const CELL_TYPE *baseAddress() const
    {
        return &(*this)[Coord<DIM>()];
    }

    inline const CELL_TYPE& getEdgeCell() const
    {
        return edgeCell;
    }

    inline CELL_TYPE& operator[](const Coord<DIM>& coord)
    {
        return Topology::locate(*this, coord);
    }

    inline const CELL_TYPE& operator[](const Coord<DIM>& coord) const
    {
        return (const_cast<Grid&>(*this))[coord];
    }

    /**
     * WARNING: this operator doesn't honor topology properties
     */
    inline const ConstSliceRef operator[](const Index& y) const
    {
        return cellMatrix[y];
    }

    /**
     * WARNING: this operator doesn't honor topology properties
     */
    inline SliceRef operator[](const Index& y) 
    {
        return cellMatrix[y];
    }

    inline bool operator==(const Grid& other) const
    {
        if (this->boundingBox() == CoordBox<DIM>() && 
            other.boundingBox() == CoordBox<DIM>())
            return true;

        return 
            (edgeCell   == other.edgeCell) && 
            (cellMatrix == other.cellMatrix);
    }

    inline bool operator!=(const Grid& other) const
    {
        return !(*this == other);
    }

    virtual CELL_TYPE& at(const Coord<DIM>& coord)
    {
        return (*this)[coord];
    }

    virtual const CELL_TYPE& at(const Coord<DIM>& coord) const
    {
        return (*this)[coord];
    }

    virtual CELL_TYPE& atEdge()
    {
        return getEdgeCell();
    }

    virtual const CELL_TYPE& atEdge() const
    {
        return getEdgeCell();
    }

    inline const Coord<DIM>& getDimensions() const
    {
        return dimensions;
    }
  
    inline std::string diff(const Grid& other) const
    {
        if (this->boundingBox() != other.boundingBox()) {
            std::ostringstream message;
            message << 
                "dimensions mismatch (is (" << boundingBox() << 
                "), got (" << other.boundingBox() << "))";
            return message.str();
        }

        std::ostringstream message;
        if (edgeCell != other.edgeCell) {
            message << "\nedge cell differs (self (" << edgeCell 
                    << "), other (" << other.edgeCell << "))\n";
        }

        CoordBox<DIM> b = boundingBox();
        for (CoordBoxSequence<DIM> s = b.sequence(); s.hasNext();) {
            Coord<DIM> c = s.next();
            if ((*this)[c] != other[c]) {
                message << "\nat coordinate " << c << "\n" <<
                    "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n" <<
                    (*this)[c] <<
                    "========================================\n" <<
                    other[c] << 
                    ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
            }
        }

        if (message.str() != "") 
            return "cell differences:\n" + message.str();
        else 
            return "";
    }


    inline std::string toString() const
    {
        std::ostringstream message;
        message << "Grid\n"
                << "boundingBox: " << boundingBox() 
                << "edgeCell:\n"
                << edgeCell << "\n";

        for (CoordBoxSequence<DIM> s = 
                 this->boundingBox().sequence(); s.hasNext();) {
            Coord<DIM> coord(s.next());
            message << "Coord" << coord << ":\n" << (*this)[coord] << "\n";
        }

        return message.str();
    }

    virtual CoordBox<DIM> boundingBox() const
    {
        return CoordBox<DIM>(Coord<DIM>(), dimensions);
    }


private:
    Coord<DIM> dimensions;
    CellMatrix cellMatrix;
    // This dummy stores the constant edge constraint
    CELL_TYPE edgeCell;
    
};

template<typename MATRIX_TYPE, typename CELL_TYPE>
class GridFiller<1, MATRIX_TYPE, CELL_TYPE>
{
public:

    void operator()(MATRIX_TYPE  *mat, const CELL_TYPE& cell) const
    {
        std::fill(mat->begin(), mat->end(), cell);
    }
};

template<typename MATRIX_TYPE, typename CELL_TYPE>
class GridFiller<2, MATRIX_TYPE, CELL_TYPE>
{
public:

    void operator()(MATRIX_TYPE *mat, const CELL_TYPE& cell) const
    {
        for (typename MATRIX_TYPE::iterator i = mat->begin();
             i != mat->end(); 
             i++) {
            std::fill(i->begin(), i->end(), cell);
        }
    }
};

template<typename MATRIX_TYPE, typename CELL_TYPE>
class GridFiller<3, MATRIX_TYPE, CELL_TYPE>
{
public:

    void operator()(MATRIX_TYPE *mat, const CELL_TYPE& cell) const
    {
        for (int z = 0; z < (*mat).size(); ++z) {
            for (int y = 0; y < (*mat)[z].size(); ++y) {
                int size = (*mat)[z][y].size();
                CELL_TYPE *start = &(*mat)[z][y][0];
                CELL_TYPE *end = start + size;
                std::fill(start, end, cell);
            }
        }
    }
};

}

template<typename _CharT, typename _Traits, typename _CellT, typename _TopologyT>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const LibGeoDecomp::Grid<_CellT, _TopologyT>& grid)
{
    __os << grid.toString();
    return __os;
}


#endif
