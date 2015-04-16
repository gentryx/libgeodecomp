#ifndef _UNSTRUCTUREDNEIGHBORHOOD_H_
#define _UNSTRUCTUREDNEIGHBORHOOD_H_

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/storage/unstructuredgrid.h>

#include <iterator>
#include <utility>
#include <vector>

namespace LibGeoDecomp {


template<typename VALUE_TYPE>
class WeightIterator final :
        public std::iterator<std::input_iterator_tag, std::pair<int, VALUE_TYPE> >
{
private:
    std::pair<int, VALUE_TYPE> *data;
public:
    inline explicit
    WeightIterator(std::pair<int, VALUE_TYPE> *_data) :
        data(_data)
    {}

    inline
    WeightIterator(const WeightIterator& other) :
        data(other.data)
    {}

    inline
    WeightIterator& operator++()
    {
        ++data;
        return *this;
    }

    inline
    WeightIterator operator++(int)
    {
        WeightIterator tmp(*this);
        operator++();
        return tmp;
    }

    inline
    WeightIterator operator+(int value) const
    {
        return WeightIterator(data + value);
    }

    inline
    WeightIterator operator-(int value) const
    {
        return WeightIterator(data - value);
    }

    inline
    WeightIterator& operator+=(int value)
    {
        data += value;
        return *this;
    }

    inline
    WeightIterator& operator-=(int value)
    {
        data -= value;
        return *this;
    }

    inline
    bool operator==(const WeightIterator& rhs)
    {
        return data == rhs.data;
    }

    inline
    bool operator!=(const WeightIterator& rhs)
    {
        return data != rhs.data;
    }

    inline
    std::pair<int, VALUE_TYPE>& operator*()
    {
        return *data;
    }

    inline
    const std::pair<int, VALUE_TYPE>& operator*() const
    {
        return *data;
    }

    inline
    const std::pair<int, VALUE_TYPE>* operator->() const
    {
        return data;
    }
};

template<typename VALUE_TYPE>
class WeightContainer
{
private:
    std::vector<std::pair<int, VALUE_TYPE> > neighbors;
public:
    inline explicit
    WeightContainer(const std::vector<std::pair<int, VALUE_TYPE> >& data) :
        neighbors(data)
    {}

    inline explicit
    WeightContainer(std::vector<std::pair<int, VALUE_TYPE> >&& data) :
        neighbors(data)
    {}

    inline
    WeightIterator<VALUE_TYPE> begin()
    {
        return WeightIterator<VALUE_TYPE>(neighbors.size() > 0 ? neighbors.data() : nullptr);
    }

    inline
    const WeightIterator<VALUE_TYPE> begin() const
    {
        return WeightIterator<VALUE_TYPE>(neighbors.size() > 0 ? neighbors.data() : nullptr);
    }

    inline
    WeightIterator<VALUE_TYPE> end()
    {
        return WeightIterator<VALUE_TYPE>(begin() + neighbors.size());
    }

    inline
    const WeightIterator<VALUE_TYPE> end() const
    {
        return WeightIterator<VALUE_TYPE>(begin() + neighbors.size());
    }
};

/**
 * Simple neighborhood for UnstructuredGrid.
 *
 * Usage:
 *  for (const auto& i: hoodOld.weights()) {
 *    CELL cell = hoodOld[i.first];
 *    cell.sum += cell.sum * i.second;
 *  }
 */
template<typename CELL, std::size_t MATRICES = 1,
         typename VALUE_TYPE = double, int C = 64, int SIGMA = 1>
class UnstructuredNeighborhood
{
private:
    UnstructuredGrid<CELL, MATRICES, VALUE_TYPE, C, SIGMA>& grid;
    long long xOffset;
public:
    inline explicit
    UnstructuredNeighborhood(UnstructuredGrid<CELL, MATRICES, VALUE_TYPE, C, SIGMA>& _grid,
                             long long startX) :
        grid(_grid), xOffset(startX)
    {}

    inline
    const CELL& operator[](int index) const
    {
        return grid[index];
    }

    inline
    CELL& operator[](int index)
    {
        return grid[index];
    }

    inline
    UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA>& operator++()
    {
        ++xOffset;
        return *this;
    }

    inline
    UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA> operator++(int)
    {
        UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA> tmp(*this);
        ++xOffset;
        return tmp;
    }

    inline
    UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA>& operator--()
    {
        --xOffset;
        return *this;
    }

    inline
    UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA> operator--(int)
    {
        UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA> tmp(*this);
        --xOffset;
        return tmp;
    }

    inline
    UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA>& operator+=(int value)
    {
        xOffset += value;
        return *this;
    }

    inline
    UnstructuredNeighborhood<CELL, MATRICES, VALUE_TYPE, C, SIGMA>& operator-=(int value)
    {
        xOffset -= value;
        return *this;
    }

    inline
    const long& index() const { return xOffset; }

    inline
    long& index() { return xOffset; }

    inline
    WeightContainer<VALUE_TYPE> weights() const
    {
        // FIXME: this is only the neighborhood for matrices[0]
        return weights(0);
    }

    inline
    WeightContainer<VALUE_TYPE> weights(std::size_t matrixID) const
    {
        auto row = grid.getAdjacency(matrixID).getRow(xOffset);
        return WeightContainer<VALUE_TYPE>(std::move(row));
    }
};

}

#endif /* _UNSTRUCTUREDNEIGHBORHOOD_H_ */
