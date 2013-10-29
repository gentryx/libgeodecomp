#ifndef LIBGEODECOMP_STORAGE_GRIDBASE_H
#define LIBGEODECOMP_STORAGE_GRIDBASE_H

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/streak.h>

namespace LibGeoDecomp {

/**
 * This is an abstract base class for all grid classes. It's generic
 * because all methods are virtual, but not very efficient -- for the
 * same reason.
 */
template<typename CELL, int DIMENSIONS>
class GridBase
{
public:
    typedef CELL CellType;
    const static int DIM = DIMENSIONS;

    /**
     * Convenice class for reading multiple cells. Incurs overhead due
     * to copying cells -- probably more often than desired.
     */
    class ConstIterator
    {
    public:
	ConstIterator(const GridBase<CELL, DIM> *grid, const Coord<DIM>& origin) :
	    grid(grid),
	    cursor(origin)
	{
	    cell = grid->get(cursor);
	}

	const CELL& operator*() const
	{
	    return cell;
	}

	const CELL *operator->() const
	{
	    return &cell;
	}

	ConstIterator& operator>>(CELL& target)
	{
	    target = cell;
	    ++(*this);
	    return *this;
	}

	void operator++()
	{
	    ++cursor.x();
	    cell = grid->get(cursor);
	}

    private:
	const GridBase<CELL, DIM> *grid;
	Coord<DIM> cursor;
	CELL cell;
    };

    /**
     * Convenice class for reading/writing multiple cells. Incurs
     * overhead due to copying cells -- probably more often than
     * desired.
     */
    class Iterator
    {
    public:
	Iterator(GridBase<CELL, DIM> *grid, const Coord<DIM>& origin) :
	    grid(grid),
	    cursor(origin)
	{
	    cell = grid->get(cursor);
	}

	const CELL& operator*() const
	{
	    return cell;
	}

	const CELL *operator->() const
	{
	    return &cell;
	}

	Iterator& operator>>(CELL& target)
	{
	    target = cell;
	    ++(*this);
	    return *this;
	}

	Iterator& operator<<(const CELL& source)
	{
	    cell = source;
	    grid->set(cursor, cell);
	    ++(*this);
	    return *this;
	}

	void operator++()
	{
	    ++cursor.x();
	    cell = grid->get(cursor);
	}

    private:
	GridBase<CELL, DIM> *grid;
	Coord<DIM> cursor;
	CELL cell;
    };

    virtual ~GridBase()
    {}

    // fixme: use functions for getting/setting streaks of cells in mpiio.h, writers, steerers, mpilayer...
    virtual void set(const Coord<DIM>&, const CELL&) = 0;
    virtual void set(const Streak<DIM>&, const CELL*) = 0;
    virtual CELL get(const Coord<DIM>&) const = 0;
    virtual void get(const Streak<DIM>&, CELL *) const = 0;
    virtual void setEdge(const CELL&) = 0;
    virtual const CELL& getEdge() const = 0;
    virtual CoordBox<DIM> boundingBox() const = 0;

    ConstIterator at(const Coord<DIM>& coord) const
    {
	return ConstIterator(this, coord);
    }

    Iterator at(const Coord<DIM>& coord)
    {
	return Iterator(this, coord);
    }

    Coord<DIM> dimensions() const
    {
        return boundingBox().dimensions;
    }


    bool operator==(const GridBase<CELL, DIMENSIONS>& other) const
    {
        if (getEdge() != other.getEdge()) {
            return false;
        }

        CoordBox<DIM> box = boundingBox();
        if (box != other.boundingBox()) {
            return false;
        }

        for (typename CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            if (get(*i) != other.get(*i)) {
                return false;
            }
        }

        return true;
    }

    bool operator!=(const GridBase<CELL, DIMENSIONS>& other) const
    {
        return !(*this == other);
    }
};

}

#endif
