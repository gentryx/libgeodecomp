#ifndef LIBGEODECOMP_IO_SIMPLECELLPLOTTER_H
#define LIBGEODECOMP_IO_SIMPLECELLPLOTTER_H

#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/misc/palette.h>
#include <libgeodecomp/misc/quickpalette.h>
#include <libgeodecomp/misc/sharedptr.h>
#include <libgeodecomp/storage/filter.h>
#include <libgeodecomp/storage/selector.h>

// Kill warning 4514 in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <stdexcept>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

class PPMWriterTest;

namespace SimpleCellPlotterHelpers {

// do not warn about padding...
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4820 )
#endif

/**
 * Converts a cell to color, based on a user-supplied palette and
 * user-selected data field of the cell. Useful of a Writer should
 * generate images colored by a certain aspect (e.g. temperature) of
 * the simulation model.
 */
template<typename CELL, typename MEMBER, typename PALETTE>
class CellToColor : public Filter<CELL, MEMBER, Color>
{
public:
    explicit
    CellToColor(const PALETTE& palette = PALETTE()) :
        palette(palette)
    {}

    void copyStreakInImpl(
        const Color* /* source */,
        MemoryLocation::Location sourceLocation,
        MEMBER* /* target */,
        MemoryLocation::Location targetLocation,
        const std::size_t /* num */,
        const std::size_t /* stride */)
    {
        checkMemoryLocations(sourceLocation, targetLocation);

        throw std::logic_error(
            "undefined behavior: can only convert members to colors, not the other way around");
    }

    void copyStreakOutImpl(
        const MEMBER *source,
        MemoryLocation::Location sourceLocation,
        Color *target,
        MemoryLocation::Location targetLocation,
        const std::size_t num,
        const std::size_t /* stride */)
    {
        checkMemoryLocations(sourceLocation, targetLocation);

        for (std::size_t i = 0; i < num; ++i) {
            target[i] = palette[source[i]];
        }
    }

    void copyMemberInImpl(
        const Color* /* source */,
        MemoryLocation::Location sourceLocation,
        CELL* /* target */,
        MemoryLocation::Location targetLocation,
        std::size_t /* num */,
        MEMBER CELL::* /* memberPointer */)
    {
        checkMemoryLocations(sourceLocation, targetLocation);

        throw std::logic_error("undefined behavior: can only convert cells to colors, not the other way around");
    }

    void copyMemberOutImpl(
        const CELL *source,
        MemoryLocation::Location sourceLocation,
        Color *target,
        MemoryLocation::Location targetLocation,
        std::size_t num,
        MEMBER CELL:: *memberPointer)
    {
        checkMemoryLocations(sourceLocation, targetLocation);

        for (std::size_t i = 0; i < num; ++i) {
            target[i] = palette[source[i].*memberPointer];
        }
    }

private:
    PALETTE palette;

    void checkMemoryLocations(
        MemoryLocation::Location sourceLocation,
        MemoryLocation::Location targetLocation)
    {
        if ((sourceLocation == MemoryLocation::CUDA_DEVICE) ||
            (targetLocation == MemoryLocation::CUDA_DEVICE)) {
            throw std::logic_error("DefaultFilter can't access CUDA device memory");
        }

        if ((sourceLocation != MemoryLocation::HOST) ||
            (targetLocation != MemoryLocation::HOST)) {
            throw std::invalid_argument("unknown combination of source and target memory locations");
        }
    }
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

/**
 * This is a convenience class which uses a Palette to map a single
 * value of a cell to a color range.
 */
template<typename CELL_TYPE>
class SimpleCellPlotter
{
public:
    friend class PPMWriterTest;

    template<typename MEMBER, typename PALETTE>
    explicit SimpleCellPlotter(MEMBER CELL_TYPE:: *memberPointer, const PALETTE& palette) :
        cellToColorSelector(
            memberPointer,
            "unnamed parameter",
            typename SharedPtr<FilterBase<CELL_TYPE> >::Type(
                new SimpleCellPlotterHelpers::CellToColor<CELL_TYPE, MEMBER, PALETTE>(palette)))
    {}

    virtual ~SimpleCellPlotter()
    {}

    template<typename PAINTER>
    void operator()(
        const CELL_TYPE& cell,
        PAINTER& painter,
        const Coord<2>& cellDimensions) const
    {
        Color color;
        cellToColorSelector.copyMemberOut(
            &cell,
            MemoryLocation::HOST,
            reinterpret_cast<char*>(&color),
            MemoryLocation::HOST,
            1);

        painter.fillRect(
            0, 0,
            cellDimensions.x(), cellDimensions.y(),
            color);
    }

private:
    Selector<CELL_TYPE> cellToColorSelector;
};

}

#endif
