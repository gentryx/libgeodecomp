#ifndef LIBGEODECOMP_IO_TESTINITIALIZER_H
#define LIBGEODECOMP_IO_TESTINITIALIZER_H

#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/misc/testcell.h>

namespace LibGeoDecomp {

/**
 * Helper class which sets up grids for use with various variants of
 * TestCell.
 */
template<class TEST_CELL>
class TestInitializer : public Initializer<TEST_CELL>
{
public:
    using Initializer<TEST_CELL>::NANO_STEPS;
    typedef typename Initializer<TEST_CELL>::Topology Topology;
    static const int DIM = TEST_CELL::DIMENSIONS;

    explicit TestInitializer(
        const Coord<DIM>& dim = defaultDimensions(Coord<DIM>()),
        const unsigned maxSteps = defaultTimeSteps(Coord<DIM>()),
        const unsigned startStep = 0) :
        dimensions(dim),
        maximumSteps(maxSteps),
        step1(startStep)
    {}

    virtual void grid(GridBase<TEST_CELL, DIM> *ret)
    {
        CoordBox<DIM> rect = ret->boundingBox();
        unsigned cycle = startStep() * NANO_STEPS;
        for (typename CoordBox<DIM>::Iterator i = rect.begin(); i != rect.end(); ++i) {
            Coord<DIM> coord = Topology::normalize(*i, dimensions);
            double index = 1 + coord.toIndex(dimensions);
            ret->set(*i, TEST_CELL(coord, dimensions, cycle, index));
        }

        TEST_CELL edgeCell(Coord<DIM>::diagonal(-1), dimensions);
        edgeCell.isEdgeCell = true;
        ret->setEdge(edgeCell);
    }

    Coord<DIM> gridDimensions() const
    {
        return dimensions;
    }

    unsigned maxSteps() const
    {
        return maximumSteps;
    }

    unsigned startStep() const
    {
        return step1;
    }

private:
    Coord<DIM> dimensions;
    unsigned maximumSteps;
    unsigned step1;

    static Coord<2> defaultDimensions(const Coord<2>&)
    {
        return Coord<2>(17, 12);
    }

    static Coord<3> defaultDimensions(const Coord<3>&)
    {
        return Coord<3>(13, 12, 11);
    }

    static int defaultTimeSteps(const Coord<2>&)
    {
        return 31;
    }

    static int defaultTimeSteps(const Coord<3>&)
    {
        return 21;
    }

};

}

#endif
