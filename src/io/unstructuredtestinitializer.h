#ifndef LIBGEODECOMP_IO_UNSTRUCTUREDTESTINITIALIZER_H
#define LIBGEODECOMP_IO_UNSTRUCTUREDTESTINITIALIZER_H

#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/misc/unstructuredtestcell.h>

namespace LibGeoDecomp {

/**
 * Helper class which sets up grids for use with various variants of
 * TestCell.
 */
template<class TEST_CELL = UnstructuredTestCell<> >
class UnstructuredTestInitializer : public Initializer<TEST_CELL>
{
public:
    using Initializer<TEST_CELL>::NANO_STEPS;

    UnstructuredTestInitializer(
        int dim,
        unsigned maxSteps,
        unsigned startStep = 0) :
        dim(dim),
        lastStep(maxSteps),
        firstStep(startStep)
    {}

    virtual void grid(GridBase<TEST_CELL, 1> *ret)
    {
        int cycle = NANO_STEPS * firstStep;
        CoordBox<1> boundingBox = ret->boundingBox();

        for (CoordBox<1>::Iterator i = boundingBox.begin(); i != boundingBox.end(); ++i) {
            TEST_CELL cell(i->x(), cycle, true);

            int startNeighbors = i->x() + 1;
            int numNeighbors   = i->x() + 1;
            int endNeighbors   = startNeighbors + numNeighbors;

            for (int j = startNeighbors; j != endNeighbors; ++j) {
                int actualNeighbor = j % dim;
                cell.expectedNeighborWeights[actualNeighbor] = actualNeighbor + 0.1;
            }

            ret->set(*i, cell);
        }
    }

    Coord<1> gridDimensions() const
    {
        return Coord<1>(dim);
    }

    unsigned maxSteps() const
    {
        return lastStep;
    }

    unsigned startStep() const
    {
        return firstStep;
    }


private:
    int dim;
    unsigned lastStep;
    unsigned firstStep;
};

}

#endif
