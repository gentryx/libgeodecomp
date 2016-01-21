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
        // fixme: needs test
    }

    Coord<1> gridDimensions() const
    {
        // fixme: needs test
        return Coord<1>(dim);
    }

    unsigned maxSteps() const
    {
        // fixme: needs test
        return lastStep;
    }

    unsigned startStep() const
    {
        // fixme: needs test
        return firstStep;
    }


private:
    int dim;
    unsigned lastStep;
    unsigned firstStep;
};

}

#endif
