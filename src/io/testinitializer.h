#ifndef LIBGEODECOMP_IO_TESTINITIALIZER_H
#define LIBGEODECOMP_IO_TESTINITIALIZER_H

#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/misc/testcell.h>

namespace LibGeoDecomp {

template<int DIM>
class TestInitializerHelper;

template<>
class TestInitializerHelper<2>
{
public:
    static Coord<2> getDimensions()
    {
        return Coord<2>(17, 12);
    }

    static const int maxSteps = 31;
};

template<>
class TestInitializerHelper<3>
{
public:
    static Coord<3> getDimensions()
    {
        return Coord<3>(13, 12, 11);
    }

    static const int maxSteps = 21;
};

template<class TEST_CELL>
class TestInitializer : public Initializer<TEST_CELL>
{
public:
    static const int DIM = TEST_CELL::DIMENSIONS;

    TestInitializer(
        const Coord<DIM>& dim = TestInitializerHelper<DIM>::getDimensions(),
        const unsigned& maxSteps = TestInitializerHelper<DIM>::maxSteps ,
        const unsigned& startStep = 0) :
        dimensions(dim),
        maximumSteps(maxSteps),
        step1(startStep)
    {}

    virtual void grid(GridBase<TEST_CELL, DIM> *ret)
    {
        CoordBox<DIM> rect = ret->boundingBox();
        unsigned cycle = startStep() * TEST_CELL::nanoSteps();
        for (typename CoordBox<DIM>::Iterator i = rect.begin(); i != rect.end(); ++i) {
            Coord<DIM> coord = TEST_CELL::Topology::normalize(*i, dimensions);
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

    std::string dump()
    {
        return "foo";
    }

private:
    Coord<DIM> dimensions;
    unsigned maximumSteps;
    unsigned step1;
};

}

#endif
