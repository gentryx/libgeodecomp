#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_io_testinitializer_h_
#define _libgeodecomp_io_testinitializer_h_

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

template<int DIM>
class TestInitializer : public Initializer<TestCell<DIM> >
{
public:

    TestInitializer(
        const Coord<DIM>& dim = TestInitializerHelper<DIM>::getDimensions(),
        const unsigned& maxSteps = TestInitializerHelper<DIM>::maxSteps ,
        const unsigned& startStep = 0) :
        dimensions(dim),
        maximumSteps(maxSteps),
        step1(startStep)
    {}

    virtual void grid(GridBase<TestCell<DIM>, DIM> *ret)
    {
        CoordBox<DIM> rect = ret->boundingBox();
        unsigned cycle = startStep() * TestCell<DIM>::nanoSteps();
        for (CoordBoxSequence<DIM> s = rect.sequence(); s.hasNext();) {
            Coord<DIM> rawPos = s.next();
            Coord<DIM> coord = TestCell<DIM>::Topology::normalize(rawPos, dimensions);
            double i = 1 + CoordToIndex<DIM>()(coord, dimensions);
            ret->at(rawPos) = TestCell<DIM>(coord, dimensions, cycle, i);
        }
        ret->atEdge() = TestCell<DIM>(Coord<DIM>::diagonal(-1), dimensions);
        ret->atEdge().isEdgeCell = true;
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

    std::string dump() { return "foo"; }    

private:
    Coord<DIM> dimensions;
    unsigned maximumSteps;
    unsigned step1;
};

};

#endif
#endif
