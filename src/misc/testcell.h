#ifndef LIBGEODECOMP_MISC_TESTCELL_H
#define LIBGEODECOMP_MISC_TESTCELL_H

#include <iostream>
#include <libflatarray/flat_array.hpp>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/coordmap.h>
#include <libgeodecomp/misc/fixedneighborhood.h>
#include <libgeodecomp/misc/stencils.h>

namespace LibGeoDecomp {

namespace TestCellHelpers {

class EmptyAPI
{
};

class SoAAPI :
        public APITraits::HasSoA,
        public APITraits::HasFixedCoordsOnlyUpdate,
        public APITraits::HasUpdateLineX
{};

template<int DIM>
class TopologyType
{
public:
    typedef typename Topologies::Cube<DIM>::Topology Topology;
};

// make the 3D TestCell use a torus topology for a change...
template<>
class TopologyType<3>
{
public:
    typedef Topologies::Torus<3>::Topology Topology;
};

class StdOutput
{
public:
    template<typename T>
    const StdOutput& operator<<(T output) const
    {
        std::cout << output;
        return *this;
    }
};

class NoOutput
{
public:
    template<typename T>
    const NoOutput& operator<<(T /*unused*/) const
    {
        return *this;
    }
};

template<class STENCIL, int INDEX>
class CheckNeighbor
{
public:
    typedef typename STENCIL::template Coords<INDEX> RelCoord;

    template<class TESTCELL, class NEIGHBORHOOD>
    void operator()(bool *isValid, TESTCELL *cell, const NEIGHBORHOOD& neighborhood)
    {
        (*isValid) &= cell->checkNeighbor(neighborhood[RelCoord()], RelCoord());
    }
};

}

/**
 * Useful for verifying the various parallelizations in LibGeoDecomp
 */
template<
    int DIM,
    typename STENCIL = Stencils::Moore<DIM, 1>,
    typename TOPOLOGY = typename TestCellHelpers::TopologyType<DIM>::Topology,
    typename ADDITIONAL_API = TestCellHelpers::EmptyAPI,
    typename OUTPUT = TestCellHelpers::StdOutput>
class TestCell
{
public:
    friend class Typemaps;
    friend class TestCellTest;

    static const int DIMENSIONS = DIM;
    static const unsigned NANO_STEPS = 27;

    class API :
        public ADDITIONAL_API,
        public APITraits::HasTopology<TOPOLOGY>,
        public APITraits::HasNanoSteps<NANO_STEPS>,
        public APITraits::HasStencil<STENCIL>
    {};

    Coord<DIM> pos;
    CoordBox<DIM> dimensions;
    unsigned cycleCounter;
    bool isEdgeCell;
    bool isValid;
    double testValue;

    static double defaultValue()
    {
        return 666;
    }

    TestCell() :
        cycleCounter(0),
        isEdgeCell(false),
        isValid(false),
        testValue(defaultValue())
    {}

    TestCell(
        const Coord<DIM>& pos,
        const Coord<DIM>& gridDim,
        const unsigned& cycleCounter = 0,
        const double& testValue = defaultValue()) :
        pos(pos),
        dimensions(Coord<DIM>(), gridDim),
        cycleCounter(cycleCounter),
        isValid(true),
        testValue(testValue)
    {
        isEdgeCell = !inBounds(pos);
    }

    // fixme: reduce number of parametersx
    template<int DIM_X, int DIM_Y, int DIM_Z, int INDEX>
    TestCell(const LibFlatArray::soa_accessor<TestCell, DIM_X, DIM_Y, DIM_Z, INDEX>& hood)
    {
        *this << hood;
    }

    const bool& valid() const
    {
        return isValid;
    }

    bool inBounds(const Coord<DIM>& c) const
    {
        return !TOPOLOGY::isOutOfBounds(c, dimensions.dimensions);
    }

    bool operator==(const TestCell& other) const
    {
        return (pos == other.pos)
            && (dimensions == other.dimensions)
            && (cycleCounter == other.cycleCounter)
            && (isEdgeCell == other.isEdgeCell)
            && (isValid == other.isValid)
            && (testValue == other.testValue);
    }

    bool operator!=(const TestCell& other) const
    {
        return !((*this) == other);
    }

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned& nanoStep)
    {
        // initialize Cell by copying from previous state
        *this = TestCell(neighborhood[FixedCoord<0, 0, 0>()]);

        if (isEdgeCell) {
            OUTPUT() << "TestCell error: update called for edge cell\n";
            isValid = false;
            return;
        }

        Stencils::Repeat<STENCIL::VOLUME,
                         TestCellHelpers::CheckNeighbor,
                         STENCIL>()(&isValid, this, neighborhood);

        if (nanoStep >= NANO_STEPS) {
            OUTPUT() << "TestCell error: nanoStep too large: "
                     << nanoStep << "\n";
            isValid = false;
            return;
        }

        unsigned expectedNanoStep = cycleCounter % NANO_STEPS;
        if (nanoStep != expectedNanoStep) {
            OUTPUT() << "TestCell error: nanoStep out of sync. got "
                     << nanoStep << " but expected "
                     << expectedNanoStep << "\n";
            isValid = false;
            return;
        }

        ++cycleCounter;
    }

    template<typename ACCESSOR1, typename ACCESSOR2>
    static void updateLineX(
        ACCESSOR1 hoodOld, int *indexOld, int indexEnd,
        ACCESSOR2 hoodNew, int *indexNew,
        unsigned nanoStep)
    {
        for (; *indexOld < indexEnd; ++(*indexOld)) {
            TestCell cell1;
            TestCell cell2;
            hoodOld >> cell1;
            cell2 << hoodNew;

            std::cout << "updateLine\n"
                      << "1:\n"
                      << cell1 << "\n"
                      << "2:\n"
                      << cell2 << "\n";

            TestCell cell;
            cell.update(hoodOld, nanoStep);
            hoodNew << cell;

            ++(*indexNew);
        }
    }

    std::string toString() const
    {
        std::ostringstream ret;
        ret << "TestCell\n"
            << "  pos: " << pos << "\n"
            << "  dimensions: " << dimensions << "\n"
            << "  cycleCounter: " << cycleCounter << "\n"
            << "  isEdgeCell: " << (isEdgeCell ? "true" : "false") << "\n"
            << "  testValue: " << testValue << "\n"
            << "  isValid: " << (isValid ? "true" : "false") << "\n";
        return ret.str();
    }

    // returns true if valid neighbor is found (at the right place, in
    // the same cycle etc.)
    bool checkNeighbor(
        const TestCell& other,
        const Coord<DIM>& relativeLoc) const
    {
        if (!other.isValid) {
            OUTPUT() << "Update Error for " << toString() << ":\n"
                     << "Invalid Neighbor at " << relativeLoc << ":\n"
                     << other.toString()
                     << "--------------" << "\n";
            return false;
        }
        bool otherShouldBeEdge = !inBounds(pos + relativeLoc);
        if (other.isEdgeCell != otherShouldBeEdge) {
            OUTPUT() << "TestCell error: bad edge cell (expected: "
                     << otherShouldBeEdge << ", is: "
                     << other.isEdgeCell << " at relative coord "
                     << relativeLoc << ")\n";
            return false;
        }
        if (!otherShouldBeEdge) {
            if (other.cycleCounter != cycleCounter) {
                OUTPUT() << "Update Error for TestCell "
                         << toString() << ":\n"
                         << "cycle counter out of sync with neighbor "
                         << other.toString() << "\n";
                return false;
            }
            if (other.dimensions != dimensions) {
                OUTPUT() << "TestCell error: grid dimensions differ. Expected: "
                         << dimensions << ", but got " << other.dimensions << "\n";
                return false;
            }

            Coord<DIM> rawPos = pos + relativeLoc;
            Coord<DIM> expectedPos =
                TOPOLOGY::normalize(rawPos, dimensions.dimensions);

            if (other.pos != expectedPos) {
                OUTPUT() << "TestCell error: other position "
                         << other.pos
                         << " doesn't match expected "
                         << expectedPos << "\n";
                return false;
            }
        }
        return true;
    }

    template<int X, int Y, int Z>
    bool checkNeighbor(
        const TestCell& other,
        FixedCoord<X, Y, Z> coord) const
    {
        return checkNeighbor(other, Coord<DIM>(coord));
    }
};

typedef TestCell<
    3,
    Stencils::Moore<3, 1>,
    Topologies::Cube<3>::Topology,
    TestCellHelpers::SoAAPI
    > TestCellSoA;

/**
 * The MPI typemap generator need to find out for which template
 * parameter values it should generate typemaps. It does so by
 * scanning all class members. Therefore this dummy class forces the
 * typemap generator to create MPI datatypes for TestCells with the
 * dimensions as specified below.
 */
class TestCellMPIDatatypeHelper
{
    friend class Typemaps;
    TestCell<1> a;
    TestCell<2> b;
    TestCell<3> c;
};

}

LIBFLATARRAY_REGISTER_SOA(LibGeoDecomp::TestCellSoA, ((LibGeoDecomp::Coord<3>)(pos))((LibGeoDecomp::CoordBox<3>)(dimensions))((unsigned)(cycleCounter))((bool)(isEdgeCell))((bool)(isValid))((double)(testValue)))

template<typename _CharT, typename _Traits, int _Dim, typename _Stencil, typename _Topology, typename _AdditionalAPI, typename _Output>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const LibGeoDecomp::TestCell<_Dim, _Stencil, _Topology, _AdditionalAPI, _Output>& cell)
{
    __os << cell.toString();
    return __os;
}

#endif
