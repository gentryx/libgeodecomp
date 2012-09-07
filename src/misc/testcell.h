#ifndef _libgeodecomp_misc_testcell_h_
#define _libgeodecomp_misc_testcell_h_

#include <iostream>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/coordmap.h>
#include <libgeodecomp/misc/stencils.h>

namespace LibGeoDecomp {

template<int DIM>
class TestCellTopology
{
public:
    typedef typename Topologies::Cube<DIM>::Topology Topology;
};

// make the 3D TestCell use a torus topology for a change...
template<>
class TestCellTopology<3>
{
public:
    typedef Topologies::Torus<3>::Topology Topology;
};

class TestCellStdOutput
{
public:
    template<typename T>
    const TestCellStdOutput& operator<<(T output) const
    {
        std::cout << output;
        return *this;
    }
};

class TestCellNoOutput
{
public:
    template<typename T>
    const TestCellNoOutput& operator<<(T /*unused*/) const
    {
        return *this;
    }
};

/**
 * Useful for verifying the various parallelizations in LibGeoDecomp
 */
template<int DIM, class OUTPUT=TestCellStdOutput>
class TestCell
{
    friend class Typemaps;
    friend class TestCellTest;

public:
    typedef typename Stencils::Moore<DIM, 1> Stencil;
    typedef typename TestCellTopology<DIM>::Topology Topology;

    static inline unsigned nanoSteps() 
    { 
        return 27; 
    }

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
        const Coord<DIM>& _pos, 
        const Coord<DIM>& _gridDim,
        const unsigned& _cycleCounter = 0,
        const double& _testValue = defaultValue()) :
        pos(_pos), 
        dimensions(Coord<DIM>(), _gridDim),
        cycleCounter(_cycleCounter), 
        isValid(true),
        testValue(_testValue)
    {
        isEdgeCell = !inBounds(pos);
    }

    const bool& valid() const { return isValid; }    

    bool inBounds(const Coord<DIM>& c) const
    {
        return !Topologies::IsOutOfBoundsHelper<
            DIM - 1, Coord<DIM>, Topology>()(
                c, dimensions.dimensions);
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
        *this = neighborhood[FixedCoord<0, 0, 0>()];

        if (isEdgeCell) {
            OUTPUT() << "TestCell error: update called for edge cell\n";
            isValid = false;
            return;
        }

        // check complete Moore neighborhood (including old self), ugly manual loop, 
        isValid &= checkNeighbor(neighborhood[FixedCoord<-1, -1,  0>()], FixedCoord<-1, -1,  0>());
        isValid &= checkNeighbor(neighborhood[FixedCoord< 0, -1,  0>()], FixedCoord< 0, -1,  0>());
        isValid &= checkNeighbor(neighborhood[FixedCoord< 1, -1,  0>()], FixedCoord< 1, -1,  0>());

        isValid &= checkNeighbor(neighborhood[FixedCoord<-1,  0,  0>()], FixedCoord<-1,  0,  0>());
        isValid &= checkNeighbor(neighborhood[FixedCoord< 0,  0,  0>()], FixedCoord< 0,  0,  0>());
        isValid &= checkNeighbor(neighborhood[FixedCoord< 1,  0,  0>()], FixedCoord< 1,  0,  0>());

        isValid &= checkNeighbor(neighborhood[FixedCoord<-1,  1,  0>()], FixedCoord<-1,  1,  0>());
        isValid &= checkNeighbor(neighborhood[FixedCoord< 0,  1,  0>()], FixedCoord< 0,  1,  0>());
        isValid &= checkNeighbor(neighborhood[FixedCoord< 1,  1,  0>()], FixedCoord< 1,  1,  0>());
        
        if (DIM == 3) {
            isValid &= checkNeighbor(neighborhood[FixedCoord<-1, -1, -1>()], FixedCoord<-1, -1, -1>());
            isValid &= checkNeighbor(neighborhood[FixedCoord< 0, -1, -1>()], FixedCoord< 0, -1, -1>());
            isValid &= checkNeighbor(neighborhood[FixedCoord< 1, -1, -1>()], FixedCoord< 1, -1, -1>());

            isValid &= checkNeighbor(neighborhood[FixedCoord<-1,  0, -1>()], FixedCoord<-1,  0, -1>());
            isValid &= checkNeighbor(neighborhood[FixedCoord< 0,  0, -1>()], FixedCoord< 0,  0, -1>());
            isValid &= checkNeighbor(neighborhood[FixedCoord< 1,  0, -1>()], FixedCoord< 1,  0, -1>());

            isValid &= checkNeighbor(neighborhood[FixedCoord<-1,  1, -1>()], FixedCoord<-1,  1, -1>());
            isValid &= checkNeighbor(neighborhood[FixedCoord< 0,  1, -1>()], FixedCoord< 0,  1, -1>());
            isValid &= checkNeighbor(neighborhood[FixedCoord< 1,  1, -1>()], FixedCoord< 1,  1, -1>());
        
            isValid &= checkNeighbor(neighborhood[FixedCoord<-1, -1,  1>()], FixedCoord<-1, -1,  1>());
            isValid &= checkNeighbor(neighborhood[FixedCoord< 0, -1,  1>()], FixedCoord< 0, -1,  1>());
            isValid &= checkNeighbor(neighborhood[FixedCoord< 1, -1,  1>()], FixedCoord< 1, -1,  1>());

            isValid &= checkNeighbor(neighborhood[FixedCoord<-1,  0,  1>()], FixedCoord<-1,  0,  1>());
            isValid &= checkNeighbor(neighborhood[FixedCoord< 0,  0,  1>()], FixedCoord< 0,  0,  1>());
            isValid &= checkNeighbor(neighborhood[FixedCoord< 1,  0,  1>()], FixedCoord< 1,  0,  1>());

            isValid &= checkNeighbor(neighborhood[FixedCoord<-1,  1,  1>()], FixedCoord<-1,  1,  1>());
            isValid &= checkNeighbor(neighborhood[FixedCoord< 0,  1,  1>()], FixedCoord< 0,  1,  1>());
            isValid &= checkNeighbor(neighborhood[FixedCoord< 1,  1,  1>()], FixedCoord< 1,  1,  1>());
        }
    
        if (nanoStep >= nanoSteps()) {
            OUTPUT() << "TestCell error: nanoStep too large: " 
                     << nanoStep << "\n";
            isValid = false;
            return;
        }

        unsigned expectedNanoStep = cycleCounter % nanoSteps();
        if (nanoStep != expectedNanoStep) {
            OUTPUT() << "TestCell error: nanoStep out of sync. got " 
                     << nanoStep << " but expected " 
                     << expectedNanoStep << "\n";
            isValid = false;
            return;
        }

        ++cycleCounter;
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
                     << other.toString() << "\n"
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
                Topology::normalize(rawPos, dimensions.dimensions);

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

template<typename _CharT, typename _Traits, int _Dim>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const LibGeoDecomp::TestCell<_Dim>& cell)
{
    __os << cell.toString();
    return __os;
}

#endif
