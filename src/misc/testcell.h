#ifndef _libgeodecomp_misc_testcell_h_
#define _libgeodecomp_misc_testcell_h_

#include <iostream>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/coordmap.h>

namespace LibGeoDecomp {

class TestCellBase 
{
public:
    static std::ostream *stream;
};

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

/**
 * Useful for verifying the various parallelizations in LibGeoDecomp
 */
template<int DIM>
class TestCell
{
    friend class Typemaps;
    friend class TestCellTest;

public:
    typedef typename TestCellTopology<DIM>::Topology Topology;

    static inline unsigned nanoSteps() { return 27; }

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
        return (this->pos == other.pos)
            && (this->dimensions == other.dimensions)
            && (this->cycleCounter == other.cycleCounter)
            && (this->isEdgeCell == other.isEdgeCell)
            && (this->isValid == other.isValid)            
            && (this->testValue == other.testValue);            
    }

    bool operator!=(const TestCell& other) const
    {
        return !((*this) == other);
    }

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned& nanoStep) 
    {
        // initialize Cell by copying from previous state
        *this = neighborhood[Coord<DIM>(0, 0)];

        if (isEdgeCell) {
            (*TestCellBase::stream) 
                << "TestCell error: update called for edge cell\n";
            isValid = false;
            return;
        }

        // check complete Moore neighborhood (including old self)
        CoordBox<DIM> neighborBox(CoordDiagonal<DIM>()(-1),
                                  CoordDiagonal<DIM>()(3));
        CoordBoxSequence<DIM> seq = neighborBox.sequence();
        while (seq.hasNext()) {
            Coord<DIM> relativeLoc = seq.next();
            isValid &= checkNeighbor(neighborhood[relativeLoc], relativeLoc);
        }

        if (nanoStep >= nanoSteps()) {
            (*TestCellBase::stream) << "TestCell error: nanoStep too large: " 
                                    << nanoStep << "\n";
            isValid = false;
            return;
        }

        unsigned expectedNanoStep = cycleCounter % nanoSteps();
        if (nanoStep != expectedNanoStep) {
            (*TestCellBase::stream) 
                << "TestCell error: nanoStep out of sync. got " 
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
    bool checkNeighbor(const TestCell& other, 
                       const Coord<DIM>& relativeLoc) const
    {
        if (!other.isValid) {
            (*TestCellBase::stream) 
                << "Update Error for " << this->toString() << ":\n"
                << "Invalid Neighbor at " << relativeLoc << ":\n" 
                << other.toString() << "\n"
                << "--------------" << "\n";
            return false;
        }
        bool otherShouldBeEdge = !inBounds(pos + relativeLoc);
        if (other.isEdgeCell != otherShouldBeEdge) {
            (*TestCellBase::stream) 
                << "TestCell error: bad edge cell (expected: " 
                << otherShouldBeEdge << ", is: " 
                << other.isEdgeCell << " at relative coord " 
                << relativeLoc << ")\n";
            return false;
        }        
        if (!otherShouldBeEdge) {
            if (other.cycleCounter != cycleCounter) {
                (*TestCellBase::stream) 
                    << "Update Error for TestCell " 
                    << this->toString() << ":\n"
                    << "cycle counter out of sync with neighbor " 
                    << other.toString() << "\n";
                return false;
            }
            if (other.dimensions != dimensions) {
                (*TestCellBase::stream) 
                    << "TestCell error: grid dimensions differ. Expected: " 
                    << dimensions << ", but got " << other.dimensions << "\n";
                return false;
            }

            Coord<DIM> rawPos = pos + relativeLoc;
            Coord<DIM> expectedPos = 
                Topology::normalize(rawPos, dimensions.dimensions);

            if (other.pos != expectedPos) {
                (*TestCellBase::stream) << "TestCell error: other position " 
                                        << other.pos
                                        << " doesn't match expected "
                                        << expectedPos << "\n";
                return false;
            }
        }
        return true;
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
