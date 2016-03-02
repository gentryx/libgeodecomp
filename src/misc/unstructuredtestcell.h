#ifndef LIBGEODECOMP_MISC_UNSTRUCTUREDTESTCELL_H
#define LIBGEODECOMP_MISC_UNSTRUCTUREDTESTCELL_H

#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/testcell.h>
#include <libflatarray/macros.hpp>

namespace LibGeoDecomp {

namespace UnstructuredTestCellHelpers {

/**
 * We'll use UnstructuredTestCell with different API specs. This empty
 * one doesn't add anything to the default.
 */
class EmptyAPI
{};

/**
 * Struct of Arrays is another important dimension of the test range.
 */
class SoAAPI :
        public APITraits::HasSoA,
        public APITraits::HasUpdateLineX,
        public APITraits::HasSellC<32>,
        public APITraits::HasSellSigma<1>
{};

}

/**
 * This class is the counterpart to TestCell for structured grids. We
 * use this class to verify that our Simulators correcly invoke
 * updates on unstructed grids and that ghost zone synchronization is
 * working as expected.
 */
template<typename ADDITIONAL_API = UnstructuredTestCellHelpers::EmptyAPI, typename OUTPUT = TestCellHelpers::StdOutput>
class UnstructuredTestCell
{
public:
    class API :
        public ADDITIONAL_API,
        public APITraits::HasUnstructuredTopology,
        public APITraits::HasNanoSteps<13>
    {
    public:
        LIBFLATARRAY_CUSTOM_SIZES((16)(32)(64)(128)(256)(512)(1024), (1), (1))
    };

    const static unsigned NANO_STEPS = APITraits::SelectNanoSteps<UnstructuredTestCell>::VALUE;

    inline explicit
    UnstructuredTestCell(int id = -1, unsigned cycleCounter = 0, bool isValid = false, bool isEdgeCell = false) :
        id(id),
        cycleCounter(cycleCounter),
        isValid(isValid),
        isEdgeCell(isEdgeCell)
    {}

    bool valid() const
    {
        return isValid;
    }

    bool edgeCell() const
    {
        return isEdgeCell;
    }

    template<typename HOOD>
    void update(const HOOD& hood, int nanoStep)
    {
        *this = hood[hood.index()];
        verify(hood.begin(), hood.end(), hood, nanoStep);
    }

    template<typename HOOD_OLD, typename HOOD_NEW>
    static void updateLineX(HOOD_NEW& hoodNew, int indexEnd, HOOD_OLD& hoodOld, int nanoStep)
    {
        // Important: index is actually the index in the chunkVector, not necessarily a cell id.
        for (; hoodOld.index() < indexEnd / HOOD_OLD::ARITY; ++hoodOld) {

            // assemble weight maps:
            std::vector<std::map<int, double> > weights(HOOD_OLD::ARITY);
            for (typename HOOD_OLD::Iterator i = hoodOld.begin(); i != hoodOld.end(); ++i) {
                const int *columnPointer = i.first();
                const double *weightPointer = i.second();

                for (int i = 0; i < HOOD_OLD::ARITY; ++i) {
                    // ignore 0-padding
                    if ((columnPointer[i] != 0) || (weightPointer[i] != 0.0)) {
                        weights[i][columnPointer[i]] = weightPointer[i];
                    }
                }
            }

            // we need to create actual cells so we can call
            // member functions. users would not do this (because
            // it's slow), but it's good for testing.
            std::vector<UnstructuredTestCell> cells;
            for (int i = 0; i < HOOD_OLD::ARITY; ++i) {
                int index = hoodOld.index() * HOOD_OLD::ARITY + i;
                cells << hoodOld[index];
                cells.back().verify(weights[i].begin(), weights[i].end(), hoodOld, nanoStep);
            }

            // copy back to new grid:
            for (int i = 0; i < HOOD_OLD::ARITY; ++i) {
                hoodNew << cells[i];
                ++hoodNew;
            }
        }
    }

    int id;
    unsigned cycleCounter;
    bool isValid;
    bool isEdgeCell;
    std::map<int, double> expectedNeighborWeights;

private:
    template<typename CELL>
    void checkNeighbor(const int otherID, const CELL& otherCell)
    {
        if (otherCell.id != otherID) {
            OUTPUT() << "UnstructuredTestCell error: other cell has ID " << otherCell.id
                     << ", but expected ID " << otherID << "\n";
            isValid = false;
        }

        if (otherCell.cycleCounter != cycleCounter) {
            OUTPUT() << "UnstructuredTestCell error: other cell on ID " << otherCell.id
                     << " is in cycle " << otherCell.cycleCounter
                     << ", but expected " << cycleCounter << "\n";
            isValid = false;
        }

        if (!otherCell.isValid) {
            OUTPUT() << "UnstructuredTestCell error: other cell on ID " << otherCell.id
                     << " is invalid\n";
            isValid = false;
        }
    }

    template<typename ITERATOR1, typename ITERATOR2, typename HOOD>
    void verify(const ITERATOR1& begin, const ITERATOR2& end, const HOOD& hood, int nanoStep)
    {
        std::map<int, double> actualNeighborWeights;

        for (ITERATOR1 i = begin; i != end; ++i) {
            checkNeighbor((*i).first, hood[(*i).first]);
            actualNeighborWeights[(*i).first] = (*i).second;
        }

        if (expectedNeighborWeights != actualNeighborWeights) {
            OUTPUT() << "UnstructuredTestCell error: id " << id
                     << " is not valid on cycle " << cycleCounter
                     << ", nanoStep: " << nanoStep << "\n"
                     << "  expected weights: " << expectedNeighborWeights << "\n"
                     << "  got weights: " << actualNeighborWeights << "\n";
            isValid = false;
        }

        int expectedNanoStep = cycleCounter % APITraits::SelectNanoSteps<UnstructuredTestCell>::VALUE;
        if (expectedNanoStep != nanoStep) {
            OUTPUT() << "UnstructuredTestCell error: id " << id
                     << " saw bad nano step " << nanoStep
                     << " (expected: " << expectedNanoStep << ")\n";
            isValid = false;
        }

        if (expectedNeighborWeights.size() != std::size_t(id + 1)) {
            OUTPUT() << "UnstructuredTestCell error: id " << id
                     << " has a bad weights set\n";
            isValid = false;
        }

        if (!isValid) {
            OUTPUT() << "UnstructuredTestCell error: id " << id << " is invalid\n";
        }
        ++cycleCounter;

    }
};

typedef std::map<int, double> WeightsMap;
typedef UnstructuredTestCell<UnstructuredTestCellHelpers::SoAAPI> UnstructuredTestCellSoA;

}

LIBFLATARRAY_REGISTER_SOA(
    LibGeoDecomp::UnstructuredTestCellSoA,
    ((int)(id))
    ((unsigned)(cycleCounter))
    ((bool)(isValid))
    ((bool)(isEdgeCell))
    ((LibGeoDecomp::WeightsMap)(expectedNeighborWeights))
                          )

#endif
