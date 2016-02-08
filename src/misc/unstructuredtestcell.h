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
        std::map<int, double> actualNeighborWeights;

        for (typename HOOD::Iterator i = hood.begin(); i != hood.end(); ++i) {
            checkNeighbor((*i).first, hood[(*i).first]);
            actualNeighborWeights[(*i).first] = (*i).second;
        }

        if (expectedNeighborWeights != actualNeighborWeights) {
            OUTPUT() << "UnstructuredTestCell error: id " << id
                     << " is not valid on cycle " << cycleCounter
                     << ", nanoStep: " << nanoStep << "\n";
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

    template<typename HOOD_OLD, typename HOOD_NEW>
    static void updateLineX(HOOD_NEW& hoodNew, int indexEnd, HOOD_OLD& hoodOld, int nanoStep)
    {
        for (; hoodOld.index() < indexEnd; ++hoodOld, ++hoodNew) {
            UnstructuredTestCell cell;
            cell.update(hoodOld, nanoStep);
            hoodNew << cell;
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
