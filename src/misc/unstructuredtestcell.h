#ifndef LIBGEODECOMP_MISC_UNSTRUCTUREDTESTCELL_H
#define LIBGEODECOMP_MISC_UNSTRUCTUREDTESTCELL_H

#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/testcell.h>

namespace LibGeoDecomp {

// fixme: needs UnstructuredSoATestCell, too
template<typename OUTPUT = TestCellHelpers::StdOutput>
class UnstructuredTestCell
{
public:
    class API :
        public APITraits::HasUnstructuredTopology,
        public APITraits::HasNanoSteps<13>
    {};

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
                     << "is not valid on cycle " << cycleCounter
                     << ", nanoStep: " << nanoStep << "\n";
            // fixme: needs test
            isValid = false;
        }

        int expectedNanoStep = cycleCounter % APITraits::SelectNanoSteps<UnstructuredTestCell>::VALUE;
        if (expectedNanoStep != nanoStep) {
            OUTPUT() << "UnstructuredTestCell error: id " << id
                     << " saw bad nano step " << nanoStep
                     << " (expected: " << expectedNanoStep << ")\n";
            // fixme: needs test
            isValid = false;
        }

        if (!isValid) {
            OUTPUT() << "UnstructuredTestCell error: id " << id << "is invalid\n";
        }
        ++cycleCounter;
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
            // fixme: needs test
            isValid = false;
        }

        if (otherCell.cycleCounter != cycleCounter) {
            OUTPUT() << "UnstructuredTestCell error: other cell on ID " << otherCell.id
                     << " is in cycle " << otherCell.cycleCounter
                     << ", but expected " << cycleCounter << "\n";
            // fixme: needs test
            isValid = false;
        }

        if (!otherCell.isValid) {
            OUTPUT() << "UnstructuredTestCell error: other cell on ID " << otherCell.id
                     << " is invalid\n";
            // fixme: needs test
            isValid = false;
        }
    }
};

}

#endif
