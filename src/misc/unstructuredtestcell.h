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

    template<typename HOOD>
    void update(const HOOD& hood, int nanoStep)
    {
        *this = hood[hood.index()];
        std::map<int, double> actualNeighborWeights;

        for (typename HOOD::Iterator i = hood.begin(); i != hood.end(); ++i) {
            checkNeighbor(i->first, hood[i->first]);
            actualNeighborWeights[i->first] = i->second;
        }

        if (expectedNeighborWeights != actualNeighborWeights) {
            OUTPUT() << "UnstructuredTestCell error: id " << id
                     << "is not valid on cycle " << cycle
                     << ", nanoStep: " << nanoStep << "\n";
            // fixme: needs test
            isValid = false;
        }

        int expectedNanoStep = cycle % APITraits::SelectNanoSteps<UnstructuredTestCell>::VALUE;
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
        ++cycle;
    }

private:
    int id;
    int cycle;
    bool isValid;
    std::map<int, double> expectedNeighborWeights;

    template<typename CELL>
    void checkNeighbor(const int otherID, const CELL& otherCell)
    {
        if (otherCell.id != otherID) {
            OUTPUT() << "UnstructuredTestCell error: other cell has ID " << otherCell.id
                     << ", but expected ID " << otherID << "\n";
            // fixme: needs test
            isValid = false;
        }

        if (otherCell.cycle != cycle) {
            OUTPUT() << "UnstructuredTestCell error: other cell on ID " << otherCell.id
                     << " is in cycle " << otherCell.cycle
                     << ", but expected " << cycle << "\n";
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
