#include <cxxtest/TestSuite.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <libgeodecomp/misc/supervector.h>
#include <libgeodecomp/parallelization/cacheblockingsimulator.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class Line
{
public:

    Line() :
        offset(-2)
    {
        std::stringstream buf;
        for (int i = 0; i < 30; ++i) {
            buf << ".";
        }
        string = buf.str();
    }

    template<typename ELEM>
    Line(int offset, ELEM elem) :
        offset(offset)
    {
        std::stringstream buf;
        for (int i = 0; i < 30; ++i) {
            buf << elem;
        }
        string = buf.str();
    }

    std::string toString()
    {
        std::stringstream buf;
        buf << std::setw(2) << offset << " " << string;
        return buf.str();
    }

    int offset;
    std::string string;
};

class CacheBlockingSimulatorTest : public CxxTest::TestSuite 
{
public:
    void testBasic()
    {
        std::cout << "\n";
        pipelineLength = 5;

        SuperVector<Line> gridSource(10);
        SuperVector<Line> gridTarget(10);
        SuperVector<Line> gridBuffer(bufferSize());

        for (int i = 0; i < 10; ++i) {
            gridSource[i] = Line(i, 0);
        }
        for (int i = 1; i < bufferSize(); i += 2) {
            gridBuffer[i] = Line(-1, 'X');
        }

        printGrid("source", gridSource);
        printGrid("buffer", gridBuffer);
        printGrid("target", gridTarget);
        std::cout << "\n";

        std::string command;

        pipelinedUpdate( 0,  0, 0, 1, gridSource, gridBuffer, gridTarget);
        pipelinedUpdate( 1,  1, 0, 2, gridSource, gridBuffer, gridTarget);
        pipelinedUpdate( 2,  2, 0, 3, gridSource, gridBuffer, gridTarget);
        pipelinedUpdate( 3,  3, 0, 4, gridSource, gridBuffer, gridTarget);
        pipelinedUpdate( 4,  4, 0, 5, gridSource, gridBuffer, gridTarget);
        pipelinedUpdate( 5,  5, 0, 5, gridSource, gridBuffer, gridTarget);
        pipelinedUpdate( 6,  6, 0, 5, gridSource, gridBuffer, gridTarget);
        pipelinedUpdate( 7,  7, 0, 5, gridSource, gridBuffer, gridTarget);
        pipelinedUpdate( 8,  8, 0, 5, gridSource, gridBuffer, gridTarget);
        pipelinedUpdate( 9,  9, 0, 5, gridSource, gridBuffer, gridTarget);
        pipelinedUpdate(10, 10, 0, 5, gridSource, gridBuffer, gridTarget);
        pipelinedUpdate(11, 11, 1, 5, gridSource, gridBuffer, gridTarget);
        pipelinedUpdate(12, 12, 2, 5, gridSource, gridBuffer, gridTarget);
        pipelinedUpdate(13, 13, 3, 5, gridSource, gridBuffer, gridTarget);
    }

    void pipelinedUpdate(int globalIndex, int localIndex, int firstStage, int lastStage,
                         SuperVector<Line>& gridSource,
                         SuperVector<Line>& gridBuffer,
                         SuperVector<Line>& gridTarget)
    {
        for (int i = firstStage; i < lastStage; ++i) {
            std::string source = (i == 0)                    ? "source" : "buffer";
            std::string target = (i == (pipelineLength - 1)) ? "target" : "buffer";

            int sourceIndex = (i == 0)                    ? globalIndex                : 
                                                            normalizeIndex(localIndex + 2 - 3 * i);
            int targetIndex = (i == (pipelineLength - 1)) ? globalIndex - pipelineLength + 1 : 
                                                            normalizeIndex(localIndex + 0 - 3 * i);

            std::cout << "update(grid: " 
                      << source << "[" << std::setw(2) << sourceIndex << "] -> " 
                      << target << "[" << std::setw(2) << targetIndex << "], time: " 
                      << i << " -> " << (i + 1) << ")\n";

            SuperVector<Line> *gridOld = &gridBuffer;
            if (i == 0) {
                gridOld = &gridSource;
            }

            SuperVector<Line> *gridNew = &gridBuffer;
            if (i == (pipelineLength - 1)) {
                gridNew = &gridTarget;
            }
            
            if ((globalIndex >= gridSource.size()) && (i == firstStage)) {
                (*gridNew)[targetIndex] = Line(-1, "X");
            } else {
                (*gridNew)[targetIndex] = Line((*gridOld)[sourceIndex].offset, i + 1);
            }

            printGrid("source", gridSource);
            printGrid("buffer", gridBuffer);
            printGrid("target", gridTarget);
            std::cout << "\n";
        }

        std::cout << "\n";
    }

    int normalizeIndex(int localIndex)
    {
        return (localIndex + bufferSize()) % bufferSize();
    }

    int bufferSize()
    {
        return pipelineLength * 3 - 3;
    }

    void printGrid(std::string name, SuperVector<Line> grid)
    {
        std::cout << name << ":\n";
        for (int i = 0; i < grid.size(); ++i) {
            std::cout << std::setw(2) << i << " " << grid[i].toString() << "\n";
        }
    }

private:
    int pipelineLength;
};

}
