#include <cxxtest/TestSuite.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <libgeodecomp/misc/supervector.h>
#include <libgeodecomp/parallelization/cacheblockingsimulator.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class CacheBlockingSimulatorTest : public CxxTest::TestSuite 
{
public:
    void testBasic()
    {
        std::cout << "\n";
        pipelineLength = 5;

        SuperVector<std::string> gridSource(10);
        SuperVector<std::string> gridTarget(10);
        SuperVector<std::string> gridBuffer(bufferSize());

        for (int i = 0; i < 10; ++i) {
            gridSource[i] = line(0);
        }
        for (int i = 0; i < 10; ++i) {
            gridTarget[i] = line('.');
        }
        for (int i = 0; i < bufferSize(); i += 2) {
            gridBuffer[i] = line('.');
        }
        for (int i = 1; i < bufferSize(); i += 2) {
            gridBuffer[i] = line('X');
        }

        printGrid("source", gridSource);
        printGrid("buffer", gridBuffer);
        printGrid("target", gridTarget);
        std::cout << "\n";

        std::string command;

        pipelinedUpdate(0, 0, 1, gridSource, gridBuffer, gridTarget);
        std::cin >> command;
        pipelinedUpdate(1, 1, 2, gridSource, gridBuffer, gridTarget);
        std::cin >> command;
        pipelinedUpdate(2, 2, 3, gridSource, gridBuffer, gridTarget);
        std::cin >> command;
        pipelinedUpdate(3, 3, 4, gridSource, gridBuffer, gridTarget);
        std::cin >> command;
        pipelinedUpdate(4, 4, 5, gridSource, gridBuffer, gridTarget);
        std::cin >> command;

        // pipelinedUpdate(0, 14, 5);

    }

    void pipelinedUpdate(int sourceZ, int localZ, int currentLength,
                         SuperVector<std::string>& gridSource,
                         SuperVector<std::string>& gridBuffer,
                         SuperVector<std::string>& gridTarget)
    {
        for (int i = 0; i < currentLength; ++i) {
            std::string source = (i == 0)                    ? "source" : "buffer";
            std::string target = (i == (pipelineLength - 1)) ? "target" : "buffer";

            int sourceIndex = (i == 0)                    ? sourceZ                  : 
                                                            normalizeLocalZ(localZ + 2 - 3 * i);
            int targetIndex = (i == (pipelineLength - 1)) ? sourceZ - pipelineLength + 1 : 
                                                            normalizeLocalZ(localZ + 0 - 3 * i);

            std::cout << "update(grid: " 
                      << source << "[" << std::setw(2) << sourceIndex << "] -> " 
                      << target << "[" << std::setw(2) << targetIndex << "], time: " 
                      << i << " -> " << (i + 1) << ")\n";
            SuperVector<std::string> *grid = &gridBuffer;
            if (i == (pipelineLength - 1)) {
                grid = &gridTarget;
            }
            (*grid)[targetIndex] = line(i + 1);

            printGrid("source", gridSource);
            printGrid("buffer", gridBuffer);
            printGrid("target", gridTarget);
            std::cout << "\n";
        }

        std::cout << "\n";
    }

    int normalizeLocalZ(int localZ)
    {
        return (localZ + bufferSize()) % bufferSize();
    }

    int bufferSize()
    {
        return pipelineLength * 3 - 3;
    }

    template<typename T>
    std::string line(T item)
    {
        std::stringstream buf;
        for (int i = 0; i < 20; ++i) {
            buf << item;
        }
        return buf.str();
    }

    void printGrid(std::string name, SuperVector<std::string> grid)
    {
        std::cout << name << ":\n";
        for (int i = 0; i < grid.size(); ++i) {
            std::cout << std::setw(2) << i << " " << grid[i] << "\n";
        }
    }

private:
    int pipelineLength;
};

}
