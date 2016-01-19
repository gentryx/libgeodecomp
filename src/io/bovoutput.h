#ifndef LIBGEODECOMP_IO_BOVOUTPUT_H
#define LIBGEODECOMP_IO_BOVOUTPUT_H

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/storage/selector.h>

#include <fstream>

namespace LibGeoDecomp {

template<int DIM>
class Region;

/**
 * Utility class which takes over writing the Brick of Values (BOV)
 * format for regular grid data, which is used by VisIt and others.
 */
template<typename CELL_TYPE, int DIM>
class BOVOutput
{
public:

    static void writeHeader(
        const std::string& filenameBoV,
        const std::string& filenameData,
        int step,
        const Coord<DIM>& dimensions,
        const Coord<3>& brickletDim,
        const Selector<CELL_TYPE>& selector)
    {
        std::ofstream file;
        file.open(filenameBoV.c_str());

        // BOV only accepts 3D data, so we'll have to inflate 1D
        // and 2D dimensions.
        Coord<DIM> c = dimensions;
        Coord<3> bovDim = Coord<3>::diagonal(1);
        for (int i = 0; i < DIM; ++i) {
            bovDim[i] = c[i];
        }

        Coord<3> bricDim = (brickletDim == Coord<3>()) ? bovDim : brickletDim;

        file << "TIME: " << step << "\n"
             << "DATA_FILE: " << filenameData << "\n"
             << "DATA_SIZE: "
             << bovDim.x() << " " << bovDim.y() << " " << bovDim.z() << "\n"
             << "DATA_FORMAT: " << selector.typeName() << "\n"
             << "VARIABLE: " << selector.name() << "\n"
             << "DATA_ENDIAN: LITTLE\n"
             << "BRICK_ORIGIN: 0 0 0\n"
             << "BRICK_SIZE: "
             << bovDim.x() << " " << bovDim.y() << " " << bovDim.z() << "\n"
             << "DIVIDE_BRICK: true\n"
             << "DATA_BRICKLETS: "
             << bricDim.x() << " " << bricDim.y() << " " << bricDim.z() << "\n"
             << "DATA_COMPONENTS: " << selector.arity() << "\n";

        file.close();
    }

    template<typename GRID_TYPE>
    static void writeGrid(
        const std::string& filename,
        const GRID_TYPE& grid,
        const Selector<CELL_TYPE>& selector)
    {
        std::ofstream file;
        file.open(filename.c_str(), std::ios::binary);
        if (!file.good()) {
            throw std::runtime_error("could not open output file");
        }

        std::vector<char> buffer;
        Coord<DIM> dimensions = grid.dimensions();
        std::size_t length = dimensions.x();
        std::size_t byteSize = length * selector.sizeOfExternal();
        buffer.resize(byteSize);

        CoordBox<DIM> boundingBox = grid.boundingBox();
        for (typename CoordBox<DIM>::StreakIterator i = boundingBox.beginStreak();
             i != boundingBox.endStreak();
             ++i) {
            Streak<DIM> s(*i);

            Region<DIM> tempRegion;
            tempRegion << s;
            grid.saveMemberUnchecked(&buffer[0], MemoryLocation::HOST, selector, tempRegion);

            file.write(
                &buffer[0],
                byteSize);
        }

        file.close();
    }

    static void writeRegion(
        const std::string& prefix,
        const std::string& variableName,
        const Coord<DIM>& dim,
        int value = 1,
        int time = 0)
    {
        writeHeader(
            prefix + ".bov",
            prefix + ".data",
            time,
            dim,
            dim,
            Selector<float>(variableName));
    }
};

}

#endif

