#ifndef LIBGEODECOMP_IO_PPMWRITER_H
#define LIBGEODECOMP_IO_PPMWRITER_H

#include <cerrno>
#include <fstream>
#include <iomanip>
#include <string>
#include <libgeodecomp/io/imagepainter.h>
#include <libgeodecomp/io/ioexception.h>
#include <libgeodecomp/io/plotter.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/storage/image.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE, typename CELL_PLOTTER>
class PPMWriter : public Writer<CELL_TYPE>
{
public:
    friend class PPMWriterTest;
    typedef typename Writer<CELL_TYPE>::GridType GridType;
    using Writer<CELL_TYPE>::period;
    using Writer<CELL_TYPE>::prefix;

    explicit PPMWriter(
        const std::string& prefix,
        const unsigned period = 1,
        const unsigned dimX = 20,
        const unsigned dimY = 20) :
        Writer<CELL_TYPE>(prefix, period),
        plotter(Coord<2>(dimX, dimY))
    {}

    virtual void stepFinished(const GridType& grid, unsigned step, WriterEvent event)
    {
        if ((event == WRITER_STEP_FINISHED) && (step % period != 0)) {
            return;
        }

        Coord<2> imageDim = plotter.calcImageDim(grid.boundingBox().dimensions);
        Image image(imageDim);
        ImagePainter painter(&image);
        plotter.plotGrid(grid, painter);
        writePPM(image, step);
    }

 private:
    Plotter<CELL_TYPE, CELL_PLOTTER> plotter;

    void writePPM(const Image& img, unsigned step)
    {
        std::ostringstream filename;
        filename << prefix << "." << std::setfill('0') << std::setw(4)
                 << step << ".ppm";
        std::ofstream outfile(filename.str().c_str());
        if (!outfile) {
            throw FileOpenException(filename.str());
        }

        // header first:
        outfile << "P6 " << img.getDimensions().x()
                << " "   << img.getDimensions().y() << " 255\n";

        // body second:
        for (int y = 0; y < img.getDimensions().y(); ++y) {
            for (int x = 0; x < img.getDimensions().x(); ++x) {
                const Color& rgb = img[y][x];
                outfile << (char)rgb.red()
                        << (char)rgb.green()
                        << (char)rgb.blue();
            }
        }

        if (!outfile.good()) {
            throw FileWriteException(filename.str());
        }
        outfile.close();
    }
};

}

#endif
