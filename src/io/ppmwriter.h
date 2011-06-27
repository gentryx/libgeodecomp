#ifndef _libgeodecomp_io_ppmwriter_h_
#define _libgeodecomp_io_ppmwriter_h_

#include <string>
#include <cerrno>
#include <fstream>
#include <iomanip>
#include <libgeodecomp/parallelization/simulator.h>
#include <libgeodecomp/io/image.h>
#include <libgeodecomp/io/ioexception.h>
#include <libgeodecomp/io/plotter.h>
#include <libgeodecomp/io/writer.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE, typename CELL_PLOTTER>
class PPMWriter : public Writer<CELL_TYPE>
{    
    friend class PPMWriterTest;

 public:
    PPMWriter(
        const std::string& prefix, 
        MonolithicSimulator<CELL_TYPE> *sim, 
        const unsigned& period = 1,
        const unsigned& dimX = 20,
        const unsigned& dimY = 20) :
        Writer<CELL_TYPE>(prefix, sim, period),
        _gridPlotter(&_cellPlotter, dimX, dimY)
    {}

    virtual void initialized()
    {
        writeStep();
    }

    virtual void stepFinished()
    {
        if (this->sim->getStep() % this->period == 0)
            writeStep();
    }

    virtual void allDone() {}

 private:
    CELL_PLOTTER _cellPlotter;
    Plotter<CELL_TYPE, CELL_PLOTTER> _gridPlotter;


    void writeStep()
    {
        writePPM(
            _gridPlotter.plotGrid(
                *this->sim->getGrid()));
    }

    void writePPM(Image img)
    {
        std::ostringstream filename;
        filename << this->prefix << "." << std::setfill('0') << std::setw(4)
                 << this->sim->getStep() << ".ppm";
        std::ofstream outfile(filename.str().c_str());
        if (!outfile) 
            throw FileOpenException("Cannot open output file", 
                                    filename.str(), errno);

        // header first:
        outfile << "P6 " << img.getDimensions().x() 
                << " "   << img.getDimensions().y() << " 255\n";

        for(unsigned y = 0; y < img.getDimensions().y(); ++y) {
            for(unsigned x = 0; x < img.getDimensions().x(); ++x) {
                Color rgb = img[y][x];
                outfile << (char)rgb.red() 
                        << (char)rgb.green() 
                        << (char)rgb.blue();
            }
        }
        if (!outfile.good()) 
            throw FileWriteException("Cannot write to output file",
                                     filename.str(), errno);
        outfile.close();
    }
};

}

#endif
