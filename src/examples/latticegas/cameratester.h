#ifndef _libgeodecomp_examples_latticegas_cameratester_h_
#define _libgeodecomp_examples_latticegas_cameratester_h_

#include <cerrno>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <QObject>
#include <libgeodecomp/examples/latticegas/interactivesimulator.h>

class CameraTester : public QObject
{
    Q_OBJECT

public:
    CameraTester() :
        t(0)
    {}

public slots:
    void updateCam(char *rawFrame, unsigned width, unsigned height)
    {
        std::ostringstream filename;
        filename << "cameraTest." << std::setfill('0') << std::setw(4) << t;
        writeImage(filename.str() + ".orig.ppm", rawFrame, width, height, true);
        writeImage(filename.str() + ".bin.ppm",  rawFrame, width, height, false);
        ++t;
    }

private: 
    unsigned t;

    void writeImage(std::string filename, char *rawFrame, unsigned width, unsigned height, bool passThrough)
    {
        std::ofstream outfile(filename.c_str());
        if (!outfile) 
            throw std::runtime_error("Cannot open output file");

        // header first:
        outfile << "P6 " << width 
                << " "   << height << " 255\n";
        
        for (unsigned y = 0; y < height; ++y) {
            for (unsigned x = 0; x < width; ++x) {
                int offset = y * width + x;
                char r = rawFrame[offset * 3 + 0];
                char g = rawFrame[offset * 3 + 1];
                char b = rawFrame[offset * 3 + 2];

                if (!passThrough) {
                    char state = InteractiveSimulator::pixelToState(r, g, b);
                    r = simParamsHost.palette[(int)state][0];
                    g = simParamsHost.palette[(int)state][1];
                    b = simParamsHost.palette[(int)state][2];
                }

                outfile << r << g << b;
            }
        }

        if (!outfile.good()) 
            throw std::runtime_error("Cannot write output file");
        outfile.close();
    }
};

#endif
