#ifndef _libgeodecomp_examples_latticegas_simparams_h_
#define _libgeodecomp_examples_latticegas_simparams_h_

class SimParams
{
public:
    static void initParams(int argc, char **argv)
    {
        // fixme: parse command line!

        modelWidth = 1024;
        modelHeight = 512;
        modelSize = modelWidth * modelHeight;
        maxImageSize = 2048 * 2048;
        cudaDevice = 0;

        // must be >= 0 and <= 64
        effluxSize = 64;

        colorSwitchCycles = 2048;

        // andis test mode
        // testCamera = false;
        // fakeCamera = true;
        // dumpFrames = true;
        // debug = false;
        // weightR = 0.0070;
        // weightG = 0.0070;
        // weightB = 0.0020;

        // prime time mode
        testCamera = false;
        fakeCamera = false;
        dumpFrames = false;
        debug = false;
        weightR = 0.0060;
        weightG = 0.0060;
        weightB = 0.0010;

        threads = 512;
    }
    
    static unsigned modelWidth;
    static unsigned modelHeight;
    static unsigned modelSize;
    static unsigned threads;
    static unsigned maxImageSize;
    static unsigned cudaDevice;
    static unsigned effluxSize;
    static unsigned colorSwitchCycles;
    static bool fakeCamera;
    static bool testCamera;
    static bool dumpFrames;
    static bool debug;
    static float weightR;
    static float weightG;
    static float weightB;
};

#endif
