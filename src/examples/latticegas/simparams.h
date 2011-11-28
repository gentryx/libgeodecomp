#ifndef _libgeodecomp_examples_latticegas_simparams_h_
#define _libgeodecomp_examples_latticegas_simparams_h_

class SimParams
{
public:
    void initParams(int argc, char **argv)
    {
        // fixme: parse command line!

        modelWidth = 512;
        modelHeight = 256;
        modelSize = modelWidth * modelHeight;
        maxImageSize = 2048 * 2048;
        cudaDevice = 0;

        // must be >= 0 and <= 64
        effluxSize = 32;

        colorSwitchCycles = 512;

        // andis test mode
        testCamera = false;
        fakeCamera = true;
        dumpFrames = false;
        debug = false;
        weightR = 0.0070;
        weightG = 0.0070;
        weightB = 0.0020;

        // prime time mode
        // testCamera = false;
        // fakeCamera = false;
        // dumpFrames = false;
        // debug = false;
        // weightR = 0.0060;
        // weightG = 0.0060;
        // weightB = 0.0010;

        threads = 512;
    }
    
    unsigned modelWidth;
    unsigned modelHeight;
    unsigned modelSize;
    unsigned threads;
    unsigned maxImageSize;
    unsigned cudaDevice;
    unsigned effluxSize;
    unsigned colorSwitchCycles;
    bool fakeCamera;
    bool testCamera;
    bool dumpFrames;
    bool debug;
    float weightR;
    float weightG;
    float weightB;

    // defines for each of the 2^7 flow states which particle moves to
    // which position. stores four variants since the FHP-II model
    // sometimes requires a probabilistic selection. Don't pad to 8
    // bytes to reduce bank conflicts on Nvidia GPUs.
    char transportTable[128][4][7];
    unsigned char palette[256][3];
    int randomFields[3][1024];
};

extern SimParams simParamsHost;

#endif
