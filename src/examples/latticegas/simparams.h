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
	testCamera = false;
        fakeCamera = true;
	debug = false;

        // good for fake camera mode
        weightR = 0.0080;
        weightG = 0.0070;
        weightB = 0.0020;
        
        // good for real camera images
        // weightR = 0.0060;
        // weightG = 0.0060;
        // weightB = 0.0010;
        threads = 512;
    }
    
    static unsigned modelWidth;
    static unsigned modelHeight;
    static unsigned modelSize;
    static unsigned threads;
    static unsigned maxImageSize;
    static unsigned cudaDevice;
    static bool fakeCamera;
    static bool testCamera;
    static bool debug;
    static float weightR;
    static float weightG;
    static float weightB;
};

#endif
