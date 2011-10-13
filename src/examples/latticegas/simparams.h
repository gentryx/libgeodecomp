#ifndef _libgeodecomp_examples_latticegas_simparams_h_
#define _libgeodecomp_examples_latticegas_simparams_h_

class SimParams
{
public:
    static void initParams(int argc, char **argv)
    {
        // fixme: parse command line!

        modelWidth = 1000;
        modelHeight = 400;
        modelSize = modelWidth * modelHeight;
        weightR = 0.0030;
        weightG = 0.0020;
        weightB = 0.0020;
    }
    
    static unsigned modelWidth;
    static unsigned modelHeight;
    static unsigned modelSize;
    static float weightR;
    static float weightG;
    static float weightB;
};

#endif
