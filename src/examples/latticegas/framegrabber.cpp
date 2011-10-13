#include <libgeodecomp/examples/latticegas/framegrabber.h>
#include <cv.h>
#include <highgui.h>

FrameGrabber::FrameGrabber(bool _fakeCam, QObject *parent) :
    QObject(parent),
    capture((void*)cvCaptureFromCAM(CV_CAP_ANY)),
    fakeCam(_fakeCam)
{
    if (!fakeCam && !capture)
        throw std::runtime_error("could not access any capture devices");
}

FrameGrabber::~FrameGrabber()
{
    cvReleaseCapture((CvCapture**)&capture);
}

void FrameGrabber::grab()
{
    incFrames();

    if (fakeCam) {
        const int MAX_X = 400;
        const int MAX_Y = 300;
        unsigned *frame = new unsigned[MAX_X * MAX_Y];
        for (int y = 0; y < MAX_Y; ++y) {
            for (int x = 0; x < MAX_X; ++x) {
                // double r = y > (MAX_Y / 2) ? MAX_Y - y : y;
                // double g = x > (MAX_X / 2) ? MAX_X - x : x;
                // int fixedR = 255.0 * r * (1.0 / (MAX_Y / 2));
                // int fixedG = 255.0 * g * (1.0 / (MAX_X / 2));

                unsigned char r =  255 * x / MAX_X;
                unsigned char g =  255 * y / MAX_Y;
                
                unsigned val = (0xff << 24) + (r << 16) + (g << 8) + 0;
                frame[y * MAX_X + x] = val;
            }
        }
                
        emit newFrame(frame, MAX_X, MAX_Y);
        return;
    }

    unsigned *frame = 0;
    IplImage *rawFrame = cvQueryFrame((CvCapture*)capture);
    if (!rawFrame) 
        throw std::runtime_error("could not capture frame");
        
    std::cout << "Capture:\n"
              << "  nChannels: " << rawFrame->nChannels << "\n"
              << "  depth: " << rawFrame->depth << "\n"
              << "  IPL_DEPTH_8U:  " << IPL_DEPTH_8U  << "\n"
              << "  IPL_DEPTH_8S:  " << IPL_DEPTH_8S << "\n"
              << "  IPL_DEPTH_16U: " << IPL_DEPTH_16U  << "\n"
              << "  IPL_DEPTH_16S: " << IPL_DEPTH_16S << "\n"
              << "  IPL_DEPTH_32S: " << IPL_DEPTH_32S << "\n"
              << "  IPL_DEPTH_32F: " << IPL_DEPTH_32F  << "\n"
              << "  IPL_DEPTH_64F: " << IPL_DEPTH_64F << "\n"
              << "  dataOrder: " << rawFrame->dataOrder << "\n"
              << "  width: " << rawFrame->width << "\n"
              << "  height: " << rawFrame->height << "\n"
              << "  imageSize: " << rawFrame->imageSize << "\n"
              << "  widthStep: " << rawFrame->widthStep << "\n\n";
    emit newFrame(frame, rawFrame->width, rawFrame->height);
}

