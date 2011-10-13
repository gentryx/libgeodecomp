#include <libgeodecomp/examples/latticegas/framegrabber.h>
#include <cv.h>
#include <highgui.h>

FrameGrabber::FrameGrabber(QObject *parent) :
    QObject(parent),
    capture((void*)cvCaptureFromCAM(CV_CAP_ANY))
{
    if (!capture)
        throw std::runtime_error("could not access any capture devices");
}


FrameGrabber::~FrameGrabber()
{
    cvReleaseCapture((CvCapture**)&capture);
}


void FrameGrabber::grab()
{
    unsigned *realFrame = 0;
    IplImage *frame = cvQueryFrame((CvCapture*)capture);
    if (!frame) 
        throw std::runtime_error("could not capture frame");
        
    std::cout << "Capture:\n"
              << "  nChannels: " << frame->nChannels << "\n"
              << "  depth: " << frame->depth << "\n"
              << "  IPL_DEPTH_8U:  " << IPL_DEPTH_8U  << "\n"
              << "  IPL_DEPTH_8S:  " << IPL_DEPTH_8S << "\n"
              << "  IPL_DEPTH_16U: " << IPL_DEPTH_16U  << "\n"
              << "  IPL_DEPTH_16S: " << IPL_DEPTH_16S << "\n"
              << "  IPL_DEPTH_32S: " << IPL_DEPTH_32S << "\n"
              << "  IPL_DEPTH_32F: " << IPL_DEPTH_32F  << "\n"
              << "  IPL_DEPTH_64F: " << IPL_DEPTH_64F << "\n"
              << "  dataOrder: " << frame->dataOrder << "\n"
              << "  width: " << frame->width << "\n"
              << "  height: " << frame->height << "\n"
              << "  imageSize: " << frame->imageSize << "\n"
              << "  widthStep: " << frame->widthStep << "\n\n";
    emit newFrame(realFrame);
    incFrames();
}

