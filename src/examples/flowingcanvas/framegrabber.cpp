#include <cv.h>
#include <highgui.h>
#include <libgeodecomp/examples/flowingcanvas/framegrabber.h>

FrameGrabber::FrameGrabber(bool _fakeCam, QObject *parent) :
    QObject(parent),
    capture((void*)cvCaptureFromCAM(CV_CAP_ANY)),
    fakeCam(_fakeCam),
    time(0)
{
    std::cout << "fakeCam: " << fakeCam << "\n";
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
        ++time;
        if (time > 50) {
            time = 0;
        }

        const int MAX_X = 400;
        const int MAX_Y = 300;

        int offsetX = time * 10 - 50;
        int offsetY = 100;
        if (time > 25)
            offsetY += (time - 25) * 10;

        std::vector<char> frame(MAX_X * MAX_Y * 3);
        for (int y = 0; y < MAX_Y; ++y) {
            for (int x = 0; x < MAX_X; ++x) {
                int r = 0xff;
                int g = 0xff;
                int b = 0xff;
                
                if ((y >= offsetY) &&(y <= (offsetY + 100)) && (x >= offsetX) && (x <= (offsetX + 50)) ) {
                    r = 1;
                    g = 0x8f;
                    b = 0x00;
                }
                
                frame[(y * MAX_X + x) * 3 + 0] = r;
                frame[(y * MAX_X + x) * 3 + 1] = g;
                frame[(y * MAX_X + x) * 3 + 2] = b;
            }
        }
                
        emit newFrame(&frame[0], MAX_X, MAX_Y);
        return;
    }

    IplImage *frame = cvQueryFrame((CvCapture*)capture);
    if (!frame) 
        throw std::runtime_error("could not capture frame");

    if (frame->depth != 8)
        throw std::runtime_error("unexpected color depth");

    if (frame->nChannels != 3)
        throw std::runtime_error("unexpected number of channels");

    // fixme:
    // if (simParamsHost.debug)
    //   std::cerr << "Capture:\n"
	// 	<< "  nChannels: " << frame->nChannels << "\n"
	// 	<< "  depth: " << frame->depth << "\n"
	// 	<< "  IPL_DEPTH_8U:  " << IPL_DEPTH_8U  << "\n"
	// 	<< "  IPL_DEPTH_8S:  " << IPL_DEPTH_8S << "\n"
	// 	<< "  IPL_DEPTH_16U: " << IPL_DEPTH_16U  << "\n"
	// 	<< "  IPL_DEPTH_16S: " << IPL_DEPTH_16S << "\n"
	// 	<< "  IPL_DEPTH_32S: " << IPL_DEPTH_32S << "\n"
	// 	<< "  IPL_DEPTH_32F: " << IPL_DEPTH_32F  << "\n"
	// 	<< "  IPL_DEPTH_64F: " << IPL_DEPTH_64F << "\n"
	// 	<< "  dataOrder: " << frame->dataOrder << "\n"
	// 	<< "  width: " << frame->width << "\n"
	// 	<< "  height: " << frame->height << "\n"
	// 	<< "  imageSize: " << frame->imageSize << "\n"
	// 	<< "  widthStep: " << frame->widthStep << "\n\n";

    emit newFrame(frame->imageData, frame->width, frame->height);
}

