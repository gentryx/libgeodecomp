#include <iostream>

#include <QtGui/QApplication>
#include <QtCore/QTimer>
#include <QThreadPool>
#include <libgeodecomp/examples/latticegas/cell.h>
#include <libgeodecomp/examples/latticegas/framegrabber.h>
#include <libgeodecomp/examples/latticegas/flowwidget.h>
#include <libgeodecomp/examples/latticegas/interactivesimulator.h>
#include <libgeodecomp/examples/latticegas/simparams.h>

// fixme: using namespace LibGeoDecomp;

__global__ void plasma1(int *image, int offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * 1024 + x;
    image[idx + 0] = 
        (0xff << 24) +
        ((x & 0xff) << 16) +
        (((y * 2 + offset) & 0xff) << 8) +
        (128 << 0);
}

int simpleRand(int i) {
    return i * 69069 + 1327217885;
}

void testModel()
{
    int width = 7;
    int height = 6;
    Cell gridA[height][width][2];
    Cell gridB[height][width][2];
    // gridA[1][1][0] = Cell(Cell::liquid, Cell::R);
    // gridA[1][1][0] = Cell(Cell::liquid, Cell::LR);
    // gridA[1][1][1] = Cell(Cell::liquid, Cell::C);
    // gridA[1][5][0] = Cell(Cell::liquid, Cell::LL);
    // gridA[4][1][0] = Cell(Cell::liquid, Cell::UR);
    // gridA[4][5][1] = Cell(Cell::liquid, Cell::UL);
    // gridA[4][5][1] = Cell(Cell::liquid, Cell::L);

    gridA[3][1][0] = Cell(Cell::liquid, Cell::R);
    gridA[3][3][0] = Cell(Cell::liquid, Cell::L);


    for (int t = 0; t < 5; ++t) {
        std::cout << "t: " << t << "\n";

        for (int y = 0; y < height; ++y) {
            for (int ly = 0; ly < 2; ++ly) {
                std::cout << "(" << y << ", " << ly << ") ";
                for (int x = 0; x < width; ++x) {
                    std::cout << gridA[y][x][ly].toString();
                }
                std::cout << "\n";
            }
        }

        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                gridB[y][x][0].update(
                    simpleRand(x + y * 2 * width + 0 * width + t),
                    gridA[y + 0][x + 0][0].getState(),
                    gridA[y - 1][x + 0][1][Cell::LR],
                    gridA[y - 1][x + 1][1][Cell::LL],
                    gridA[y + 0][x - 1][0][Cell::R ],
                    gridA[y + 0][x + 0][0][Cell::C ],
                    gridA[y + 0][x + 1][0][Cell::L ],
                    gridA[y + 0][x + 0][1][Cell::UR],
                    gridA[y + 0][x + 1][1][Cell::UL]);

                gridB[y][x][1].update(
                    simpleRand(x + y * 2 * width + 1 * width + t),
                    gridA[y + 0][x + 0][1].getState(),
                    gridA[y + 0][x - 1][0][Cell::LR],
                    gridA[y + 0][x + 0][0][Cell::LL],
                    gridA[y + 0][x - 1][1][Cell::R ],
                    gridA[y + 0][x + 0][1][Cell::C ],
                    gridA[y + 0][x + 1][1][Cell::L ],
                    gridA[y + 1][x - 1][0][Cell::UR],
                    gridA[y + 1][x + 0][0][Cell::UL]);
            }
        }


        for (int y = 0; y < height; ++y) {
            for (int ly = 0; ly < 2; ++ly) {
                for (int x = 0; x < width; ++x) {
                    gridA[y][x][ly] = gridB[y][x][ly];
                }
            }
        }

        std::cout << "\n";
    }
}

int runQtApp(int argc, char **argv)
{
    QApplication app(argc, argv);
    FlowWidget flow;
    flow.resize(1200, 900);

    InteractiveSimulator *sim = new InteractiveSimulator(&flow);
    FrameGrabber *grabber = new FrameGrabber(true, &flow);

    QTimer *timerFlow = new QTimer(&flow);
    QTimer *timerGrab = new QTimer(&flow);
    QTimer *timerInfo = new QTimer(&flow);

    // cudaSetDevice(0);
    // int *imageDev;

    // dim3 gridDim(2, 768);
    // dim3 blockDim(512, 1);
    // long imageWidth = gridDim.x * blockDim.x;
    // long imageHeight = gridDim.y * blockDim.y;
    // long size = imageWidth * imageHeight;
    // long byteSize = size * 4;
    
    // cudaMalloc(&imageDev, byteSize);
    // plasma1<<<gridDim, blockDim>>>(imageDev, 0);
    // cudaDeviceSynchronize();
    // cudaMemcpy(flow.getImage(), imageDev, byteSize, cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // cudaFree(imageDev);

    QObject::connect(timerInfo, SIGNAL(timeout()),           &flow,   SLOT(info()));
    QObject::connect(timerInfo, SIGNAL(timeout()),           grabber, SLOT(info()));
    QObject::connect(timerInfo, SIGNAL(timeout()),           sim,     SLOT(info()));
    QObject::connect(timerFlow, SIGNAL(timeout()),           &flow,   SLOT(ping()));
    QObject::connect(timerGrab, SIGNAL(timeout()),           grabber, SLOT(grab()));

    QObject::connect(grabber,   SIGNAL(newFrame(unsigned*, unsigned, unsigned)), 
                     sim,       SLOT(updateCam( unsigned*, unsigned, unsigned)));
    QObject::connect(&flow,     SIGNAL(updateImage(unsigned*, unsigned, unsigned)),
                     sim,       SLOT(renderImage(unsigned*, unsigned, unsigned)));
    QObject::connect(&app,      SIGNAL(lastWindowClosed()),  sim,     SLOT(quit()));

    QThreadPool *threadPool = QThreadPool::globalInstance();
    threadPool->start(sim);

    timerFlow->start(10);
    timerGrab->start(1000);
    timerInfo->start(5000);
    flow.show();
    int ret = app.exec();
    threadPool->waitForDone();
    return ret;
}

int main(int argc, char **argv)
{
    std::cout << "ok\n";
    Cell::initTransportTable();
    Cell::initPalette();
    SimParams::initParams(argc, argv);

    // testModel();

    return runQtApp(argc, argv);
 }
