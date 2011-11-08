#include <iostream>

#include <QtGui/QApplication>
#include <QtCore/QTimer>
#include <QThreadPool>
#include <libgeodecomp/examples/latticegas/cell.h>
#include <libgeodecomp/examples/latticegas/cameratester.h>
#include <libgeodecomp/examples/latticegas/framegrabber.h>
#include <libgeodecomp/examples/latticegas/flowwidget.h>
#include <libgeodecomp/examples/latticegas/interactivesimulator.h>
#include <libgeodecomp/examples/latticegas/interactivesimulatorcpu.h>
#include <libgeodecomp/examples/latticegas/interactivesimulatorgpu.h>
#include <libgeodecomp/examples/latticegas/simparams.h>

// fixme: using namespace LibGeoDecomp;

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
                    &simParamsHost,
                    t,
                    gridA[y - 1][x + 0][1],
                    gridA[y - 1][x + 1][1],
                    gridA[y + 0][x - 1][0],
                    gridA[y + 0][x + 0][0],
                    gridA[y + 0][x + 1][0],
                    gridA[y + 0][x + 0][1],
                    gridA[y + 0][x + 1][1]);

                gridB[y][x][1].update(
                    &simParamsHost,
                    t,
                    gridA[y + 0][x - 1][0],
                    gridA[y + 0][x + 0][0],
                    gridA[y + 0][x - 1][1],
                    gridA[y + 0][x + 0][1],
                    gridA[y + 0][x + 1][1],
                    gridA[y + 1][x - 1][0],
                    gridA[y + 1][x + 0][0]);
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

    //fixme: make this configurable via simparams
    InteractiveSimulator *sim = new InteractiveSimulatorGPU(&flow);
    FrameGrabber *grabber = new FrameGrabber(simParamsHost.fakeCamera, &flow);

    QTimer *timerFlow = new QTimer(&flow);
    QTimer *timerGrab = new QTimer(&flow);
    QTimer *timerInfo = new QTimer(&flow);

    QObject::connect(timerInfo, SIGNAL(timeout()),           &flow,   SLOT(info()));
    QObject::connect(timerInfo, SIGNAL(timeout()),           grabber, SLOT(info()));
    QObject::connect(timerInfo, SIGNAL(timeout()),           sim,     SLOT(info()));
    QObject::connect(timerFlow, SIGNAL(timeout()),           &flow,   SLOT(ping()));
    QObject::connect(timerGrab, SIGNAL(timeout()),           grabber, SLOT(grab()));

    QObject::connect(grabber,   SIGNAL(newFrame(char*, unsigned, unsigned)), 
                     sim,       SLOT(updateCam( char*, unsigned, unsigned)));
    QObject::connect(&flow,     SIGNAL(updateImage(unsigned*, unsigned, unsigned)),
                     sim,       SLOT(renderImage(unsigned*, unsigned, unsigned)));
    QObject::connect(&app,      SIGNAL(lastWindowClosed()),  sim,     SLOT(quit()));

    QThreadPool *threadPool = QThreadPool::globalInstance();
    threadPool->start(sim);

    grabber->grab();
    timerFlow->start(10);
    timerGrab->start(5000);
    timerInfo->start(5000);
    flow.show();
    int ret = app.exec();
    threadPool->waitForDone();
    return ret;
}

// determine optimal weights for cutoff
void testCamera()
{
    std::cout << "hello\n";
    FrameGrabber grabber(false, 0);
    CameraTester tester;

    QObject::connect(&grabber,   SIGNAL(newFrame(char*, unsigned, unsigned)), 
                     &tester,    SLOT(updateCam( char*, unsigned, unsigned)));


    // upper and lower boundaries
    float upperR = 1.0;
    float upperG = 1.0;
    float upperB = 0.3;

    float lowerR = 0.0;
    float lowerG = 0.0;
    float lowerB = 0.0;

    for (int i = 0; i < 50; ++i) {
        simParamsHost.weightR = (upperR + lowerR) * 0.5;
        simParamsHost.weightG = (upperG + lowerG) * 0.5;
        simParamsHost.weightB = (upperB + lowerB) * 0.5;

        std::cout << " determining (weightR, weightG, weightB)\n"
		  << " current: (" 
		  << simParamsHost.weightR << ", " 
		  << simParamsHost.weightG << ", "
	          << simParamsHost.weightB << ")\n";
        grabber.grab();

	for (;;) {
	    std::cout << "is this too dark? (y/n)\n";
	    std::string answer;
	    std::cin >> answer;
	    if (answer == "y") {
	        upperR = simParamsHost.weightR;
		upperG = simParamsHost.weightG;
		upperB = simParamsHost.weightB;
		break;
            }

	    if (answer == "n") {
	        lowerR = simParamsHost.weightR;
		lowerG = simParamsHost.weightG;
		lowerB = simParamsHost.weightB;
		break;
	    }

	    std::cout << "please answer \"y\" or \"n\"\n";
	}
	std::cout << "\n";
    }
    std::cout << "bye\n";
}

int main(int argc, char **argv)
{
    Cell::init();
    simParamsHost.initParams(argc, argv);
    cudaSetDevice(simParamsHost.cudaDevice);

    if (simParamsHost.testCamera) {
      testCamera();
      // testModel();
      return 0;
    } else {
      return runQtApp(argc, argv);
    }
 }
