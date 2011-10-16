#include <libgeodecomp/examples/latticegas/interactivesimulator.h>

__constant__ char paletteDev[256][3];

__device__ unsigned *frameDev;
__device__ unsigned *imageDev;
__device__ BigCell *gridOldDev;
__device__ BigCell *gridNewDev;

__device__ unsigned bigCellToColor(const BigCell& c)
{
        unsigned r = 0;
        unsigned g = 0;
        unsigned b = 0;

        for (int y = 0; y < 2; ++y) {
            if (c.cells[y].state != Cell::liquid) {
                r += 255;
                g += 255;
                b += 255;
            } else {
                for (int i = 0; i < 7; ++i) {
                    int col = c.cells[y].particles[i];
                    r += paletteDev[col][0];
                    g += paletteDev[col][1];
                    b += paletteDev[col][2];
                }
            }
        }

        if (r > 255)
            r = 255;
        if (g > 255)
            g = 255;
        if (b > 255)
            b = 255;

        return (0xff << 24) +
            (r << 16) +
            (g << 8) +
            (b << 0);
}

__global__ void cellsToFrame(BigCell *grid, unsigned *frame)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = blockDim.x * gridDim.x;
    int offset = y * width + x;
    
    frame[offset] = bigCellToColor(grid[offset]);
}

__global__ void scaleFrame(unsigned *frame, unsigned *image, int sourceWidth, int sourceHeight)
{
    // fixme: use textures here?
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int width  = blockDim.x * gridDim.x;
    int height = blockDim.y * gridDim.y;

    int offset = y * width + x;

    int sourceX = x * (sourceWidth  - 1) / width;
    int sourceY = y * (sourceHeight - 1) / height;
    int sourceOffset = sourceY * sourceWidth + sourceX;

    image[offset] = frame[sourceOffset];
}

void checkCudaError()
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        const char *errorMessage = cudaGetErrorString(error);
        std::cerr << "CUDA: " << errorMessage << "\n";
        throw std::runtime_error("CUDA call failed");
    }
}

InteractiveSimulator::InteractiveSimulator(QObject *parent) :
    QObject(parent),
    t(0),
    states(SimParams::modelSize, Cell::liquid),
    gridOld(SimParams::modelSize),
    gridNew(SimParams::modelSize),
    frame(SimParams::modelSize)
{
    // add initial cells
    for (int y = 5; y < SimParams::modelHeight - 5; y += 10) {
        for (int x = 5; x < SimParams::modelWidth - 5; x += 10) {
            gridOld[y  * SimParams::modelWidth + x][0] = Cell(Cell::liquid, Cell::R, 1);
        }
    }

    cudaMalloc(&frameDev, SimParams::modelSize * 4);
    cudaMalloc(&imageDev, SimParams::maxImageSize * 4);
    cudaMalloc(&gridOldDev, SimParams::modelSize * sizeof(BigCell));
    cudaMalloc(&gridNewDev, SimParams::modelSize * sizeof(BigCell));
    cudaMemcpyToSymbol(paletteDev, Cell::palette, sizeof(Cell::palette));
    checkCudaError();
}

InteractiveSimulator::~InteractiveSimulator()
{
    cudaFree(frameDev);
    cudaFree(imageDev);
    cudaFree(gridOldDev);    
    cudaFree(gridNewDev);

}

void InteractiveSimulator::renderImage(unsigned *image, unsigned width, unsigned height)
{
    outputFrame = image;
    outputFrameWidth = width;
    outputFrameHeight = height;
    newOutputFrameRequested.release(1);
    newOutputFrameAvailable.acquire(1);
}

void InteractiveSimulator::step()
{
    if (newCameraFrame.tryAcquire()) {
        // std::cout << "  states -> cells\n";
        for (int y = 0; y < SimParams::modelHeight; ++y) {
            for (int x = 0; x < SimParams::modelWidth; ++x) {
                unsigned pos = y * SimParams::modelWidth + x;
                gridOld[pos][0].getState() = states[pos];
                gridOld[pos][1].getState() = states[pos];
            }
        }
    }

    if (newOutputFrameRequested.tryAcquire()) {
        cudaMemcpy(gridOldDev, &gridOld[0], SimParams::modelSize * sizeof(BigCell), cudaMemcpyHostToDevice);
        {
            dim3 blockDim(SimParams::threads, 1);
            dim3 gridDim(SimParams::modelWidth / SimParams::threads, SimParams::modelHeight);
            cellsToFrame<<<gridDim, blockDim>>>(gridOldDev, frameDev);
        }
        {
            dim3 blockDim(outputFrameWidth, 1);
            dim3 gridDim(1, outputFrameHeight);
            scaleFrame<<<gridDim, blockDim>>>(frameDev, imageDev, SimParams::modelWidth, SimParams::modelHeight);
        }
        cudaMemcpy(outputFrame, imageDev, outputFrameWidth * outputFrameHeight * 4, cudaMemcpyDeviceToHost);
        checkCudaError();
        newOutputFrameAvailable.release();
    }

    for (int y = 1; y < SimParams::modelHeight - 1; ++y) {
        for (int x = 1; x < SimParams::modelWidth - 1; ++x) {
            unsigned pos = y * SimParams::modelWidth + x;
            gridNew[pos].update(t,
                                Cell::simpleRand(pos + t),
                                &gridOld[pos - SimParams::modelWidth],
                                &gridOld[pos],
                                &gridOld[pos + SimParams::modelWidth]);
        }
    }
    std::swap(gridNew, gridOld);

    incFrames();
    ++t;
}
