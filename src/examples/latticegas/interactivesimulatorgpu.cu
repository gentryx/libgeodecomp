#include <libgeodecomp/examples/latticegas/interactivesimulatorgpu.h>

__constant__ SimParams simParamsDev;

// fixme: make these members of the simulator, somehow
__device__ unsigned *frameDev;
__device__ unsigned *imageDev;
__device__ char *statesDev;
__device__ BigCell *gridOldDev;
__device__ BigCell *gridNewDev;

__global__ void cellsToFrame(BigCell *grid, unsigned *frame)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = blockDim.x * gridDim.x;
    int offset = y * width + x;
    
    frame[offset] = grid[offset].toColor(&simParamsDev);
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

__global__ void updateGrid(unsigned t, BigCell *gridOld, BigCell *gridNew)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = blockDim.x * gridDim.x;
    int offset = y * width + x;
    gridNew[offset].update(
        &simParamsDev,
        t,
        &gridOld[offset - simParamsDev.modelWidth],
        &gridOld[offset],
        &gridOld[offset + simParamsDev.modelWidth]);
}

__global__ void setStates(char *states, BigCell *grid)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = blockDim.x * gridDim.x;
    int offset = y * width + x;

    grid[offset][0].getState() = states[offset];
    grid[offset][1].getState() = states[offset];
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

InteractiveSimulatorGPU::InteractiveSimulatorGPU(QObject *parent) :
    InteractiveSimulator(parent)
{
    std::vector<BigCell> grid(simParamsHost.modelSize);
    // fixme: put this in the initializer
    // add initial cells
    for (int y = 5; y < simParamsHost.modelHeight - 5; y += 10) {
        for (int x = 5; x < simParamsHost.modelWidth - 5; x += 10) {
            grid[y  * simParamsHost.modelWidth + x][0] = Cell(Cell::liquid, Cell::R, 1);
        }
    }
    cudaMemcpy(gridOldDev, &grid[0], simParamsHost.modelSize * sizeof(BigCell), 
               cudaMemcpyHostToDevice);

    cudaMalloc(&frameDev, simParamsHost.modelSize * 4);
    cudaMalloc(&imageDev, simParamsHost.maxImageSize * 4);
    cudaMalloc(&statesDev, simParamsHost.modelSize);
    cudaMalloc(&gridOldDev, simParamsHost.modelSize * sizeof(BigCell));
    cudaMalloc(&gridNewDev, simParamsHost.modelSize * sizeof(BigCell));
    cudaMemcpyToSymbol(&simParamsDev, &simParamsHost, sizeof(SimParams));
    checkCudaError();
}

InteractiveSimulatorGPU::~InteractiveSimulatorGPU()
{
    cudaFree(frameDev);
    cudaFree(imageDev);
    cudaFree(statesDev);
    cudaFree(gridOldDev);    
    cudaFree(gridNewDev);
}

void InteractiveSimulatorGPU::loadStates()
{
    dim3 blockDim(simParamsHost.threads, 1);
    dim3 gridDim(simParamsHost.modelWidth / simParamsHost.threads, simParamsHost.modelHeight);
    cudaMemcpy(statesDev, &states[0], simParamsHost.modelSize, 
               cudaMemcpyHostToDevice);
    setStates<<<gridDim, blockDim>>>(statesDev, gridOldDev);
}

void InteractiveSimulatorGPU::renderOutput()
{
    dim3 blockDim1(simParamsHost.threads, 1);
    dim3 gridDim1(simParamsHost.modelWidth / simParamsHost.threads, 
                  simParamsHost.modelHeight);
    cellsToFrame<<<gridDim1, blockDim1>>>(gridOldDev, frameDev);

    dim3 blockDim2(outputFrameWidth, 1);
    dim3 gridDim2(1, outputFrameHeight);
    scaleFrame<<<gridDim2, blockDim2>>>(
        frameDev, imageDev, simParamsHost.modelWidth, simParamsHost.modelHeight);

    cudaMemcpy(outputFrame, imageDev, outputFrameWidth * outputFrameHeight * 4, 
               cudaMemcpyDeviceToHost);
    checkCudaError();
}

void InteractiveSimulatorGPU::update()
{
    dim3 blockDim(simParamsHost.threads, 1);
    dim3 gridDim(simParamsHost.modelWidth / simParamsHost.threads, simParamsHost.modelHeight);
    updateGrid<<<gridDim, blockDim>>>(t, gridOldDev, gridNewDev);
    std::swap(gridNewDev, gridOldDev);
}
