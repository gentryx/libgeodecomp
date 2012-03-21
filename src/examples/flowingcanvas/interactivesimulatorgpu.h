#ifndef _libgeodecomp_examples_flowingcanvas_interactivesimulatorgpu_h_
#define _libgeodecomp_examples_flowingcanvas_interactivesimulatorgpu_h_

#include <libgeodecomp/examples/flowingcanvas/interactivesimulator.h>
#include <libgeodecomp/misc/grid.h>

// fixme: kernels need to be templates. should be included by GPUSimulator
__global__ void updateKernel(CanvasCell *curGrid, CanvasCell *newGrid, unsigned width, unsigned nanoStep)
{
    int x = 1 + blockDim.x * blockIdx.x + threadIdx.x;
    int y = 1 + blockDim.y * blockIdx.y + threadIdx.y;
    int index = y * width + x;

    newGrid[index].update(
        curGrid + index - width, 
        curGrid + index,
        curGrid + index + width,
        nanoStep);
}

__global__ void loadGridFromTransferBuffer(CanvasCell *grid, CanvasCell *buffer, unsigned widthGrid, unsigned widthBuffer)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int indexGrid   = (y + 1) * widthGrid + x + 1;
    int indexBuffer = y * widthBuffer + x;

    grid[indexGrid] = buffer[indexBuffer];
}

__global__ void storeGridToTransferBuffer(CanvasCell *grid, CanvasCell *buffer, unsigned widthGrid, unsigned widthBuffer)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int indexGrid   = (y + 1) * widthGrid + x + 1;
    int indexBuffer = y * widthBuffer + x;

    buffer[indexBuffer] = grid[indexGrid];
}


__global__ void updateCam(CanvasCell *grid, unsigned char *cameraBuffer, float factorX, float factorY, unsigned widthGrid, unsigned widthBuffer)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int indexGrid   = (y + 1) * widthGrid + x + 1;
    int indexBuffer = (int)(y * factorY) * widthBuffer + x * factorX;
    unsigned char *pixel = cameraBuffer + indexBuffer * 3;
   
    grid[indexGrid].readCam(pixel[0], pixel[1], pixel[2]);
}

namespace LibGeoDecomp {

// fixme: this is hardcoded to 2d ATM
// fixme: move to dedicated file
template<typename CELL_TYPE>
class GPUSimulator : public MonolithicSimulator<CELL_TYPE>
{
public:
    typedef typename CELL_TYPE::Topology Topology;
    typedef Grid<CELL_TYPE, Topology> GridType;
    static const int DIM = Topology::DIMENSIONS;

    GPUSimulator(Initializer<CELL_TYPE> *initializer, const int& device = 0) :
        MonolithicSimulator<CELL_TYPE>(initializer),
        inputBufferDevice(0),
        inputBufferSize(0)
    {
        Coord<DIM> dim = this->initializer->gridDimensions();
        gridHost.resize(dim);
        this->initializer->grid(&gridHost);

        cudaSetDevice(device);
        int byteSize = dim.prod() * sizeof(CELL_TYPE);
        cudaMalloc(&transferGridDevice, byteSize);
        cudaMemcpy(transferGridDevice, gridHost.baseAddress(), byteSize, cudaMemcpyHostToDevice);

        // pad actual grids to avoid edge cell handling
        Coord<2> paddedDim = dim + Coord<2>(2, 2);
        byteSize = paddedDim.prod() * sizeof(CELL_TYPE);
        cudaMalloc(&curGridDevice, byteSize);
        cudaMalloc(&newGridDevice, byteSize);
        GridType initGrid(paddedDim, gridHost.getEdgeCell(), gridHost.getEdgeCell());
        cudaMemcpy(curGridDevice, initGrid.baseAddress(), byteSize, cudaMemcpyHostToDevice);
        cudaMemcpy(newGridDevice, initGrid.baseAddress(), byteSize, cudaMemcpyHostToDevice);

        dim3 blockDim;
        dim3 gridDim;
        genKernelDimensions(&blockDim, &gridDim);
        loadGridFromTransferBuffer<<<gridDim, blockDim>>>(curGridDevice, transferGridDevice, gridWidth() + 2, gridWidth());

        allocInputBuffer(Coord<2>(1, 1));
        checkCudaError();
    }

    virtual ~GPUSimulator()
    {
        cudaFree(transferGridDevice);
        cudaFree(curGridDevice);
        cudaFree(newGridDevice);
        cudaFree(inputBufferDevice);
    }

    virtual void step()
    {
        for (unsigned i = 0; i < CELL_TYPE::nanoSteps(); i++)
            nanoStep(i);

        this->stepNum++;    
        // call back all registered Writers
        for(unsigned i = 0; i < this->writers.size(); i++) 
            this->writers[i]->stepFinished();
    }

    virtual void run()
    {
        std::cout << "fixme InteractiveSimulatorGPU::run()\n";
    }

    virtual const GridType *getGrid()
    {
        dim3 blockDim;
        dim3 gridDim;
        genKernelDimensions(&blockDim, &gridDim);
        storeGridToTransferBuffer<<<gridDim, blockDim>>>(curGridDevice, transferGridDevice, gridWidth() + 2, gridWidth());
        int byteSize = gridHost.getDimensions().prod() * sizeof(CELL_TYPE);
        cudaMemcpy(gridHost.baseAddress(), transferGridDevice, byteSize, cudaMemcpyDeviceToHost);
        checkCudaError();
        // fixme: copy only particles back
        return &gridHost;
    }

    virtual void updateInputBuffer(unsigned char *data, const Coord<DIM>& inputDim)
    {
        allocInputBuffer(inputDim);
        // fixme: set element size externally
        int byteSize = inputDim.prod() * 3 * sizeof(unsigned char);
        cudaMemcpy(inputBufferDevice, data, byteSize, cudaMemcpyHostToDevice);
  
        Coord<2> simGridDim = this->getInitializer()->gridDimensions();
        float factorX = 1.0 * inputDim.x() / simGridDim.x();
        float factorY = 1.0 * inputDim.y() / simGridDim.y();

        dim3 blockDim;
        dim3 gridDim;
        genKernelDimensions(&blockDim, &gridDim);
        updateCam<<<gridDim, blockDim>>>(curGridDevice, inputBufferDevice, factorX, factorY, gridWidth() + 2, inputDim.x());
    }

private:
    GridType gridHost;
    CELL_TYPE *transferGridDevice;
    CELL_TYPE *curGridDevice;
    CELL_TYPE *newGridDevice;
    unsigned char *inputBufferDevice;
    int inputBufferSize;

    void allocInputBuffer(const Coord<DIM>& dim)
    {
        // fixme: set element size externally
        int byteSize = dim.prod() * 3 * sizeof(unsigned char);
        if (byteSize == inputBufferSize) {
            return;
        }

        if (inputBufferDevice != 0) {
            cudaFree(inputBufferDevice);
        }

        cudaMalloc(&inputBufferDevice, byteSize);
        checkCudaError();
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

    void nanoStep(const unsigned& nanoStep)
    {
        dim3 blockDim;
        dim3 gridDim;
        genKernelDimensions(&blockDim, &gridDim);
        updateKernel<<<gridDim, blockDim>>>(curGridDevice, newGridDevice, gridWidth() + 2, nanoStep);
        checkCudaError();
        std::swap(curGridDevice, newGridDevice);
    }

    void genKernelDimensions(dim3 *blockDim, dim3 *gridDim)
    {
        Coord<DIM> dim = gridHost.getDimensions();
        int blockDimX = 32;
        int blockDimY = 8;
        *blockDim = dim3(blockDimX, blockDimY);
        *gridDim = dim3(dim.x() / blockDimX, dim.y() / blockDimY);
    }

    int gridWidth()
    {
        return gridHost.getDimensions().x();
    }
};

template<typename CELL_TYPE>
class InteractiveSimulatorGPU : public GPUSimulator<CELL_TYPE>, public InteractiveSimulator
{
public:
    typedef typename CELL_TYPE::Topology Topology;
    typedef Grid<CELL_TYPE, Topology> GridType;
    typedef std::vector<boost::shared_ptr<Writer<CELL_TYPE> > > WriterVector;
    static const int DIM = Topology::DIMENSIONS;

    InteractiveSimulatorGPU(QObject *parent, Initializer<CELL_TYPE> *initializer) :
        GPUSimulator<CELL_TYPE>(initializer),
        InteractiveSimulator(parent)
    {}

    virtual ~InteractiveSimulatorGPU()
    {}

    virtual void readCam()
    {
        Coord<DIM> dim = this->initializer->gridDimensions();
        this->updateInputBuffer(&cameraFrame[0], Coord<2>(cameraFrameWidth, cameraFrameHeight));
    }

    virtual void renderOutput()
    {
        // fixme: this is the same for InteractiveSimulatorCPU. refactor?
        for(unsigned i = 0; i < this->writers.size(); i++) 
            this->writers[i]->stepFinished();
    }

    virtual void update()
    {
        GPUSimulator<CELL_TYPE>::step();
    }

    virtual void registerWriter(Writer<CELL_TYPE> *writer)
    {
        writers.push_back(boost::shared_ptr<Writer<CELL_TYPE> >(writer));
    }

protected:
    WriterVector writers;
};

}

#endif
