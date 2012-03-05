#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_OPENCL

#ifndef _libgeodecomp_parallelization_hiparsimulator_openclstepper_h_
#define _libgeodecomp_parallelization_hiparsimulator_openclstepper_h_

#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif

#include <boost/shared_ptr.hpp>
#include <CL/cl.h>

#include <libgeodecomp/misc/cl.hpp>
#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<typename CELL_TYPE>
class OpenCLStepper : public Stepper<CELL_TYPE>
{
    friend class OpenCLStepperTest;
public:
    const static int DIM = CELL_TYPE::Topology::DIMENSIONS;

    typedef class Stepper<CELL_TYPE> ParentType;
    typedef typename ParentType::GridType GridType;
    typedef PartitionManager< 
        DIM, typename CELL_TYPE::Topology> MyPartitionManager;
  
    inline OpenCLStepper(
        const std::string& cellSourceFile,
        boost::shared_ptr<MyPartitionManager> _partitionManager,
        Initializer<CELL_TYPE> *_initializer,
        const int& platformID=0,
        const int& deviceID=0) :
        ParentType(_partitionManager, _initializer)
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        std::vector<cl::Device> devices;
        platforms[platformID].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        cl::Device usedDevice = devices[deviceID];
        context = cl::Context(devices);
        cmdQueue = cl::CommandQueue(context, usedDevice);

        std::string clSourceString = 
"#if defined(cl_khr_fp64)\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"#elif defined(cl_amd_fp64)\n"
"#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
"#endif\n"
"\n"
"#include \"" + cellSourceFile + "\"\n"
"\n"
#include <libgeodecomp/parallelization/hiparsimulator/escapedopenclkernel.h>
            ;

        cl::Program::Sources clSource(
            1, 
            std::make_pair(clSourceString.c_str(), 
                           clSourceString.size()));
        cl::Program clProgram(context, clSource);

        try {
            clProgram.build(devices);
        } catch (...) {
            // Normally we don't catch exceptions, but in this case
            // printing the build log (which might get lost otherwise)
            // is valuable for the user who needs to debug his code.
            std::cerr << "Build Log: " 
                      << clProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(usedDevice) << "\n";
            throw;
        }

        kernel = cl::Kernel(clProgram, "execute");

        // fixme:
        // curStep = initializer().startStep();
        // curNanoStep = 0;
        // initGrids();
    }

    inline virtual std::pair<int, int> currentStep() const
    {
        return std::make_pair(curStep, curNanoStep);
    }

    inline virtual void update(int nanoSteps) 
    {
        // fixme: implement me (later)
        try {
            cl::Buffer startCoordsBuffer, endCoordsBuffer;
        
            Coord<DIM> c = this->initializer->gridDimensions();
            int zDim = c.z();
            int yDim = c.y();
            int xDim = c.x();
	
            int actualX = xDim;
            int actualY = yDim;
                
            std::vector<int> startCoords;
            std::vector<int> endCoords;
                
            genThreadCoords(
                &startCoords,
                &endCoords,
                0,
                0,
                0,
                xDim,
                yDim,
                zDim,
                actualX,
                actualY,
                zDim,
                1);

            startCoordsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, startCoords.size()*sizeof(int), &startCoords[0]);
            endCoordsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, endCoords.size()*sizeof(int), &endCoords[0]);
        
            cl::NDRange global(actualX, actualY, zDim);
            //fixme: local range could be chosen dynamically
            cl::NDRange local(16, 16, 1);
        
            cl::KernelFunctor livingKernel = kernel.bind(cmdQueue, global, local);
            livingKernel(inputDeviceGrid, outputDeviceGrid, zDim, yDim, xDim,
                         1, 0, 0, 0,
                         startCoordsBuffer, endCoordsBuffer, actualX, actualY);
            livingKernel.getError();
            cmdQueue.finish();
        

        } catch (cl::Error& err) {
            std::cerr << "OpenCL error: " << err.what() << ", " << oclStrerror(err.err()) << std::endl;
            throw err;
        } catch (...) {
            throw;
        }
    }

    inline virtual const GridType& grid() const
    {
        cmdQueue.enqueueReadBuffer(
            outputDeviceGrid, true, 0, 
            hostGrid->getDimensions().prod() * sizeof(CELL_TYPE), hostGrid->baseAddress());
        return *hostGrid;
    }

private:
    int curStep;
    int curNanoStep;
    boost::shared_ptr<GridType> hostGrid;

    cl::Buffer inputDeviceGrid;
    cl::Buffer outputDeviceGrid;
    cl::Context context;
    cl::CommandQueue cmdQueue;
    cl::Kernel kernel;
    
    inline void genThreadCoords(std::vector<int> *startCoords,
			 std::vector<int> *endCoords,
			 const int& offset_x,
			 const int& offset_y,
			 const int& offset_z,
			 const int& active_x,
			 const int& active_y,
			 const int& active_z,
			 const int& actual_x,
			 const int& actual_y,
			 const int& actual_z,
			 const int& planes)
    {
      int maxX = active_x;
      int maxY = active_y;
      int maxZ = ceil(1.0 * actual_z/planes);
      int numThreads = actual_x * actual_y * maxZ;
      startCoords->resize(numThreads);
      endCoords->resize(numThreads);
    
      for (int z = 0; z < maxZ; ++z) {
        int startZ = offset_z + z * planes;
        int endZ = std::min(offset_z + active_z, 
                            startZ + planes);
        
        for (int y = 0; y < actual_y; ++y) {
	  for (int x = 0; x < actual_x; ++x) {
	    int threadID = (z * actual_x * actual_y) + (y * actual_x) + x; 
	    int myEndZ = endZ;
	    if (x >= maxX || y >= maxY)
	      myEndZ = startZ;
                
	    (*startCoords)[threadID] = startZ;
	    (*endCoords)[threadID] = myEndZ;
	  }
        }
      }
    }

    inline void initGrids()
    {
        const CoordBox<DIM>& gridBox = 
            this->partitionManager->ownRegion().boundingBox();
        hostGrid.reset(new GridType(gridBox, CELL_TYPE()));
        this->initializer->grid(&*hostGrid);
        
        inputDeviceGrid = cl::Buffer(
            context, 
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
            hostGrid->getDimensions().prod() * sizeof(CELL_TYPE), 
            hostGrid->baseAddress());
	std::vector<CELL_TYPE> zeroMem(hostGrid->getDimensions().prod(), 0);
	outputDeviceGrid = cl::Buffer(
            context, 
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
            hostGrid->getDimensions().prod() * sizeof(CELL_TYPE), 
            &zeroMem[0]);
    }

    inline std::string oclStrerror (int nr) {
      switch (nr) {
      case 0:
	return "CL_SUCCESS";
      case -1:
	return "CL_DEVICE_NOT_FOUND";
      case -2:
	return "CL_DEVICE_NOT_AVAILABLE";
      case -3:
	return "CL_COMPILER_NOT_AVAILABLE";
      case -4:
	return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
      case -5:
	return "CL_OUT_OF_RESOURCES";
      case -6:
	return "CL_OUT_OF_HOST_MEMORY";
      case -7:
	return "CL_PROFILING_INFO_NOT_AVAILABLE";
      case -8:
	return "CL_MEM_COPY_OVERLAP";
      case -9:
	return "CL_IMAGE_FORMAT_MISMATCH";
      case -10:
	return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
      case -11:
	return "CL_BUILD_PROGRAM_FAILURE";
      case -12:
	return "CL_MAP_FAILURE";
      case -13:
	return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
      case -14: 
	return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
      case -30:
	return "CL_INVALID_VALUE";
      case -31:
	return "CL_INVALID_DEVICE_TYPE";
      case -32:
	return "CL_INVALID_PLATFORM";
      case -33:
	return "CL_INVALID_DEVICE";
      case -34:
	return "CL_INVALID_CONTEXT";
      case -35:
	return "CL_INVALID_QUEUE_PROPERTIES";
      case -36:
	return "CL_INVALID_COMMAND_QUEUE";
      case -37:
	return "CL_INVALID_HOST_PTR";
      case -38:
	return "CL_INVALID_MEM_OBJECT";
      case -39:
	return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
      case -40:
	return "CL_INVALID_IMAGE_SIZE";
      case -41:
	return "CL_INVALID_SAMPLER";
      case -42:
	return "CL_INVALID_BINARY";
      case -43:
	return "CL_INVALID_BUILD_OPTIONS";
      case -44:
	return "CL_INVALID_PROGRAM";
      case -45:
	return "CL_INVALID_PROGRAM_EXECUTABLE";
      case -46:
	return "CL_INVALID_KERNEL_NAME";
      case -47:
	return "CL_INVALID_KERNEL_DEFINITION";
      case -48:
	return "CL_INVALID_KERNEL";
      case -49:
	return "CL_INVALID_ARG_INDEX";
      case -50:
	return "CL_INVALID_ARG_VALUE";
      case -51:
	return "CL_INVALID_ARG_SIZE";
      case -52:
	return "CL_INVALID_KERNEL_ARGS";
      case -53:
	return "CL_INVALID_WORK_DIMENSION";
      case -54:
	return "CL_INVALID_WORK_GROUP_SIZE";
      case -55:
	return "CL_INVALID_WORK_ITEM_SIZE";
      case -56:
	return "CL_INVALID_GLOBAL_OFFSET";
      case -57:
	return "CL_INVALID_EVENT_WAIT_LIST";
      case -58:
	return "CL_INVALID_EVENT";
      case -59:
	return "CL_INVALID_OPERATION";
      case -60:
	return "CL_INVALID_GL_OBJECT";
      case -61:
	return "CL_INVALID_BUFFER_SIZE";
      case -62:
	return "CL_INVALID_MIP_LEVEL";
      case -63:
	return "CL_INVALID_GLOBAL_WORK_SIZE";
      case -64:
	return "CL_INVALID_PROPERTY";
      }
      return "nothing found";
    }
};

}
}

#endif
#endif
