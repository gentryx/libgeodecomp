/* vim:set expandtab tabstop=2 shiftwidth=2 softtabstop=2: */

#include <fstream>
#include <iostream>
#include <libgeodecomp/libgeodecomp.h>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

using namespace LibGeoDecomp;

std::string
get_error_description(cl_int error);

std::ostream& operator<<(std::ostream& o, cl::Platform p) {
  o << "CL_PLATFORM_VERSION\t= "     << p.getInfo<CL_PLATFORM_VERSION>()
    << std::endl
    << "CL_PLATFORM_NAME\t= "        << p.getInfo<CL_PLATFORM_NAME>()
    << std::endl
    << "CL_PLATFORM_VENDOR\t= "      << p.getInfo<CL_PLATFORM_VENDOR>()
    << std::endl
    << "CL_PLATFORM_EXTENSIONS\t= "  << p.getInfo<CL_PLATFORM_EXTENSIONS>();
  return o;
}

std::ostream& operator<<(std::ostream& o, cl::Device d) {
  o << "CL_DEVICE_EXTENSIONS\t\t\t= "
    << d.getInfo<CL_DEVICE_EXTENSIONS>()                << std::endl
    << "CL_DEVICE_GLOBAL_MEM_SIZE\t\t= "
    << d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()           << std::endl
    << "CL_DEVICE_LOCAL_MEM_SIZE\t\t= "
    << d.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()            << std::endl
    << "CL_DEVICE_MAX_CLOCK_FREQUENCY\t\t= "
    << d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>()       << std::endl
    << "CL_DEVICE_MAX_COMPUTE_UNITS\t\t= "
    << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()         << std::endl
    << "CL_DEVICE_MAX_CONSTANT_ARGS\t\t= "
    << d.getInfo<CL_DEVICE_MAX_CONSTANT_ARGS>()         << std::endl
    << "CL_DEVICE_MAX_MEM_ALLOC_SIZE\t\t= "
    << d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()        << std::endl
    << "CL_DEVICE_MAX_PARAMETER_SIZE\t\t= "
    << d.getInfo<CL_DEVICE_MAX_PARAMETER_SIZE>()        << std::endl
    << "CL_DEVICE_MAX_WORK_GROUP_SIZE\t\t= "
    << d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()       << std::endl
    << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS\t= "
    << d.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>()  << std::endl
    ;

  o << "CL_DEVICE_EXECUTION_CAPABILITIES\t= [";
  unsigned int ecs = d.getInfo<CL_DEVICE_EXECUTION_CAPABILITIES>();
    if (ecs & CL_EXEC_KERNEL)        { o << "CL_EXEC_KERNEL"; }
    if (ecs & CL_EXEC_NATIVE_KERNEL) { o << ", CL_EXEC_NATIVE_KERNEL"; }
  o << "]" << std::endl;

  o << "CL_DEVICE_MAX_WORK_ITEM_SIZES\t\t= [";
  auto wiss = d.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
  for (auto wis = wiss.begin(); wis != wiss.end(); ) {
    o << *wis << (++wis != wiss.end() ? ", " : "");
  }
  o << "]" << std::endl;

  o << "CL_DEVICE_NAME\t\t\t\t= "
    << d.getInfo<CL_DEVICE_NAME>()                      << std::endl
    << "CL_DEVICE_VENDOR\t\t\t= "
    << d.getInfo<CL_DEVICE_VENDOR>()                    << std::endl
    << "CL_DEVICE_VERSION\t\t\t= "
    << d.getInfo<CL_DEVICE_VERSION>()                   << std::endl
    << "CL_DRIVER_VERSION\t\t\t= "
    << d.getInfo<CL_DRIVER_VERSION>()                   << std::endl
    << "CL_DEVICE_EXTENSIONS\t\t\t= "
    << d.getInfo<CL_DEVICE_EXTENSIONS>();

  return o;
}

class Cell
{
  public:
    typedef Stencils::VonNeumann<3, 1> Stencil;
    typedef Topologies::Cube<3>::Topology Topology;
    class API : public CellAPITraits::Fixed {};

    static inline unsigned nanoSteps() { return 1; }

    inline explicit Cell(const double& v=0) : temp(v) {}

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned& /* nanoStep */)
    {
      temp = (neighborhood[FixedCoord< 0,  0, -1>()].temp +
          neighborhood[FixedCoord< 0, -1,  0>()].temp +
          neighborhood[FixedCoord<-1,  0,  0>()].temp +
          neighborhood[FixedCoord< 1,  0,  0>()].temp +
          neighborhood[FixedCoord< 0,  1,  0>()].temp +
          neighborhood[FixedCoord< 0,  0,  1>()].temp) * (1.0 / 6.0);
      std::cerr << "neighborhood: " << neighborhood.toString() << std::endl;
    }

    double temp;
};

template<typename CELL>
class MyFutureOpenCLStepper
{
  public:
    const static int DIM = CELL::Topology::DIM;

    typedef typename CELL::Topology Topology;
    typedef DisplacedGrid<CELL, Topology>  GridType;
    const static int DIM = Topology::DIM;

    MyFutureOpenCLStepper(const CoordBox<DIM> box,
                          size_t platform_id, size_t device_id,
                          const std::string & kernel_name,
                          const std::string & kernel_file) :
      box(box),
      hostGrid(box)
  {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    std::cerr << "# of Platforms: " << platforms.size() << std::endl;
    for (auto & platform : platforms) { std::cerr << platform; }

    const cl::Platform & platform = platforms[platform_id];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    std::cerr << "# of Devices: " << devices.size() << std::endl;
    for (auto & device : devices) { std::cerr << device; }
    std::cerr << std::endl;

    const cl::Device & device = devices[device_id];

    context = cl::Context({ device });

    cmdq = cl::CommandQueue(context, device);


    size_t size = hostGrid.getDimensions().prod();
    double * in_address = new double[size];

    for (int i = 0; i < size; ++i) { in_address[i] = i; }

    cl::Buffer cl_points, cl_input, cl_output;

    try {
      cl_points = cl::Buffer(context,
                             CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                             points.size() * sizeof(cl_int3),
                             points.data());

      cmdq.enqueueWriteBuffer(cl_points, CL_TRUE, 0,
                              points.size(), points.data());

      cl_input = cl::Buffer(context,
                            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            size * sizeof(double),
                            in_address);

      cmdq.enqueueWriteBuffer(cl_input, CL_TRUE, 0, size, in_address);

      cl_output = cl::Buffer(context, CL_MEM_WRITE_ONLY, size);

    } catch (cl::Error & error) {
      std::cerr << "Error: " << error.what() << ": "
                << get_error_description(error.err())
                << " (" << error.err() << ")"
                << std::endl;
      exit(EXIT_FAILURE);
    }

    // std::string kernel_source_code("#pragma OPENCL EXTENSION cl_intel_printf:");
    std::string kernel_source_code;
    kernel_source_code.append("#pragma OPENCL EXTENSION all:");
    kernel_source_code.append("enable");
    kernel_source_code.append("\n");

    try {
      std::ifstream kernel_stream;
      kernel_stream.exceptions(std::ios::failbit | std::ios::badbit);
      kernel_stream.open(kernel_file.c_str());

      kernel_source_code.append(std::istreambuf_iterator<char>(kernel_stream),
                                std::istreambuf_iterator<char>());

      kernel_stream.close();
    } catch (std::exception & error) {
      std::cerr << "Error while trying to access \""
                << kernel_file << "\":" << std::endl
                << error.what() << std::endl;
      exit(EXIT_FAILURE);
    }

    cl::Program::Sources kernel_sources(1,
        std::make_pair(kernel_source_code.c_str(),
                       kernel_source_code.length() + 1));

    cl::Program program(context, kernel_sources);

    try {
      program.build(std::vector<cl::Device>(1, device));

      cl::Kernel kernel(program, kernel_name.c_str());

      cl_int3 cl_size = { hostGrid.getDimensions().x(),
                          hostGrid.getDimensions().y(),
                          hostGrid.getDimensions().z() };

      kernel.setArg(0, cl_size);
      kernel.setArg(1, cl_points);
      kernel.setArg(2, cl_input);
      kernel.setArg(3, cl_output);

      cmdq.enqueueNDRangeKernel(kernel,
                                cl::NullRange,
                                cl::NDRange(points.size()),
                                cl::NullRange
                               );

      cmdq.finish();

    } catch (cl::Error & error) {
      std::cerr << "Error: " << error.what() << ": "
                << get_error_description(error.err())
                << " (" << error.err() << ")"
                << std::endl
                << "Build Log:" << std::endl
                << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
    }

    // cmdq.enqueueNDRangeKernel(kernel, cl::NullRange
    // // global:
    // // describes the number of global work-items in will execute the
    // // kernel function. The total number of global work-items is computed as
    // // global_work_size[0] * ... * global_work_size[work_dim - 1].
    // // global size must be a multiple of the local size
    // // (page 23 in lecture2.pdf)
                              // , cl::NDRange ( ( bufSize + workgroupSize - 1 )
                                            // / workgroupSize * workgroupSize
                                            // )
    // // local:
    // // describes the number of work-items that make up a work-group (also
    // // referred to as the size of the work-group) that will execute the kernel
    // // specified by kernel.
                              // , cl::NDRange ( workgroupSize )
                              // );
  }

  void regionToVec(const Region<DIM>& region, SuperVector<int> *coordsX, SuperVector<int> *coordsY, SuperVector<int> *coordsZ)
    {
      // todo: iterate through region and add all coordinates to the corresponding vectors
    }

    template<typename GRID>
      void setGridRegion(const GRID& grid, const Region<DIM>& region)
      {
        // todo: copy all coords in region from grid (on host) to deviceGridOld
      }

    template<typename GRID>
      void getGridRegion(const GRID *grid, const Region<DIM>& region)
      {
        // todo: copy all coords in region from deviceGridOld (on host) to grid
      }

  private:
    CoordBox<DIM> box;
    GridType hostGrid;

    cl::Context context;
    cl::CommandQueue cmdq;
    cl::Buffer deviceGridOld;
    cl::Buffer deviceGridNew;
};

int main(int argc, char **argv)
{
  auto box = CoordBox<3>(Coord<3>(1,1,1), Coord<3>(3, 3, 3));
  if (argc == 5) {
    MyFutureOpenCLStepper<Cell> stepper(box,
                                        strtol(argv[1], NULL, 10),
                                        strtol(argv[2], NULL, 10),
                                        argv[3],
                                        argv[4]);
  }

  return 0;
}

std::string
get_error_description(cl_int error) {
    switch (error) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default: return "Unknown";
    }
}
