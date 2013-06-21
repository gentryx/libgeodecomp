/* vim:set expandtab tabstop=2 shiftwidth=2 softtabstop=2: */

#include <fstream>
#include <iostream>
#include <libgeodecomp/libgeodecomp.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/io/testinitializer.h>

#include "openclstepper.h"

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

using namespace LibGeoDecomp;

typedef struct {
  cl_int3   points_size;
  cl_int3 * points;
  cl_int  * indices;
} coords_ctx;

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

class DummyCell {
  public:
    static const int DIMENSIONS = 2;
    typedef Stencils::VonNeumann<2, 1> Stencil;
    class API : public CellAPITraits::Base {};
    typedef Topologies::Cube<2>::Topology Topology;
    static inline unsigned nanoSteps() { return 1; }
    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned& nanoStep) {}
};

template<typename CELL>
class MyFutureOpenCLStepper {
  public:
    const static int DIM = CELL::Topology::DIM;

    typedef typename CELL::Topology Topology;
    typedef DisplacedGrid<CELL, Topology>  GridType;
    const static int DIM = Topology::DIM;

    MyFutureOpenCLStepper(const CoordBox<DIM> box,
                          size_t num_neighbors, size_t num_updates,
                          size_t platform_id, size_t device_id,
                          const std::string & user_code_kernel_name,
                          const std::string & user_code_file) :
      box(box),
      hostGrid(box)
  {
    coords_ctx coords;

    size_t num_points = hostGrid.getDimensions().prod();
    coords.points_size = { hostGrid.getDimensions().x(),
                           hostGrid.getDimensions().y(),
                           0 };
                           // hostGrid.getDimensions().z() };

    std::vector<cl_int3> points;
    for (auto & p : box) { points.push_back({ p.x(), p.y(), 0 }); }

    size_t size = hostGrid.getDimensions().prod();
    double * input = new double[size];
    for (int i = 0; i < size; ++i) { input[i] = i; }

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

    cl::Buffer cl_coords, cl_input, cl_output, cl_points, cl_indices;

    try {
      cl_coords = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(coords_ctx));

      cmdq.enqueueWriteBuffer(cl_coords, CL_TRUE, 0,
                              sizeof(coords_ctx), &coords);

      cl_input = cl::Buffer(context,
                            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                            size * sizeof(double),
                            input);

      cl_output = cl::Buffer(context, CL_MEM_READ_WRITE, size * sizeof(double));

      cl_points = cl::Buffer(context,
                             CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                             points.size() * sizeof(cl_int3),
                             points.data());

      cl_indices = cl::Buffer(context, CL_MEM_READ_ONLY,
                              num_points * sizeof(cl_int));

    } catch (cl::Error & error) {
      std::cerr << "Error: " << error.what() << ": "
                << get_error_description(error.err())
                << " (" << error.err() << ")"
                << std::endl;
      exit(EXIT_FAILURE);
    }

    std::stringstream pre_code_ss;
    pre_code_ss << "#define NUM_DIMS " << DIM << "\n"
                << "#define NUM_POINTS " << num_points << "\n"
                ;

    std::string init_code_txt(pre_code_ss.str());
    std::string user_code_txt(pre_code_ss.str());

    try {
      std::ifstream kernel_stream;
      kernel_stream.exceptions(std::ios::failbit | std::ios::badbit);

      kernel_stream.open(libgeodecomp_file);
      std::string coords_ctx_txt;
      coords_ctx_txt.append(std::istreambuf_iterator<char>(kernel_stream),
                            std::istreambuf_iterator<char>());
      kernel_stream.close();

      kernel_stream.open(user_code_file);
      user_code_txt.append(std::istreambuf_iterator<char>(kernel_stream),
                                std::istreambuf_iterator<char>());
      kernel_stream.close();

      kernel_stream.open(init_code_file);
      init_code_txt.append(std::istreambuf_iterator<char>(kernel_stream),
                              std::istreambuf_iterator<char>());
      kernel_stream.close();

    } catch (std::exception & error) {
      std::cerr << "Error while trying to access \""
                << user_code_file << "\":" << std::endl
                << error.what() << std::endl;
      exit(EXIT_FAILURE);
    }

    cl::Program init_code_program(context,
        { std::make_pair(init_code_txt.c_str(), init_code_txt.length() + 1) });

    cl::Program user_code_program(context,
        { std::make_pair(user_code_txt.c_str(), user_code_txt.length() + 1) });

    try {
      init_code_program.build(std::vector<cl::Device>(1, device));
      user_code_program.build(std::vector<cl::Device>(1, device));

      cl::Kernel mem_hook_up_kernel(init_code_program, "mem_hook_up");
      cl::Kernel data_init_kernel(init_code_program, "data_init");
      cl::Kernel user_code_kernel(user_code_program,
                                  user_code_kernel_name.c_str());

      int arg_counter = 0;
      mem_hook_up_kernel.setArg(arg_counter++, cl_coords);
      mem_hook_up_kernel.setArg(arg_counter++, cl_points);
      mem_hook_up_kernel.setArg(arg_counter++, cl_indices);
      cmdq.enqueueTask(mem_hook_up_kernel);

      arg_counter = 0;
      data_init_kernel.setArg(arg_counter++, cl_coords);
      cmdq.enqueueNDRangeKernel(data_init_kernel,
                                cl::NullRange,
                                cl::NDRange(num_points),
                                cl::NullRange);

      arg_counter = 0;
      user_code_kernel.setArg(arg_counter++, cl_coords);

      for (int i = 0; i < num_updates; ++i) {
        // let's play ping - pong
        // (http://www.mathematik.uni-dortmund.de/~goeddeke/gpgpu/tutorial.html#feedback2)
        // 1 + 0 & 1 = 1; 2 - 0 & 1 = 2; 1 + 1 & 1 = 2; 2 - 1 & 1 = 1
        user_code_kernel.setArg(1 + (i & 1), cl_input);
        user_code_kernel.setArg(2 - (i & 1), cl_output);

        cmdq.enqueueNDRangeKernel(user_code_kernel,
                                  cl::NullRange,
                                  cl::NDRange(num_points),
                                  cl::NullRange);
      }


      void * output = cmdq.enqueueMapBuffer(cl_output, CL_TRUE, CL_MAP_READ, 0,
                                            size * sizeof(double));

      cmdq.finish();

      std::cerr << "[" << 0 << "] = " << ((double *)output)[0];
      for (int i = 1; i < size; ++i) {
        std::cerr << ", [" << i << "] = " << ((double *)output)[i];
      }
      std::cerr << std::endl;

    } catch (cl::Error & error) {
      std::cerr << "Error: " << error.what() << ": "
                << get_error_description(error.err())
                << " (" << error.err() << ")"
                << std::endl
                << "Build Log for user code:" << std::endl
                << user_code_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
                << "Build Log for init code:" << std::endl
                << init_code_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
      exit(EXIT_FAILURE);
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

    const std::string init_code_file = "./init_code.cl";
    const std::string libgeodecomp_file = "./libgeodecomp.cl";

    cl::Context context;
    cl::CommandQueue cmdq;
    cl::Buffer deviceGridOld;
    cl::Buffer deviceGridNew;
};

int main(int argc, char **argv)
{
  auto box = CoordBox<2>(Coord<2>(0,0), Coord<2>(3, 3));
  if (argc == 5) {
    // coordbox,
    // num_neighbors, num_updates,
    // platform_id, device_id,
    // user_code_kernel_name, user_code_file
    MyFutureOpenCLStepper<DummyCell> stepper(box, 1, 1,
                                             strtol(argv[1], NULL, 10),
                                             strtol(argv[2], NULL, 10),
                                             argv[3],
                                             argv[4]);

    boost::shared_ptr<HiParSimulator::PartitionManager<2>>
      pmp(new HiParSimulator::PartitionManager<2>());

    boost::shared_ptr<TestInitializer<TestCell<2>>>
      dcip(new TestInitializer<TestCell<2>>());

    HiParSimulator::OpenCLStepper<TestCell<2>> openclstepper(pmp, dcip);
    openclstepper.update(1);

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
