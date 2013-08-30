#ifndef _OPENCLWRAPPER_H
#define _OPENCLWRAPPER_H

#include <fstream>
#include <iostream>
#include <tuple>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#define DEBUG 0
#define DEBUG_FUNCALL() \
  if (DEBUG) { \
    std::cerr << __PRETTY_FUNCTION__ << std::endl; \
  }

typedef struct {
  cl_int4   points_size;
  cl_int4 * points;
  cl_int  * indices;
} coords_ctx;

std::string get_error_description(cl_int error);

std::ostream & operator<<(std::ostream &, cl::Platform);

std::ostream & operator<<(std::ostream &, cl::Device);

template<typename DATA_TYPE>
class OpenCLWrapper {
  public:
    typedef std::tuple<size_t, size_t, size_t> point_t;
    typedef void * data_t;

    OpenCLWrapper(unsigned int platform_id, unsigned int device_id,
                  const std::string & user_code_file,
                  const std::string & user_code_function,
                  const size_t x_size, const size_t y_size, const size_t z_size,
                  bool verbose = false);

    void run(size_t updates = 1);
    void flush(void);
    void finish(void);

    template<typename Iterator>
      void loadPoints(Iterator begin, Iterator end);

    template<typename Iterator>
      void loadHostData(Iterator begin, Iterator end);

    void * const readDeviceData(void);

  private:
    const std::string init_code_cl_file = "./init_code.cl";
    const std::string mem_hook_function = "mem_hook";
    const std::string data_init_function = "data_init";

    unsigned int platform_id, device_id;
    std::string user_code_file, user_code_function;
    std::string init_code_txt, user_code_txt;

    coords_ctx coords;
    const size_t x_size, y_size, z_size;
    const size_t num_points;

    bool verbose = false;
    size_t update_counter = 0;

    cl::Device device;
    cl::Context context;
    cl::CommandQueue cmdqueue;
    cl::Program init_code_program, user_code_program;
    cl::Kernel mem_hook_kernel, data_init_kernel, user_code_kernel;
    cl::Buffer cl_input, cl_output, cl_coords, cl_points, cl_indices;

    void createBuffers(void);
    void initCommandQueue(void);
    void readKernels(void);
    void initKernels(void);
    void printCLError(cl::Error & error, const std::string & where);
};

template<typename DATA_TYPE>
OpenCLWrapper<DATA_TYPE>::
OpenCLWrapper(unsigned int platform_id, unsigned int device_id,
              const std::string & user_code_file,
              const std::string & user_code_function,
              const size_t x_size, const size_t y_size, const size_t z_size,
              bool verbose)
  : platform_id(platform_id), device_id(device_id)
  , user_code_file(user_code_file)
  , user_code_function(user_code_function)
  , x_size(x_size), y_size(y_size), z_size(z_size)
  , num_points(x_size * y_size * z_size)
  , verbose(verbose)
{
    coords.points_size = { (cl_int)x_size, (cl_int)y_size, (cl_int)z_size };

    initCommandQueue();

    try {
      createBuffers();
    } catch (cl::Error & error) {
      printCLError(error, __PRETTY_FUNCTION__);
      exit(EXIT_FAILURE);
    }

    try {
      readKernels();
    } catch (std::exception & error) {
      std::cerr << "Error while trying to access \""
                << user_code_file << "\":" << std::endl
                << error.what() << std::endl;
      exit(EXIT_FAILURE);
    }

    try {
      initKernels();
    } catch (cl::Error & error) {
      std::cerr << "Error: " << error.what() << ": "
                << get_error_description(error.err())
                << " (" << error.err() << ")" << std::endl
                << "Build Log for user code:" << std::endl
                << user_code_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
                << "Build Log for init code:" << std::endl
                << init_code_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
      exit(EXIT_FAILURE);
    }

    try {
      cmdqueue.enqueueWriteBuffer(cl_coords, CL_TRUE, 0,
                                  sizeof(coords_ctx), &coords);
      cmdqueue.enqueueTask(mem_hook_kernel);
    } catch (cl::Error & error) {
      printCLError(error, __PRETTY_FUNCTION__);
      exit(EXIT_FAILURE);
    }
}

template<typename DATA_TYPE>
void
OpenCLWrapper<DATA_TYPE>::initCommandQueue(void)
{
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  if (verbose) {
    std::cerr << "# of Platforms: " << platforms.size() << std::endl;
    for (auto & platform : platforms) { std::cerr << platform; }
    std::cerr << std::endl;
  }

  std::vector<cl::Device> devices;
  platforms[platform_id].getDevices(CL_DEVICE_TYPE_ALL, &devices);

  if (verbose) {
    std::cerr << "# of Devices: " << devices.size() << std::endl;
    for (auto & device : devices) { std::cerr << device; }
    std::cerr << std::endl;
  }

  device = devices[device_id];
  context = cl::Context(devices);
  cmdqueue = cl::CommandQueue(context, device);
}

template<typename DATA_TYPE>
void
OpenCLWrapper<DATA_TYPE>::createBuffers(void)
{
  cl_coords = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(coords_ctx));
  cl_input = cl::Buffer(context, CL_MEM_READ_WRITE, num_points * sizeof(DATA_TYPE));
  cl_output = cl::Buffer(context, CL_MEM_READ_WRITE, num_points * sizeof(DATA_TYPE));
  cl_points = cl::Buffer(context, CL_MEM_READ_ONLY, num_points * sizeof(cl_int4));
  cl_indices = cl::Buffer(context, CL_MEM_READ_ONLY, num_points * sizeof(cl_int));
}

template<typename DATA_TYPE>
void
OpenCLWrapper<DATA_TYPE>::readKernels(void)
{
  std::ifstream fstream;
  fstream.exceptions(std::ios::failbit | std::ios::badbit);

  fstream.open(init_code_cl_file);
  init_code_txt.append(std::istreambuf_iterator<char>(fstream),
                       std::istreambuf_iterator<char>());
  fstream.close();

  fstream.open(user_code_file);
  user_code_txt.append(std::istreambuf_iterator<char>(fstream),
                       std::istreambuf_iterator<char>());
  fstream.close();
}

template<typename DATA_TYPE>
void
OpenCLWrapper<DATA_TYPE>::initKernels(void)
{
  init_code_program = cl::Program(context,
      { std::make_pair(init_code_txt.c_str(), init_code_txt.length() + 1) });

  user_code_program = cl::Program(context,
      { std::make_pair(user_code_txt.c_str(), user_code_txt.length() + 1) });

  init_code_program.build({ device });
  user_code_program.build({ device });

  mem_hook_kernel = cl::Kernel(init_code_program, mem_hook_function.c_str());
  data_init_kernel = cl::Kernel(init_code_program, data_init_function.c_str());
  user_code_kernel = cl::Kernel(user_code_program, user_code_function.c_str());

  size_t arg_counter = 0;
  mem_hook_kernel.setArg(arg_counter++, cl_coords);
  mem_hook_kernel.setArg(arg_counter++, cl_points);
  mem_hook_kernel.setArg(arg_counter++, cl_indices);

  arg_counter = 0;
  data_init_kernel.setArg(arg_counter++, cl_coords);

  arg_counter = 0;
  user_code_kernel.setArg(arg_counter++, cl_coords);
}

template<typename DATA_TYPE>
template<typename Iterator>
void
OpenCLWrapper<DATA_TYPE>::loadPoints(Iterator begin, Iterator end)
{
  int i = 0;
  std::vector<cl_int4> pvec;
  for (Iterator p = begin; p != end; ++p) {
    ++i;
    pvec.push_back({ static_cast<cl_int>(std::get<0>(*p))
                   , static_cast<cl_int>(std::get<1>(*p))
                   , static_cast<cl_int>(std::get<2>(*p))
                   // cl_int4 is really a cl_int4
                   // so, in order to not mess up with struct packing, etc.
                   // put a dummy value here
                   , 0
                   });
  }

  if (i != num_points) {
    throw std::length_error("points.size() != num_points");
  }

  try {
    cmdqueue.enqueueWriteBuffer(cl_points, CL_TRUE, 0,
                                num_points * sizeof(cl_int4), pvec.data());
    cmdqueue.enqueueNDRangeKernel(data_init_kernel, cl::NullRange,
                                  cl::NDRange(num_points),
                                  cl::NullRange);
  } catch (cl::Error & error) {
    printCLError(error, __PRETTY_FUNCTION__);
    exit(EXIT_FAILURE);
  }
}

template<typename DATA_TYPE>
template<typename Iterator>
void
OpenCLWrapper<DATA_TYPE>::loadHostData(Iterator begin, Iterator end)
{

  int i = 0;
  try {
    for (Iterator it = begin; it != end; ++it) {
      cmdqueue.enqueueWriteBuffer(cl_input, CL_TRUE,
                                  i * sizeof(DATA_TYPE),
                                  sizeof(DATA_TYPE), *it);
      ++i;
    }
  } catch (cl::Error & error) {
    printCLError(error, __PRETTY_FUNCTION__);
    exit(EXIT_FAILURE);
  }

  if (i != num_points) {
    throw std::length_error("points.size() != num_points");
  }
}

template<typename DATA_TYPE>
void * const
OpenCLWrapper<DATA_TYPE>::readDeviceData(void)
{
  return cmdqueue.enqueueMapBuffer(cl_output, CL_TRUE, CL_MAP_READ, 0,
                                   num_points * sizeof(DATA_TYPE));
}

template<typename DATA_TYPE>
void
OpenCLWrapper<DATA_TYPE>::run(size_t updates)
{
  // let's play ping - pong
  // (http://www.mathematik.uni-dortmund.de/~goeddeke/gpgpu/tutorial.html#feedback2)
  // 1 + 0 & 1 = 1; 2 - 0 & 1 = 2; 1 + 1 & 1 = 2; 2 - 1 & 1 = 1

  for (size_t i = 0; i < updates; ++i) {
    user_code_kernel.setArg(1 + (update_counter & 1), cl_input);
    user_code_kernel.setArg(2 - (update_counter & 1), cl_output);

    cmdqueue.enqueueNDRangeKernel(user_code_kernel, cl::NullRange,
                                  cl::NDRange(num_points),
                                  cl::NullRange);
    ++update_counter;
  }
}

template<typename DATA_TYPE>
void
OpenCLWrapper<DATA_TYPE>::flush(void)
{
  cmdqueue.flush();
}

template<typename DATA_TYPE>
void
OpenCLWrapper<DATA_TYPE>::finish(void)
{
  cmdqueue.finish();
}

template<typename DATA_TYPE>
void
OpenCLWrapper<DATA_TYPE>::printCLError(cl::Error & error, const std::string & where)
{
  std::cerr << where << ": " << error.what()
            << std::endl
            << "Error " << error.err() << ": "
            << get_error_description(error.err())
            << std::endl;
  exit(EXIT_FAILURE);
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

std::ostream & operator<<(std::ostream & o, cl::Platform p)
{
  o << "CL_PLATFORM_VERSION\t= "     << p.getInfo<CL_PLATFORM_VERSION>()
    << std::endl
    << "CL_PLATFORM_NAME\t= "        << p.getInfo<CL_PLATFORM_NAME>()
    << std::endl
    << "CL_PLATFORM_VENDOR\t= "      << p.getInfo<CL_PLATFORM_VENDOR>()
    << std::endl
    << "CL_PLATFORM_EXTENSIONS\t= "  << p.getInfo<CL_PLATFORM_EXTENSIONS>();
  return o;
}

std::ostream & operator<<(std::ostream & o, cl::Device d)
{
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

#endif
