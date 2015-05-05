generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if((MACHINE_ARCH MATCHES "x86_64") AND WITH_CUDA AND WITH_MPI)
  add_executable(libgeodecomp_testbed_reversetimemigration main.cpp)
  set_target_properties(libgeodecomp_testbed_reversetimemigration PROPERTIES OUTPUT_NAME rtm)
  cuda_add_library(libgeodecomp_testbed_reversetimemigration_cudaStepper cudastepperlib.cu)

  target_link_libraries(libgeodecomp_testbed_reversetimemigration libgeodecomp_testbed_reversetimemigration_cudaStepper)
  target_link_libraries(libgeodecomp_testbed_reversetimemigration ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
