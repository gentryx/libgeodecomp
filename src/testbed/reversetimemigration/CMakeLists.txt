lgd_generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_CUDA AND WITH_MPI)
  add_executable(libgeodecomp_testbed_reversetimemigration main.cpp)
  set_target_properties(libgeodecomp_testbed_reversetimemigration PROPERTIES OUTPUT_NAME rtm)
  target_link_libraries(libgeodecomp_testbed_reversetimemigration ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
