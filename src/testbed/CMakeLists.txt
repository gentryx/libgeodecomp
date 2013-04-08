generate_sourcelists("./")

add_subdirectory(boostmpi)
add_subdirectory(cactusinterfacing)
add_subdirectory(performancetests)
add_subdirectory(opencl)
add_subdirectory(reversetimemigration)
add_subdirectory(vectorization)
add_subdirectory(wavefront)

set(RELATIVE_PATH "")
include(auto.cmake)

if(FEATURE_INTRINSICS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -DMY_SIZE=7512")
  add_executable(testbed ${SOURCES})
  target_link_libraries(testbed ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()

