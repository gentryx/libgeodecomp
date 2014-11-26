generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_MPI AND WITH_INTRINSICS)
  add_executable(jacobi3dvectorized ${SOURCES})
  target_link_libraries(jacobi3dvectorized ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
