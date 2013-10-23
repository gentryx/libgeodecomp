generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(FEATURE_MPI AND FEATURE_VISIT AND FEATURE_THREADS)
  add_executable(gameoflife3d ${SOURCES})
  target_link_libraries(gameoflife3d ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
