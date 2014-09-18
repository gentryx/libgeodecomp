generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_MPI AND WITH_VISIT AND WITH_THREADS AND WITH_BOOST_ASIO)
  add_executable(gameoflife3d ${SOURCES})
  target_link_libraries(gameoflife3d ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
