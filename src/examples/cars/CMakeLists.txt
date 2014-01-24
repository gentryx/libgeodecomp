generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_MPI AND WITH_VISIT)
  add_executable(cars ${SOURCES})
  target_link_libraries(cars boost_system boost_thread-mt ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
