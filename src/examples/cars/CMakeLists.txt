generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(FEATURE_MPI AND FEATURE_VISIT)
  add_executable(cars ${SOURCES})
  target_link_libraries(cars boost_system boost_thread-mt ${LOCAL_LIBGEODECOMP_LINK_LIB})
endif()
