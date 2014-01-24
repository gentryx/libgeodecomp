generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if(WITH_BOOST_MPI AND WITH_MPI)
  add_executable(boostmpi main.cpp)
  target_link_libraries(boostmpi ${LOCAL_LIBGEODECOMP_LINK_LIB})
  target_link_libraries(boostmpi ${Boost_LIBRARIES})
endif()
