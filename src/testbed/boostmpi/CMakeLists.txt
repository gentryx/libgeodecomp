generate_sourcelists("./")

set(RELATIVE_PATH "")
include(auto.cmake)

if(FEATURE_BOOST_MPI)
  add_executable(boostmpi main.cpp)
  find_package(Boost REQUIRED date_time filesystem system mpi)
  target_link_libraries(boostmpi ${LOCAL_LIBGEODECOMP_LINK_LIB})
  target_link_libraries(boostmpi ${Boost_LIBRARIES})
endif()
