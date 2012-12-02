generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

if(FEATURE_MPI)
  add_executable(gameoflife_live ${SOURCES})
  include_directories(/home/bjoern/opt/visit/2.5.2/linux-x86_64/libsim/V2/include)
  find_library(VISIT NAMES simV2 dl PATHS /home/bjoern/opt/visit/2.5.2/linux-x86_64/libsim/V2/lib)
  target_link_libraries(gameoflife_live ${LOCAL_LIBGEODECOMP_LINK_LIB} ${VISIT})
endif(FEATURE_MPI)

