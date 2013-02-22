generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(cars ${SOURCES})
include_directories(/opt/visit/2.6.0/linux-x86_64/libsim/V2/include)
find_library(VISIT NAMES simV2 dl PATHS /opt/visit/2.6.0/linux-x86_64/libsim/V2/lib)
target_link_libraries(cars boost_system boost_thread-mt ${LOCAL_LIBGEODECOMP_LINK_LIB} ${VISIT})

