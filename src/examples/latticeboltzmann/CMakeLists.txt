set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(lbm ${SOURCES})
target_link_libraries(lbm ${LOCAL_LIBGEODECOMP_LINK_LIB})
