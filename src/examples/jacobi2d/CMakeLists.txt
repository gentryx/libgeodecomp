generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(jacobi2d ${SOURCES})
target_link_libraries(jacobi2d ${LOCAL_LIBGEODECOMP_LINK_LIB})
