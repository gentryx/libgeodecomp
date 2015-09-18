lgd_generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(libgeodecomp_examples_jacobi2d ${SOURCES})
set_target_properties(libgeodecomp_examples_jacobi2d PROPERTIES OUTPUT_NAME jacobi2d)
target_link_libraries(libgeodecomp_examples_jacobi2d ${LOCAL_LIBGEODECOMP_LINK_LIB})
