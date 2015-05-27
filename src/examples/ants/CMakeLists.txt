lgd_generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(libgeodecomp_examples_ants ${SOURCES})
set_target_properties(libgeodecomp_examples_ants PROPERTIES OUTPUT_NAME ants)
target_link_libraries(libgeodecomp_examples_ants ${LOCAL_LIBGEODECOMP_LINK_LIB})
