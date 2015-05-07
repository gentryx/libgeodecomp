lgd_generate_sourcelists("./")
set(RELATIVE_PATH "")
include(auto.cmake)

add_executable(libgeodecomp_examples_bouncingspheres ${SOURCES})
set_target_properties(libgeodecomp_examples_bouncingspheres PROPERTIES OUTPUT_NAME bouncingspheres)
target_link_libraries(libgeodecomp_examples_bouncingspheres ${LOCAL_LIBGEODECOMP_LINK_LIB})
