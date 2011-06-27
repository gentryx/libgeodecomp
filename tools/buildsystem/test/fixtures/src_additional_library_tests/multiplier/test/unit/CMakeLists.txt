include(auto.cmake)

include_directories (${DEMO_SOURCE_DIR}/../lib/cxxtest)
add_executable(multiplier_test_unit ${SOURCES})
set_target_properties(multiplier_test_unit PROPERTIES OUTPUT_NAME test)
target_link_libraries (multiplier_test_unit multiplier)
