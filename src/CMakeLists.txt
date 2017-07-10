# LIBDIRS is used to find source files and headers
set(LIBDIRS libgeodecomp/communication libgeodecomp/io libgeodecomp/io/remotesteerer libgeodecomp/geometry libgeodecomp/geometry/partitions libgeodecomp/loadbalancer libgeodecomp/misc libgeodecomp/parallelization libgeodecomp/parallelization/nesting libgeodecomp/storage)
# AUXDIRS lists auxiliary directories to be included in the main
# build. They may for instance include additionally libraries to be
# linked into the main lib.
set(AUXDIRS)
set(SUBDIRS examples libgeodecomp/communication libgeodecomp/io libgeodecomp/geometry libgeodecomp/loadbalancer libgeodecomp/misc libgeodecomp/parallelization libgeodecomp/storage testbed)

set(CXX_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib/cxxtest")
set(LGD_AGGREGATED_C_FLAGS   "${LGD_ADDITIONAL_C_COMPILE_FLAGS}"  )
set(LGD_AGGREGATED_CXX_FLAGS "${LGD_ADDITIONAL_CXX_COMPILE_FLAGS}")

list(APPEND BUILD_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR})
lgd_generate_sourcelists("./")

if(NOT WITH_CPP14)
  check_cxx_compiler_flag("-std=c++0x" SUPPORTS_CPP0X)
  if(NOT SUPPORTS_CPP0X)
    message(WARNING "LibGeoDecomp requires support for some C++11 features (e.g. shared pointers), but your compiler doesn't seem to support this.")
  endif()

  set(LGD_AGGREGATED_CXX_FLAGS "${LGD_AGGREGATED_CXX_FLAGS} -std=c++0x")
endif()

if(WITH_BOOST_SHARED_PTR OR WITH_BOOST_ASIO OR WITH_BOOST_SERIALIZATION OR WITH_BOOST_MPI OR WITH_BOOST_MOVE)
  list(APPEND INSTALL_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
endif()

if(WITH_BOOST_SERIALIZATION)
  set(ALL_BOOST_LIBS "${ALL_BOOST_LIBS};${Boost_SERIALIZATION_LIBRARIES}")
endif()

if(WITH_BOOST_MPI)
  set(ALL_BOOST_LIBS "${ALL_BOOST_LIBS};${Boost_MPI_LIBRARIES}")
endif()

if(WITH_CUDA)
  if (${CUDA_VERSION} VERSION_LESS 8.0)
    set(CUDA_NVCC_FLAGS "--gpu-architecture=compute_20")
  else()
    set(CUDA_NVCC_FLAGS "--gpu-architecture=compute_35")
  endif()

  list(APPEND INSTALL_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS})
endif()

if(WITH_FORTRAN)
  enable_language(Fortran)
endif()

if(WITH_HPX)
  list(APPEND INSTALL_INCLUDE_DIRECTORIES ${HPX_INCLUDE_DIRS})
  add_definitions(${HPX_CXX_COMPILE_FLAGS})
  set(HPX_RPATH "${CMAKE_INSTALL_PREFIX}/${LIB_DIR}/hpx:${CMAKE_BINARY_DIR}/${LIB_DIR}/hpx:${CMAKE_BINARY_DIR}/src")
  set(HPXRUN "${HPX_PREFIX}/bin/hpxrun.py")
endif()

if(WITH_MPI)
  add_definitions(${MPI_COMPILE_FLAGS})
  list(APPEND INSTALL_INCLUDE_DIRECTORIES ${MPI_INCLUDE_PATH})
endif()

if(WITH_QT5)
  list(APPEND INSTALL_INCLUDE_DIRECTORIES ${Qt5Core_INCLUDE_DIRS} ${Qt5Gui_INCLUDE_DIRS} ${Qt5OpenGL_INCLUDE_DIRS})
  set(LGD_AGGREGATED_CXX_FLAGS "${LGD_AGGREGATED_CXX_FLAGS} ${Qt5Core_EXECUTABLE_COMPILE_FLAGS}")
  # Qt5 needs -fPIC, but on some nodes Qt5Core_EXECUTABLE_COMPILE_FLAGS lacks that flag:
  set(LGD_AGGREGATED_CXX_FLAGS "${LGD_AGGREGATED_CXX_FLAGS} -fPIC")
endif()

if(WITH_SCOTCH)
  list(APPEND INSTALL_INCLUDE_DIRECTORIES ${SCOTCH_INCLUDE_DIRS})
endif()

if(WITH_SILO)
  list(APPEND INSTALL_INCLUDE_DIRECTORIES ${Silo_INCLUDE_DIR})
endif()

if(WITH_THREADS)
  set(LGD_AGGREGATED_CXX_FLAGS "${LGD_AGGREGATED_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(WITH_THREADS)
  if (NOT WITH_LIBPTHREAD)
    set(LGD_AGGREGATED_CXX_FLAGS "${LGD_AGGREGATED_CXX_FLAGS} -pthread")
  endif()
endif()

set(LGD_AGGREGATED_CXX_FLAGS "${LGD_AGGREGATED_CXX_FLAGS} ${libflatarray_FLAGS}")

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${LGD_AGGREGATED_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LGD_AGGREGATED_CXX_FLAGS}")

#============= 1. SANITY CHECKS ======================================
# import settings detected by packages to decouple from their
# interfaces, perform sanity checks to match selected options and
# detected environment...
if(WITH_BOOST_MPI AND NOT Boost_MPI_FOUND)
  message(FATAL_ERROR "WITH_BOOST_MPI selected, but Boost.MPI library not found")
endif()

if(WITH_BOOST_MOVE AND NOT Boost_MOVE_FOUND)
  message(FATAL_ERROR "WITH_BOOST_MOVE selected, but Boost.Move library not found")
endif()

if(WITH_BOOST_SERIALIZATION AND NOT Boost_SERIALIZATION_FOUND)
  message(FATAL_ERROR "WITH_BOOST_SERIALIZATION selected, but Boost.Serialization library not found")
endif()

if(WITH_CUDA AND NOT CUDA_FOUND)
  message(FATAL_ERROR "WITH_CUDA selected, but could not find the NVIDIA CUDA toolkit.")
endif()

# Force the selection of the NVCC host compiler to circumvent the
# studid FindCUDA CMake module from resolving symlinks (*wink* ccache
# *wink*).
if(CUDA_FOUND)
  get_filename_component(exe_name "${CUDA_HOST_COMPILER}" NAME)
  if(exe_name STREQUAL "ccache")
    set(CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "Override host side compiler for NVCC" FORCE)
  endif()
endif()

if(WITH_HPX AND NOT HPX_FOUND)
  message(FATAL_ERROR "WITH_HPX selected but could not find HPX. Specify HPX_DIR to point to your HPX CMake scripts (e.g. cmake -DHPX_DIR=/home/alice/local_install/lib/cmake/hpx).")
endif()

if(WITH_HPX AND NOT WITH_CPP14)
  message(FATAL_ERROR "WITH_HPX selected but no C++14 support activated. Try -DWITH_CPP14=true")
endif()

if(WITH_MPI)
  if(NOT MPI_FOUND)
    message(FATAL_ERROR "WITH_MPI selected, but could find no MPI implementation.")
  endif()

  if(NOT MPIEXEC)
    message(FATAL_ERROR "WITH_MPI selected, but mpiexec not found.")
  endif()
endif()

if(WITH_OPENCL AND NOT OPENCL_FOUND)
  message(FATAL_ERROR "WITH_OPENCL selected, but could not find any OpenCL library.")
endif()

if(WITH_OPENCV AND NOT OpenCV_FOUND)
  message(FATAL_ERROR "WITH_OPENCV selected, but could not find OpenCV.")
endif()

if(WITH_QT5 AND NOT Qt5_FOUND)
  message(FATAL_ERROR "WITH_QT5 selected, but could not find Qt5")
endif()

if(WITH_SCOTCH AND NOT SCOTCH_FOUND)
  message(FATAL_ERROR "WITH_SCOTCH selected but no installation of Scotch could be found.")
endif()

if(WITH_SILO AND NOT Silo_FOUND)
  message(FATAL_ERROR "WITH_SILO selected, but FindSilo couldn't detect any installation of it.")
endif()

if(WITH_THREADS AND NOT OPENMP_FOUND)
message(FATAL_ERROR "WITH_THREADS selected, but no OpenMP support detected.")
endif()

if(WITH_VISIT)
  if (NOT VISIT_FOUND)
    message(FATAL_ERROR "WITH_VISIT selected, but no installation of VisIt could be found")
  endif()

  if (NOT PYTHONLIBS_FOUND)
    message(FATAL_ERROR "WITH_VISIT selected, but no Python libraries/headers could not be detected")
  endif()

  list(APPEND INSTALL_INCLUDE_DIRECTORIES ${VisIt_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
endif()

if(WIN32)
  set(LOCAL_LIBGEODECOMP_LINK_LIB local_libgeodecomp_link_lib)
  add_library(local_libgeodecomp_lib ${LIB_LINKAGE_TYPE} IMPORTED)
  set_property(TARGET local_libgeodecomp_lib PROPERTY IMPORTED_LOCATION ${LIBGEODECOMP_BINARY_DIR}/src/${CMAKE_BUILD_TYPE}/geodecomp.dll)
else()
  set(LOCAL_LIBGEODECOMP_LINK_LIB geodecomp)
endif()

include_directories(SYSTEM ${INSTALL_INCLUDE_DIRECTORIES})
include_directories(${BUILD_INCLUDE_DIRECTORIES})

#============= 2. TOP-LEVEL TARGETS ==================================
if(NOT TARGET check)
  add_custom_target(check echo "Tests passed.")
endif()
if(NOT TARGET tests)
  add_custom_target(tests echo "All tests have been built.")
endif()
add_custom_target(code_generation ALL echo "Code generation done.")
add_custom_target(doc cd "${CMAKE_CURRENT_SOURCE_DIR}" && ${DOXYGEN_EXECUTABLE} doxygen.conf)

#============= 3. CPACK INSTALLER STUFF ==============================
set(CPACK_PACKAGE_NAME ${PACKAGE_NAME})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PACKAGE_NAME})
set(CPACK_PACKAGE_VERSION ${PACKAGE_VERSION})
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${PACKAGE_NAME})

# will be shown e.g. in windows' control center package info
set(CPACK_PACKAGE_VENDOR ${PACKAGE_VENDOR})
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../README")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE")
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${PACKAGE_NAME})

if(WIN32 AND NOT UNIX)
  # unused in current nsis versions, fix pending (http://www.cmake.org/Bug/print_bug_page.php?bug_id=8682)
  set(CPACK_NSIS_DISPLAY_NAME ${PACKAGE_NAME})

  # name to show in software tab of control center
  set(CPACK_NSIS_DISPLAY_NAME ${PACKAGE_NAME})
  # will be shown e.g. in windows' control center package info
  set(CPACK_NSIS_HELP_LINK ${PACKAGE_HOMEPAGE})
  # will be shown e.g. in windows' control center package info
  set(CPACK_NSIS_URL_INFO_ABOUT ${PACKAGE_HOMEPAGE})
  # will be shown e.g. in windows' control center package info
  set(CPACK_NSIS_CONTACT ${PACKAGE_EMAIL})
  set(CPACK_NSIS_MODIFY_PATH ON)

  # There is a bug in NSI that does not handle full unix paths properly. Make
  # sure there is at least one set of four (4) backlasshes.
  set(CPACK_PACKAGE_ICON             "${CMAKE_SOURCE_DIR}/..\\libgeodecomp_icon.png")
  set(CPACK_NSIS_INSTALLED_ICON_NAME "${CMAKE_SOURCE_DIR}/..\\libgeodecomp_icon.png")
  # further options:
  #else(WIN32 AND NOT UNIX)
  #  set(CPACK_STRIP_FILES "bin/MyExecutable")
  #  set(CPACK_SOURCE_STRIP_FILES "")
  #  set(CPACK_PACKAGE_EXECUTABLES "helloworld" "My Hello World")
endif(WIN32 AND NOT UNIX)

include(CPack)

#============= 4. RECURSE SUBDIRS ====================================
set(HEADERS "")
set(SOURCES "")

foreach(dir ${SUBDIRS})
  add_subdirectory(${dir})
endforeach(dir)

foreach(dir ${AUXDIRS})
  set(HEADERS "")
  set(SOURCES "")
  include(${dir}/aux.cmake)
endforeach(dir)

#============= 5. ADD THE MAIN LIBRARY ===============================

# link all sources
set(SOURCES "")
foreach(dir ${LIBDIRS})
  set(RELATIVE_PATH ${dir}/)
  include(${dir}/auto.cmake)
endforeach(dir)

set(ALL_SOURCES ${SOURCES})
set(ALL_HEADERS ${HEADERS})

add_library(geodecomp ${LIB_LINKAGE_TYPE} ${SOURCES})
set_property(TARGET geodecomp PROPERTY VERSION ${PACKAGE_VERSION})

if(CMAKE_VERSION VERSION_GREATER 2.8.12)
  set_property(TARGET geodecomp PROPERTY MACOSX_RPATH On)
endif()

# CMake <2.8.11 doesn't have target_include_directories()
if(CMAKE_VERSION VERSION_GREATER 2.8.11)
  target_include_directories(geodecomp INTERFACE ${INSTALL_INCLUDE_DIRECTORIES})
  target_include_directories(geodecomp PRIVATE ${INSTALL_INCLUDE_DIRECTORIES} ${BUILD_INCLUDE_DIRECTORIES})
endif()

# link 3rd party libraries to main lib and add them to pkg-config file down below
set(LIBGEODECOMP_LINK_LIBRARIES ${ALL_BOOST_LIBS} ${libflatarray_LIBS})

if(WITH_HPX)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DHPX_DEBUG)
  endif()
  set(LIBGEODECOMP_LINK_LIBRARIES
    ${LIBGEODECOMP_LINK_LIBRARIES}
    hpx
    hpx_init
    ${HPX_LIBRARIES})
endif()

if(WITH_CUDA)
  set(LIBGEODECOMP_LINK_LIBRARIES
    ${LIBGEODECOMP_LINK_LIBRARIES}
    ${CUDA_LIBRARIES})
endif()

if(WITH_SILO)
  set(LIBGEODECOMP_LINK_LIBRARIES
    ${LIBGEODECOMP_LINK_LIBRARIES}
    ${Silo_LIBRARY})
endif()

if(WITH_SCOTCH)
  set(LIBGEODECOMP_LINK_LIBRARIES
    ${LIBGEODECOMP_LINK_LIBRARIES}
    ${SCOTCH_LIBRARIES})
endif()

if(WITH_QT5)
  get_target_property(Qt5Core_location    Qt5::Core    LOCATION)
  get_target_property(Qt5Gui_location     Qt5::Gui     LOCATION)
  get_target_property(Qt5OpenGL_location  Qt5::OpenGL  LOCATION)
  get_target_property(Qt5Widgets_location Qt5::Widgets LOCATION)

  set(LIBGEODECOMP_LINK_LIBRARIES
    ${LIBGEODECOMP_LINK_LIBRARIES}
    ${Qt5Core_location}
    ${Qt5Gui_location}
    ${Qt5OpenGL_location}
    ${Qt5Widgets_location})
endif()

if(WITH_MPI)
  set(LIBGEODECOMP_LINK_LIBRARIES ${LIBGEODECOMP_LINK_LIBRARIES} ${MPI_LIBRARIES})
  set_property(TARGET geodecomp PROPERTY LINK_FLAGS ${MPI_LINK_FLAGS})
endif()

if(WITH_OPENCL)
  set(LIBGEODECOMP_LINK_LIBRARIES ${LIBGEODECOMP_LINK_LIBRARIES} OpenCL)
endif()

if(WITH_VISIT)
  set(LIBGEODECOMP_LINK_LIBRARIES ${LIBGEODECOMP_LINK_LIBRARIES} ${VisIt_LIBRARIES} ${PYTHON_LIBRARIES} dl)
endif()

target_link_libraries(geodecomp ${LIBGEODECOMP_LINK_LIBRARIES})

#============= 6. INSTALLER CONFIG ===================================
install(
  TARGETS geodecomp
  EXPORT ${PACKAGE_NAME}-targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(FILES "${CMAKE_BINARY_DIR}/${PACKAGE_NAME}/config.h" DESTINATION include/${PACKAGE_NAME})
install(FILES libgeodecomp.h DESTINATION include)

# install all headers
foreach(dir ${LIBDIRS})
  set(HEADERS "")
  set(RELATIVE_PATH ${dir}/)
  include(${dir}/auto.cmake)
  install(FILES ${HEADERS} DESTINATION include/${dir})
endforeach(dir)

# link against the libs we need
set(INSTALLER_LINK_FLAGS "-L${CMAKE_INSTALL_PREFIX}/${LIB_DIR} -Wl,-rpath,${CMAKE_INSTALL_PREFIX}/${LIB_DIR} -lgeodecomp ")
set(LIVE_LINK_FLAGS      "-L${CMAKE_CURRENT_BINARY_DIR}        -Wl,-rpath,${CMAKE_CURRENT_BINARY_DIR}        -lgeodecomp ")
foreach(lib ${LIBGEODECOMP_LINK_LIBRARIES})
  get_filename_component(path "${lib}" PATH)
  get_filename_component(name "${lib}" NAME)
  string(REGEX REPLACE "\\.so.*$" "" name "${name}")
  string(REGEX REPLACE "\\.a.*$" "" name "${name}")
  string(REGEX REPLACE "\\.dll.*$" "" name "${name}")

  if(name MATCHES ^lib)
    # this dirty trick is required to prevent REGEX REPLACE matching
    # every occurrence of "lib" ("^lib" will sadly match all, too, if
    # we're going to remove them)
    set(name "_dummy_${name}")
    string(REGEX REPLACE "_dummy_lib" "" name "${name}")
  endif()

  if (path)
    set(postfix "-L${path} -Wl,-rpath,${path}")
    set(INSTALLER_LINK_FLAGS "${INSTALLER_LINK_FLAGS} ${postfix}")
    set(LIVE_LINK_FLAGS      "${LIVE_LINK_FLAGS}      ${postfix}")
  endif()

  set(postfix "-l${name}")
  set(INSTALLER_LINK_FLAGS "${INSTALLER_LINK_FLAGS} ${postfix}")
  set(LIVE_LINK_FLAGS      "${LIVE_LINK_FLAGS}      ${postfix}")
endforeach()

set(PKG_CONFIG_INCLUDE_FLAGS "")
foreach(dir ${INSTALL_INCLUDE_DIRECTORIES})
  set(PKG_CONFIG_INCLUDE_FLAGS "${PKG_CONFIG_INCLUDE_FLAGS} -I${dir}")
endforeach()

set(LIVE_INCLUDE_FLAGS "")
foreach(dir ${BUILD_INCLUDE_DIRECTORIES})
  set(LIVE_INCLUDE_FLAGS "${LIVE_INCLUDE_FLAGS} -I${dir}")
endforeach()

file(WRITE "${CMAKE_BINARY_DIR}/${PACKAGE_NAME}.pc" "
Name: LibGeoDecomp
Description: Live LibGeoDecomp build tree
Version: ${PACKAGE_VERSION}
Libs: ${LIVE_LINK_FLAGS}
Cflags: ${LIVE_INCLUDE_FLAGS} ${PKG_CONFIG_INCLUDE_FLAGS} ${LGD_AGGREGATED_CXX_FLAGS}
")

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}.pc" "
Name: LibGeoDecomp
Description: High performance stencil code library
Version: ${PACKAGE_VERSION}
Libs: ${INSTALLER_LINK_FLAGS}
Cflags: -I${CMAKE_INSTALL_PREFIX}/include ${PKG_CONFIG_INCLUDE_FLAGS}
")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}.pc" DESTINATION "${LIB_DIR}/pkgconfig")

set(ConfigPackageLocation lib/cmake/${PACKAGE_NAME})

if(CMAKE_VERSION VERSION_GREATER 2.8.10)
  include(CMakePackageConfigHelpers)

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/../${PACKAGE_NAME}ConfigVersion.cmake"
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion)

  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/../${PACKAGE_NAME}ConfigVersion.cmake"
    DESTINATION "${ConfigPackageLocation}")
endif()

export(
  TARGETS geodecomp
  FILE "${CMAKE_CURRENT_BINARY_DIR}/../${PACKAGE_NAME}-targets.cmake")

# this config includes the current build tree and is meant for
# downstream projects which wish to include/link against a live build tree:
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/../${PACKAGE_NAME}Config.cmake"

"
get_filename_component(LIBGEODECOMP_CMAKE_DIR \${CMAKE_CURRENT_LIST_FILE} PATH)
include(\"\${LIBGEODECOMP_CMAKE_DIR}/${PACKAGE_NAME}-targets.cmake\")

set(libgeodecomp_INCLUDE_DIRS
  ${BUILD_INCLUDE_DIRECTORIES}
  ${INSTALL_INCLUDE_DIRECTORIES}
)

set(libgeodecomp_FLAGS \"${LGD_AGGREGATED_CXX_FLAGS}\")
")

# conversely, the to-be-installed config should not contain any
# references to the build tree:
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake"

"
get_filename_component(LIBGEODECOMP_CMAKE_DIR \${CMAKE_CURRENT_LIST_FILE} PATH)
include(\"\${LIBGEODECOMP_CMAKE_DIR}/${PACKAGE_NAME}-targets.cmake\")

set(libgeodecomp_INCLUDE_DIRS
  \"\${LIBGEODECOMP_CMAKE_DIR}/../../../include\"
  ${INSTALL_INCLUDE_DIRECTORIES}
)

set(libgeodecomp_FLAGS \"${LGD_AGGREGATED_CXX_FLAGS}\")
")

set(ConfigPackageLocation lib/cmake/${PACKAGE_NAME})
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake"
  DESTINATION "${ConfigPackageLocation}")

install(EXPORT ${PACKAGE_NAME}-targets DESTINATION ${ConfigPackageLocation})

#============= 7. TYPEMAPS GENERATION ================================
if (WITH_TYPEMAPS)
  set_directory_properties(PROPERTIES CLEAN_NO_CUSTOM true)

  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/libgeodecomp/communication/typemaps.h" "${CMAKE_CURRENT_SOURCE_DIR}/libgeodecomp/communication/typemaps.cpp"
    COMMAND cd "${CMAKE_CURRENT_SOURCE_DIR}" && doxygen doxygen.conf
    COMMAND ruby "${CMAKE_CURRENT_SOURCE_DIR}/../tools/typemapgenerator/generate.rb" --namespace LibGeoDecomp --header-fix "^.+/src:libgeodecomp" --macro-guard-mpi LIBGEODECOMP_WITH_MPI --macro-guard-hpx LIBGEODECOMP_WITH_HPX --macro-guard-boost LIBGEODECOMP_WITH_BOOST_SERIALIZATION "${CMAKE_CURRENT_SOURCE_DIR}/../doc/xml" "${CMAKE_CURRENT_SOURCE_DIR}/libgeodecomp/communication"
    DEPENDS "${CMAKE_BINARY_DIR}/cache"
    )

  list(REMOVE_ITEM ALL_HEADERS "libgeodecomp/communication/typemaps.h")
  set(CANDIDATES)
  foreach(i ${ALL_HEADERS})
    set(CANDIDATES "${i}:${CANDIDATES}")
  endforeach(i)

  add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/cache"
    COMMAND cp -a "${CMAKE_BINARY_DIR}/last_cache" "${CMAKE_BINARY_DIR}/cache" 2>/dev/null || echo "still ok >/dev/null"
    COMMAND ruby "${CMAKE_CURRENT_SOURCE_DIR}/../tools/typemapgenerator/generate.rb" --cache "${CMAKE_BINARY_DIR}/cache" "${CMAKE_CURRENT_SOURCE_DIR}" "${CANDIDATES}"
    COMMAND cp -a "${CMAKE_BINARY_DIR}/cache" "${CMAKE_BINARY_DIR}/last_cache")
endif(WITH_TYPEMAPS)

set(LIBGEODECOMP_ALL_INCLUDE_DIRECTORIES ${BUILD_INCLUDE_DIRECTORIES} ${INSTALL_INCLUDE_DIRECTORIES} PARENT_SCOPE)

set(LGD_AGGREGATED_C_FLAGS   "${LGD_AGGREGATED_C_FLAGS}"   PARENT_SCOPE)
set(LGD_AGGREGATED_CXX_FLAGS "${LGD_AGGREGATED_CXX_FLAGS}" PARENT_SCOPE)
