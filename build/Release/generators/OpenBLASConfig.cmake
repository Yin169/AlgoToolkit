########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

if(OpenBLAS_FIND_QUIETLY)
    set(OpenBLAS_MESSAGE_MODE VERBOSE)
else()
    set(OpenBLAS_MESSAGE_MODE STATUS)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/OpenBLASTargets.cmake)
include(CMakeFindDependencyMacro)

check_build_type_defined()

foreach(_DEPENDENCY ${openblas_FIND_DEPENDENCY_NAMES} )
    # Check that we have not already called a find_package with the transitive dependency
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED ${${_DEPENDENCY}_FIND_MODE})
    endif()
endforeach()

set(OpenBLAS_VERSION_STRING "0.3.27")
set(OpenBLAS_INCLUDE_DIRS ${openblas_INCLUDE_DIRS_RELEASE} )
set(OpenBLAS_INCLUDE_DIR ${openblas_INCLUDE_DIRS_RELEASE} )
set(OpenBLAS_LIBRARIES ${openblas_LIBRARIES_RELEASE} )
set(OpenBLAS_DEFINITIONS ${openblas_DEFINITIONS_RELEASE} )


# Only the last installed configuration BUILD_MODULES are included to avoid the collision
foreach(_BUILD_MODULE ${openblas_BUILD_MODULES_PATHS_RELEASE} )
    message(${OpenBLAS_MESSAGE_MODE} "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()


