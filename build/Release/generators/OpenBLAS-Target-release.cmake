# Avoid multiple calls to find_package to append duplicated properties to the targets
include_guard()########### VARIABLES #######################################################################
#############################################################################################
set(openblas_FRAMEWORKS_FOUND_RELEASE "") # Will be filled later
conan_find_apple_frameworks(openblas_FRAMEWORKS_FOUND_RELEASE "${openblas_FRAMEWORKS_RELEASE}" "${openblas_FRAMEWORK_DIRS_RELEASE}")

set(openblas_LIBRARIES_TARGETS "") # Will be filled later


######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
if(NOT TARGET openblas_DEPS_TARGET)
    add_library(openblas_DEPS_TARGET INTERFACE IMPORTED)
endif()

set_property(TARGET openblas_DEPS_TARGET
             APPEND PROPERTY INTERFACE_LINK_LIBRARIES
             $<$<CONFIG:Release>:${openblas_FRAMEWORKS_FOUND_RELEASE}>
             $<$<CONFIG:Release>:${openblas_SYSTEM_LIBS_RELEASE}>
             $<$<CONFIG:Release>:>)

####### Find the libraries declared in cpp_info.libs, create an IMPORTED target for each one and link the
####### openblas_DEPS_TARGET to all of them
conan_package_library_targets("${openblas_LIBS_RELEASE}"    # libraries
                              "${openblas_LIB_DIRS_RELEASE}" # package_libdir
                              "${openblas_BIN_DIRS_RELEASE}" # package_bindir
                              "${openblas_LIBRARY_TYPE_RELEASE}"
                              "${openblas_IS_HOST_WINDOWS_RELEASE}"
                              openblas_DEPS_TARGET
                              openblas_LIBRARIES_TARGETS  # out_libraries_targets
                              "_RELEASE"
                              "openblas"    # package_name
                              "${openblas_NO_SONAME_MODE_RELEASE}")  # soname

# FIXME: What is the result of this for multi-config? All configs adding themselves to path?
set(CMAKE_MODULE_PATH ${openblas_BUILD_DIRS_RELEASE} ${CMAKE_MODULE_PATH})

########## COMPONENTS TARGET PROPERTIES Release ########################################

    ########## COMPONENT OpenBLAS::pthread #############

        set(openblas_OpenBLAS_pthread_FRAMEWORKS_FOUND_RELEASE "")
        conan_find_apple_frameworks(openblas_OpenBLAS_pthread_FRAMEWORKS_FOUND_RELEASE "${openblas_OpenBLAS_pthread_FRAMEWORKS_RELEASE}" "${openblas_OpenBLAS_pthread_FRAMEWORK_DIRS_RELEASE}")

        set(openblas_OpenBLAS_pthread_LIBRARIES_TARGETS "")

        ######## Create an interface target to contain all the dependencies (frameworks, system and conan deps)
        if(NOT TARGET openblas_OpenBLAS_pthread_DEPS_TARGET)
            add_library(openblas_OpenBLAS_pthread_DEPS_TARGET INTERFACE IMPORTED)
        endif()

        set_property(TARGET openblas_OpenBLAS_pthread_DEPS_TARGET
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${openblas_OpenBLAS_pthread_FRAMEWORKS_FOUND_RELEASE}>
                     $<$<CONFIG:Release>:${openblas_OpenBLAS_pthread_SYSTEM_LIBS_RELEASE}>
                     $<$<CONFIG:Release>:${openblas_OpenBLAS_pthread_DEPENDENCIES_RELEASE}>
                     )

        ####### Find the libraries declared in cpp_info.component["xxx"].libs,
        ####### create an IMPORTED target for each one and link the 'openblas_OpenBLAS_pthread_DEPS_TARGET' to all of them
        conan_package_library_targets("${openblas_OpenBLAS_pthread_LIBS_RELEASE}"
                              "${openblas_OpenBLAS_pthread_LIB_DIRS_RELEASE}"
                              "${openblas_OpenBLAS_pthread_BIN_DIRS_RELEASE}" # package_bindir
                              "${openblas_OpenBLAS_pthread_LIBRARY_TYPE_RELEASE}"
                              "${openblas_OpenBLAS_pthread_IS_HOST_WINDOWS_RELEASE}"
                              openblas_OpenBLAS_pthread_DEPS_TARGET
                              openblas_OpenBLAS_pthread_LIBRARIES_TARGETS
                              "_RELEASE"
                              "openblas_OpenBLAS_pthread"
                              "${openblas_OpenBLAS_pthread_NO_SONAME_MODE_RELEASE}")


        ########## TARGET PROPERTIES #####################################
        set_property(TARGET OpenBLAS::pthread
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     $<$<CONFIG:Release>:${openblas_OpenBLAS_pthread_OBJECTS_RELEASE}>
                     $<$<CONFIG:Release>:${openblas_OpenBLAS_pthread_LIBRARIES_TARGETS}>
                     )

        if("${openblas_OpenBLAS_pthread_LIBS_RELEASE}" STREQUAL "")
            # If the component is not declaring any "cpp_info.components['foo'].libs" the system, frameworks etc are not
            # linked to the imported targets and we need to do it to the global target
            set_property(TARGET OpenBLAS::pthread
                         APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                         openblas_OpenBLAS_pthread_DEPS_TARGET)
        endif()

        set_property(TARGET OpenBLAS::pthread APPEND PROPERTY INTERFACE_LINK_OPTIONS
                     $<$<CONFIG:Release>:${openblas_OpenBLAS_pthread_LINKER_FLAGS_RELEASE}>)
        set_property(TARGET OpenBLAS::pthread APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<$<CONFIG:Release>:${openblas_OpenBLAS_pthread_INCLUDE_DIRS_RELEASE}>)
        set_property(TARGET OpenBLAS::pthread APPEND PROPERTY INTERFACE_LINK_DIRECTORIES
                     $<$<CONFIG:Release>:${openblas_OpenBLAS_pthread_LIB_DIRS_RELEASE}>)
        set_property(TARGET OpenBLAS::pthread APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
                     $<$<CONFIG:Release>:${openblas_OpenBLAS_pthread_COMPILE_DEFINITIONS_RELEASE}>)
        set_property(TARGET OpenBLAS::pthread APPEND PROPERTY INTERFACE_COMPILE_OPTIONS
                     $<$<CONFIG:Release>:${openblas_OpenBLAS_pthread_COMPILE_OPTIONS_RELEASE}>)

    ########## AGGREGATED GLOBAL TARGET WITH THE COMPONENTS #####################
    set_property(TARGET OpenBLAS::OpenBLAS APPEND PROPERTY INTERFACE_LINK_LIBRARIES OpenBLAS::pthread)

########## For the modules (FindXXX)
set(openblas_LIBRARIES_RELEASE OpenBLAS::OpenBLAS)
