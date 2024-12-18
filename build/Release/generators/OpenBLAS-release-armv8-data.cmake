########### AGGREGATED COMPONENTS AND DEPENDENCIES FOR THE MULTI CONFIG #####################
#############################################################################################

list(APPEND openblas_COMPONENT_NAMES OpenBLAS::pthread)
list(REMOVE_DUPLICATES openblas_COMPONENT_NAMES)
if(DEFINED openblas_FIND_DEPENDENCY_NAMES)
  list(APPEND openblas_FIND_DEPENDENCY_NAMES )
  list(REMOVE_DUPLICATES openblas_FIND_DEPENDENCY_NAMES)
else()
  set(openblas_FIND_DEPENDENCY_NAMES )
endif()

########### VARIABLES #######################################################################
#############################################################################################
set(openblas_PACKAGE_FOLDER_RELEASE "/Users/yincheangng/.conan2/p/b/openbda08c8633a811/p")
set(openblas_BUILD_MODULES_PATHS_RELEASE )


set(openblas_INCLUDE_DIRS_RELEASE "${openblas_PACKAGE_FOLDER_RELEASE}/include"
			"${openblas_PACKAGE_FOLDER_RELEASE}/include/openblas")
set(openblas_RES_DIRS_RELEASE )
set(openblas_DEFINITIONS_RELEASE )
set(openblas_SHARED_LINK_FLAGS_RELEASE )
set(openblas_EXE_LINK_FLAGS_RELEASE )
set(openblas_OBJECTS_RELEASE )
set(openblas_COMPILE_DEFINITIONS_RELEASE )
set(openblas_COMPILE_OPTIONS_C_RELEASE )
set(openblas_COMPILE_OPTIONS_CXX_RELEASE )
set(openblas_LIB_DIRS_RELEASE "${openblas_PACKAGE_FOLDER_RELEASE}/lib")
set(openblas_BIN_DIRS_RELEASE )
set(openblas_LIBRARY_TYPE_RELEASE STATIC)
set(openblas_IS_HOST_WINDOWS_RELEASE 0)
set(openblas_LIBS_RELEASE openblas)
set(openblas_SYSTEM_LIBS_RELEASE )
set(openblas_FRAMEWORK_DIRS_RELEASE )
set(openblas_FRAMEWORKS_RELEASE )
set(openblas_BUILD_DIRS_RELEASE )
set(openblas_NO_SONAME_MODE_RELEASE FALSE)


# COMPOUND VARIABLES
set(openblas_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${openblas_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${openblas_COMPILE_OPTIONS_C_RELEASE}>")
set(openblas_LINKER_FLAGS_RELEASE
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openblas_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openblas_SHARED_LINK_FLAGS_RELEASE}>"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openblas_EXE_LINK_FLAGS_RELEASE}>")


set(openblas_COMPONENTS_RELEASE OpenBLAS::pthread)
########### COMPONENT OpenBLAS::pthread VARIABLES ############################################

set(openblas_OpenBLAS_pthread_INCLUDE_DIRS_RELEASE "${openblas_PACKAGE_FOLDER_RELEASE}/include"
			"${openblas_PACKAGE_FOLDER_RELEASE}/include/openblas")
set(openblas_OpenBLAS_pthread_LIB_DIRS_RELEASE "${openblas_PACKAGE_FOLDER_RELEASE}/lib")
set(openblas_OpenBLAS_pthread_BIN_DIRS_RELEASE )
set(openblas_OpenBLAS_pthread_LIBRARY_TYPE_RELEASE STATIC)
set(openblas_OpenBLAS_pthread_IS_HOST_WINDOWS_RELEASE 0)
set(openblas_OpenBLAS_pthread_RES_DIRS_RELEASE )
set(openblas_OpenBLAS_pthread_DEFINITIONS_RELEASE )
set(openblas_OpenBLAS_pthread_OBJECTS_RELEASE )
set(openblas_OpenBLAS_pthread_COMPILE_DEFINITIONS_RELEASE )
set(openblas_OpenBLAS_pthread_COMPILE_OPTIONS_C_RELEASE "")
set(openblas_OpenBLAS_pthread_COMPILE_OPTIONS_CXX_RELEASE "")
set(openblas_OpenBLAS_pthread_LIBS_RELEASE openblas)
set(openblas_OpenBLAS_pthread_SYSTEM_LIBS_RELEASE )
set(openblas_OpenBLAS_pthread_FRAMEWORK_DIRS_RELEASE )
set(openblas_OpenBLAS_pthread_FRAMEWORKS_RELEASE )
set(openblas_OpenBLAS_pthread_DEPENDENCIES_RELEASE )
set(openblas_OpenBLAS_pthread_SHARED_LINK_FLAGS_RELEASE )
set(openblas_OpenBLAS_pthread_EXE_LINK_FLAGS_RELEASE )
set(openblas_OpenBLAS_pthread_NO_SONAME_MODE_RELEASE FALSE)

# COMPOUND VARIABLES
set(openblas_OpenBLAS_pthread_LINKER_FLAGS_RELEASE
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:${openblas_OpenBLAS_pthread_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:${openblas_OpenBLAS_pthread_SHARED_LINK_FLAGS_RELEASE}>
        $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${openblas_OpenBLAS_pthread_EXE_LINK_FLAGS_RELEASE}>
)
set(openblas_OpenBLAS_pthread_COMPILE_OPTIONS_RELEASE
    "$<$<COMPILE_LANGUAGE:CXX>:${openblas_OpenBLAS_pthread_COMPILE_OPTIONS_CXX_RELEASE}>"
    "$<$<COMPILE_LANGUAGE:C>:${openblas_OpenBLAS_pthread_COMPILE_OPTIONS_C_RELEASE}>")