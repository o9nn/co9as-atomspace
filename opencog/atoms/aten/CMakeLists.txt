# ATen tensor integration for OpenCog AtomSpace
#
# This module provides PyTorch ATen tensor support for AtomSpace.
# When ATen is available, it uses native PyTorch tensors.
# When ATen is not available, it falls back to a basic implementation.

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

# Try to find PyTorch/ATen
FIND_PACKAGE(Torch QUIET)

IF(Torch_FOUND)
    MESSAGE(STATUS "PyTorch found: ${TORCH_LIBRARIES}")
    ADD_DEFINITIONS(-DHAVE_ATEN)
    SET(ATEN_LIBRARIES ${TORCH_LIBRARIES})
    INCLUDE_DIRECTORIES(${TORCH_INCLUDE_DIRS})
ELSE()
    MESSAGE(STATUS "PyTorch not found - using fallback tensor implementation")
    SET(ATEN_LIBRARIES "")
ENDIF()

ADD_LIBRARY(aten
    ATenValue.cc
    ATenSpace.cc
    TensorLink.cc
)

# Without this, parallel make will race and crap up the generated files.
ADD_DEPENDENCIES(aten opencog_atom_types)

TARGET_LINK_LIBRARIES(aten
    atom_types
    value
    base
    ${ATEN_LIBRARIES}
    ${COGUTIL_LIBRARY}
)

# Install the library
INSTALL(TARGETS aten EXPORT AtomSpaceTargets
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/opencog"
)

# Install headers
INSTALL(FILES
    ATenValue.h
    ATenSpace.h
    TensorLink.h
    DESTINATION "include/opencog/atoms/aten"
)
