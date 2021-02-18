# CMakeList.txt : CMake project for opencl-motion-magnification, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
set (CMAKE_CXX_STANDARD 17)
project ("omm")
find_package(OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS}) 
add_library (Argtable argtable3.c)

# Add source to this project's executable.
add_executable (omm "omm.cpp" "PhaseDiffAndAmp.hpp" "AmpPhaseShift.hpp")
target_link_libraries(omm Argtable ${OpenCV_LIBRARIES})

# TODO: Add tests and install targets if needed.
