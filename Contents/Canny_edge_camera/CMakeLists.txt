
cmake_minimum_required(VERSION 2.8)
project( cannycamera )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
#include_directories(/usr/local/include/opencv4)
add_executable( cannycamera CannyCamera.cpp )

target_link_libraries( cannycamera ${OpenCV_LIBS} )
