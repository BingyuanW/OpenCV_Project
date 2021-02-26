
cmake_minimum_required(VERSION 2.8)
project( cameracapturecanny )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
#include_directories(/usr/local/include/opencv4)
add_executable( cameracapturecanny cameracapturecanny.cpp )

target_link_libraries( cameracapturecanny ${OpenCV_LIBS} )
