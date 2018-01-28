cmake_minimum_required(VERSION 2.8)
project( Patentes )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Patentes Patentes.cpp )
target_link_libraries( Patentes ${OpenCV_LIBS} )






