project( Python )

#    Python
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

#    Boost.Python
find_package(Boost 1.46 COMPONENTS python REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

set(CAFFE_MODULE ${CMAKE_CURRENT_BINARY_DIR}/caffe)

# create caffe module
file(MAKE_DIRECTORY ${CAFFE_MODULE})

# copy the python source files
file(GLOB_RECURSE PYTHON_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
file(GLOB_RECURSE NUMPY_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.npy)
add_custom_command(
  OUTPUT pysrc_output
  # Copy over python source files
  COMMAND cp
  ARGS ${CMAKE_CURRENT_SOURCE_DIR}/caffe/*py ${CAFFE_MODULE}
  # Copy over imagenet folder
  COMMAND cp
  ARGS -r ${CMAKE_CURRENT_SOURCE_DIR}/caffe/imagenet ${CAFFE_MODULE}
  # Dependencies
  DEPENDS ${PYTHON_SOURCES} ${NUMPY_FILES}
)

add_custom_target(
  pysrc
  DEPENDS pysrc_output
)

# create proto module
file(MAKE_DIRECTORY ${CAFFE_MODULE}/proto)
add_custom_command(
  OUTPUT pyproto_output
  # caffe.proto
  COMMAND protoc
  ARGS --proto_path=${CMAKE_SOURCE_DIR}/src/caffe/proto
  --python_out=${CAFFE_MODULE}/proto
  ${CMAKE_SOURCE_DIR}/src/caffe/proto/caffe.proto
  # caffe_pretty_print.proto
  COMMAND protoc
  ARGS --proto_path=${CMAKE_SOURCE_DIR}/src/caffe/proto
  --python_out=${CAFFE_MODULE}/proto
  ${CMAKE_SOURCE_DIR}/src/caffe/proto/caffe_pretty_print.proto
  # __init__.py
  COMMAND touch
  ARGS ${CAFFE_MODULE}/proto/__init__.py
  # Dependencies
  DEPENDS ${CMAKE_SOURCE_DIR}/src/caffe/proto/caffe.proto
  ${CMAKE_SOURCE_DIR}/src/caffe/proto/caffe_pretty_print.proto
)
add_custom_target(
  pyproto
  DEPENDS pyproto_output
)

# create the python wrapper for caffe, then place in module
file(GLOB_RECURSE Python_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(pycaffe_lib SHARED ${Python_SOURCES})
target_link_libraries(pycaffe_lib caffe ${PYTHON_LIBRARIES}
  ${Boost_LIBRARIES})
set_target_properties(pycaffe_lib PROPERTIES OUTPUT_NAME _caffe)
set_target_properties(pycaffe_lib PROPERTIES PREFIX "")
set_target_properties(pycaffe_lib PROPERTIES suffix .so)
set_target_properties(pycaffe_lib PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CAFFE_MODULE})


# copy the library and protos back to the source
get_property(pycaffe_lib_location TARGET pycaffe_lib PROPERTY LOCATION)
add_custom_command(
  OUTPUT pycaffe_insrc_output
  # Copy the caffe wrapper
  COMMAND cp
  ARGS ${pycaffe_lib_location} ${CMAKE_CURRENT_SOURCE_DIR}/caffe
  # Copy the protobuf folder
  COMMAND cp
  ARGS -r ${CAFFE_MODULE}/proto ${CMAKE_CURRENT_SOURCE_DIR}/caffe
  # Dependencies
  DEPENDS pycaffe_lib pyproto_output
)

add_custom_target(
  pycaffe_insrc
  DEPENDS pycaffe_insrc_output pysrc pyproto pycaffe_lib
)

# create the final pycaffe target which runs the above
add_custom_target(
  pycaffe ALL
  DEPENDS pysrc pyproto pycaffe_lib pycaffe_insrc
)

###    Install    #################################################################################

install(DIRECTORY caffe DESTINATION python)
install(FILES requirements.txt DESTINATION python)
