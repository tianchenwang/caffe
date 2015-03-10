#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "caffe" for configuration "Release"
set_property(TARGET caffe APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(caffe PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "proto;proto;/afs/cs.stanford.edu/u/twangcat/scratch/packages/lib/libboost_system.so;/afs/cs.stanford.edu/u/twangcat/scratch/packages/lib/libboost_thread.so;/afs/cs.stanford.edu/u/twangcat/scratch/packages/lib/libboost_python.so;/afs/cs.stanford.edu/u/twangcat/scratch/packages/lib/libboost_numpy.so;-lpthread;/usr/local/lib/libglog.so;/usr/lib64/libgflags.so;/usr/local/lib/libprotobuf.so;-lpthread;/usr/lib64/libhdf5_hl.so;/usr/lib64/libhdf5.so;/usr/lib64/liblmdb.so;/usr/lib64/libleveldb.so;/usr/lib64/libsnappy.so;/usr/local/cuda/lib64/libcudart.so;/usr/local/cuda/lib64/libcurand.so;/usr/local/cuda/lib64/libcublas.so;/usr/local/cuda/lib64/libcudnn.so;opencv_core;opencv_highgui;opencv_imgproc;/usr/lib64/atlas/liblapack.so;/usr/lib64/atlas/libptcblas.so;/usr/lib64/atlas/libatlas.so;/afs/cs.stanford.edu/u/twangcat/scratch/packages/lib/libmpicxx.so;/afs/cs.stanford.edu/u/twangcat/scratch/packages/lib/libmpi.so"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcaffe.so"
  IMPORTED_SONAME_RELEASE "libcaffe.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS caffe )
list(APPEND _IMPORT_CHECK_FILES_FOR_caffe "${_IMPORT_PREFIX}/lib/libcaffe.so" )

# Import target "proto" for configuration "Release"
set_property(TARGET proto APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(proto PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libproto.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS proto )
list(APPEND _IMPORT_CHECK_FILES_FOR_proto "${_IMPORT_PREFIX}/lib/libproto.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
