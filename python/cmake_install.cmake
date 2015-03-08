# Install script for directory: /afs/cs.stanford.edu/u/twangcat/scratch/caffenet1.0/python

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/afs/cs.stanford.edu/u/twangcat/scratch/caffenet1.0/install")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "Release")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "0")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python" TYPE FILE FILES
    "/afs/cs.stanford.edu/u/twangcat/scratch/caffenet1.0/python/classify.py"
    "/afs/cs.stanford.edu/u/twangcat/scratch/caffenet1.0/python/convert_driving_data.py"
    "/afs/cs.stanford.edu/u/twangcat/scratch/caffenet1.0/python/detect.py"
    "/afs/cs.stanford.edu/u/twangcat/scratch/caffenet1.0/python/draw_net.py"
    "/afs/cs.stanford.edu/u/twangcat/scratch/caffenet1.0/python/requirements.txt"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE FILE FILES
    "/afs/cs.stanford.edu/u/twangcat/scratch/caffenet1.0/python/caffe/__init__.py"
    "/afs/cs.stanford.edu/u/twangcat/scratch/caffenet1.0/python/caffe/classifier.py"
    "/afs/cs.stanford.edu/u/twangcat/scratch/caffenet1.0/python/caffe/detector.py"
    "/afs/cs.stanford.edu/u/twangcat/scratch/caffenet1.0/python/caffe/draw.py"
    "/afs/cs.stanford.edu/u/twangcat/scratch/caffenet1.0/python/caffe/io.py"
    "/afs/cs.stanford.edu/u/twangcat/scratch/caffenet1.0/python/caffe/pycaffe.py"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE SHARED_LIBRARY FILES "/afs/cs.stanford.edu/u/twangcat/scratch/caffenet1.0/python/CMakeFiles/CMakeRelink.dir/_caffe.so")
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE DIRECTORY FILES
    "/afs/cs.stanford.edu/u/twangcat/scratch/caffenet1.0/python/caffe/imagenet"
    "/afs/cs.stanford.edu/u/twangcat/scratch/caffenet1.0/python/caffe/proto"
    "/afs/cs.stanford.edu/u/twangcat/scratch/caffenet1.0/python/caffe/test"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

