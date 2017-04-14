# Install script for directory: /home/mopkobka/CourseWork/py-faster-rcnn/caffe-fast-rcnn-py3/python

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/mopkobka/CourseWork/py-faster-rcnn/caffe-fast-rcnn-py3/cmake-build-debug/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python" TYPE FILE FILES
    "/home/mopkobka/CourseWork/py-faster-rcnn/caffe-fast-rcnn-py3/python/classify.py"
    "/home/mopkobka/CourseWork/py-faster-rcnn/caffe-fast-rcnn-py3/python/detect.py"
    "/home/mopkobka/CourseWork/py-faster-rcnn/caffe-fast-rcnn-py3/python/draw_net.py"
    "/home/mopkobka/CourseWork/py-faster-rcnn/caffe-fast-rcnn-py3/python/requirements.txt"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE FILE FILES
    "/home/mopkobka/CourseWork/py-faster-rcnn/caffe-fast-rcnn-py3/python/caffe/__init__.py"
    "/home/mopkobka/CourseWork/py-faster-rcnn/caffe-fast-rcnn-py3/python/caffe/classifier.py"
    "/home/mopkobka/CourseWork/py-faster-rcnn/caffe-fast-rcnn-py3/python/caffe/detector.py"
    "/home/mopkobka/CourseWork/py-faster-rcnn/caffe-fast-rcnn-py3/python/caffe/draw.py"
    "/home/mopkobka/CourseWork/py-faster-rcnn/caffe-fast-rcnn-py3/python/caffe/io.py"
    "/home/mopkobka/CourseWork/py-faster-rcnn/caffe-fast-rcnn-py3/python/caffe/net_spec.py"
    "/home/mopkobka/CourseWork/py-faster-rcnn/caffe-fast-rcnn-py3/python/caffe/pycaffe.py"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe-d.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe-d.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe-d.so"
         RPATH "/home/mopkobka/CourseWork/py-faster-rcnn/caffe-fast-rcnn-py3/cmake-build-debug/install/lib:/usr/local/cuda-8.0/lib64:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE SHARED_LIBRARY FILES "/home/mopkobka/CourseWork/py-faster-rcnn/caffe-fast-rcnn-py3/cmake-build-debug/lib/_caffe-d.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe-d.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe-d.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe-d.so"
         OLD_RPATH "/usr/local/cuda-8.0/lib64:/home/mopkobka/CourseWork/py-faster-rcnn/caffe-fast-rcnn-py3/cmake-build-debug/lib:/usr/local/lib::::::::"
         NEW_RPATH "/home/mopkobka/CourseWork/py-faster-rcnn/caffe-fast-rcnn-py3/cmake-build-debug/install/lib:/usr/local/cuda-8.0/lib64:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/caffe/_caffe-d.so")
    endif()
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE DIRECTORY FILES
    "/home/mopkobka/CourseWork/py-faster-rcnn/caffe-fast-rcnn-py3/python/caffe/imagenet"
    "/home/mopkobka/CourseWork/py-faster-rcnn/caffe-fast-rcnn-py3/python/caffe/proto"
    "/home/mopkobka/CourseWork/py-faster-rcnn/caffe-fast-rcnn-py3/python/caffe/test"
    )
endif()

