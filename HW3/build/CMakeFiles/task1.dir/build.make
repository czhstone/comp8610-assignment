# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.28.3/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.28.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/ernest/Desktop/comp8610/HW3/code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/ernest/Desktop/comp8610/HW3/code/build

# Include any dependencies generated for this target.
include CMakeFiles/task1.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/task1.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/task1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/task1.dir/flags.make

CMakeFiles/task1.dir/task1.cpp.o: CMakeFiles/task1.dir/flags.make
CMakeFiles/task1.dir/task1.cpp.o: /Users/ernest/Desktop/comp8610/HW3/code/task1.cpp
CMakeFiles/task1.dir/task1.cpp.o: CMakeFiles/task1.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ernest/Desktop/comp8610/HW3/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/task1.dir/task1.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/task1.dir/task1.cpp.o -MF CMakeFiles/task1.dir/task1.cpp.o.d -o CMakeFiles/task1.dir/task1.cpp.o -c /Users/ernest/Desktop/comp8610/HW3/code/task1.cpp

CMakeFiles/task1.dir/task1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/task1.dir/task1.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ernest/Desktop/comp8610/HW3/code/task1.cpp > CMakeFiles/task1.dir/task1.cpp.i

CMakeFiles/task1.dir/task1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/task1.dir/task1.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ernest/Desktop/comp8610/HW3/code/task1.cpp -o CMakeFiles/task1.dir/task1.cpp.s

# Object files for target task1
task1_OBJECTS = \
"CMakeFiles/task1.dir/task1.cpp.o"

# External object files for target task1
task1_EXTERNAL_OBJECTS =

task1: CMakeFiles/task1.dir/task1.cpp.o
task1: CMakeFiles/task1.dir/build.make
task1: /usr/local/lib/libopencv_gapi.4.9.0.dylib
task1: /usr/local/lib/libopencv_stitching.4.9.0.dylib
task1: /usr/local/lib/libopencv_alphamat.4.9.0.dylib
task1: /usr/local/lib/libopencv_aruco.4.9.0.dylib
task1: /usr/local/lib/libopencv_bgsegm.4.9.0.dylib
task1: /usr/local/lib/libopencv_bioinspired.4.9.0.dylib
task1: /usr/local/lib/libopencv_ccalib.4.9.0.dylib
task1: /usr/local/lib/libopencv_dnn_objdetect.4.9.0.dylib
task1: /usr/local/lib/libopencv_dnn_superres.4.9.0.dylib
task1: /usr/local/lib/libopencv_dpm.4.9.0.dylib
task1: /usr/local/lib/libopencv_face.4.9.0.dylib
task1: /usr/local/lib/libopencv_freetype.4.9.0.dylib
task1: /usr/local/lib/libopencv_fuzzy.4.9.0.dylib
task1: /usr/local/lib/libopencv_hfs.4.9.0.dylib
task1: /usr/local/lib/libopencv_img_hash.4.9.0.dylib
task1: /usr/local/lib/libopencv_intensity_transform.4.9.0.dylib
task1: /usr/local/lib/libopencv_line_descriptor.4.9.0.dylib
task1: /usr/local/lib/libopencv_mcc.4.9.0.dylib
task1: /usr/local/lib/libopencv_quality.4.9.0.dylib
task1: /usr/local/lib/libopencv_rapid.4.9.0.dylib
task1: /usr/local/lib/libopencv_reg.4.9.0.dylib
task1: /usr/local/lib/libopencv_rgbd.4.9.0.dylib
task1: /usr/local/lib/libopencv_saliency.4.9.0.dylib
task1: /usr/local/lib/libopencv_sfm.4.9.0.dylib
task1: /usr/local/lib/libopencv_stereo.4.9.0.dylib
task1: /usr/local/lib/libopencv_structured_light.4.9.0.dylib
task1: /usr/local/lib/libopencv_superres.4.9.0.dylib
task1: /usr/local/lib/libopencv_surface_matching.4.9.0.dylib
task1: /usr/local/lib/libopencv_tracking.4.9.0.dylib
task1: /usr/local/lib/libopencv_videostab.4.9.0.dylib
task1: /usr/local/lib/libopencv_viz.4.9.0.dylib
task1: /usr/local/lib/libopencv_wechat_qrcode.4.9.0.dylib
task1: /usr/local/lib/libopencv_xfeatures2d.4.9.0.dylib
task1: /usr/local/lib/libopencv_xobjdetect.4.9.0.dylib
task1: /usr/local/lib/libopencv_xphoto.4.9.0.dylib
task1: libmesh.dylib
task1: /usr/local/lib/libopencv_shape.4.9.0.dylib
task1: /usr/local/lib/libopencv_highgui.4.9.0.dylib
task1: /usr/local/lib/libopencv_datasets.4.9.0.dylib
task1: /usr/local/lib/libopencv_plot.4.9.0.dylib
task1: /usr/local/lib/libopencv_text.4.9.0.dylib
task1: /usr/local/lib/libopencv_ml.4.9.0.dylib
task1: /usr/local/lib/libopencv_phase_unwrapping.4.9.0.dylib
task1: /usr/local/lib/libopencv_optflow.4.9.0.dylib
task1: /usr/local/lib/libopencv_ximgproc.4.9.0.dylib
task1: /usr/local/lib/libopencv_video.4.9.0.dylib
task1: /usr/local/lib/libopencv_videoio.4.9.0.dylib
task1: /usr/local/lib/libopencv_imgcodecs.4.9.0.dylib
task1: /usr/local/lib/libopencv_objdetect.4.9.0.dylib
task1: /usr/local/lib/libopencv_calib3d.4.9.0.dylib
task1: /usr/local/lib/libopencv_dnn.4.9.0.dylib
task1: /usr/local/lib/libopencv_features2d.4.9.0.dylib
task1: /usr/local/lib/libopencv_flann.4.9.0.dylib
task1: /usr/local/lib/libopencv_photo.4.9.0.dylib
task1: /usr/local/lib/libopencv_imgproc.4.9.0.dylib
task1: /usr/local/lib/libopencv_core.4.9.0.dylib
task1: CMakeFiles/task1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/ernest/Desktop/comp8610/HW3/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable task1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/task1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/task1.dir/build: task1
.PHONY : CMakeFiles/task1.dir/build

CMakeFiles/task1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/task1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/task1.dir/clean

CMakeFiles/task1.dir/depend:
	cd /Users/ernest/Desktop/comp8610/HW3/code/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/ernest/Desktop/comp8610/HW3/code /Users/ernest/Desktop/comp8610/HW3/code /Users/ernest/Desktop/comp8610/HW3/code/build /Users/ernest/Desktop/comp8610/HW3/code/build /Users/ernest/Desktop/comp8610/HW3/code/build/CMakeFiles/task1.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/task1.dir/depend

