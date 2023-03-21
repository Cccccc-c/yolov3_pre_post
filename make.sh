#!/bin/bash
g++ ./src/yolov3_int8.cpp -I /usr/include/opencv4 -L /usr/lib/aarch64-linux-gnu -ggdb -O3 -o yolov3 -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core
