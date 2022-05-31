//
// Created by zhou on 22-4-28.
//

#ifndef ORB_SLAM3_ALIKE_ONNX_H
#define ORB_SLAM3_ALIKE_ONNX_H

#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>
#include <cmath>
#include <mutex>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

namespace ORB_SLAM3{

    class Alike{
    public:
        Alike();
        Alike(std::string modelPath);

        int operator() (cv::InputArray _image);
        int operator() (cv::InputArray _image, cv::InputArray _mask,
                        std::vector<cv::KeyPoint>& _keypoints,
                        cv::OutputArray _descriptors, std::vector<int> &vLappingArea);

    private:
        std::string modelOnnx;
        cv::dnn::Net net;
        //cv::FileStorage cfg;
        //cv::Size inputSize;
    };
}

#endif //ORB_SLAM3_ALIKE_ONNX_H
