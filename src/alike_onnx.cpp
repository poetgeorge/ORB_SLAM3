//
// Created by zhou on 22-4-28.
//

#include "alike_onnx.h"

namespace ORB_SLAM3{
    Alike::Alike() {
        modelOnnx = "../models/alike_n_1000.onnx";
        net = cv::dnn::readNetFromONNX(modelOnnx);
        if (net.empty()) {
            std::cerr << "Can't load the network!!" << std::endl;
            std::cerr << "model path: " << modelOnnx << std::endl;
            exit(-1);
        }
    }

    Alike::Alike(std::string modelPath) {
        modelOnnx = modelPath;
        net = cv::dnn::readNetFromONNX(modelOnnx);
        if (net.empty()) {
            std::cerr << "Can't load the network!!" << std::endl;
            std::cerr << "model path: " << modelOnnx << std::endl;
            exit(-1);
        }
    }

    int Alike::operator()(cv::InputArray _image) {
        if(_image.empty()){
            std::cerr<<"Empty Image!"<<std::endl;
            return -1;
        }
        cv::Mat imageInt = _image.getMat();
        cv::Mat image;
        imageInt.convertTo(image, CV_32FC3);
        //assert(image.type() == CV_8UC3);
        cv::Mat blob;
        cv::dnn::blobFromImage(image, blob, 255, cv::Size(), cv::Scalar(), true, false);
        net.setInput(blob);
        cv::Mat output = net.forward();
        std::vector<cv::Mat> ress;
        cv::dnn::imagesFromBlob(output, ress);
        cv::Mat res;
        std::cout<<res.size()<<std::endl;
        return 0;
    }
}