//
// Created by zhou on 22-5-3.
//

#ifndef ORB_SLAM3_ALIKEEXTRACTOR_H
#define ORB_SLAM3_ALIKEEXTRACTOR_H

#include <list>
#include <vector>
#include <string>
#include <map>
#include <math.h>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <torch/script.h>
#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/nn/functional.h>
#include <torchvision/models/resnet.h>

#include "BaseExtractor.h"
#include "soft_detect.h"

namespace ORB_SLAM3
{
typedef torch::nn::Module Module;
typedef torch::nn::Conv2d Conv2d;
typedef torch::nn::BatchNorm2d BN2d;
typedef torch::Tensor Tensor;
typedef torch::nn::ReLU RELU;


class ConvBlock : Module{
public:
    ConvBlock(int in_channels, int out_channels);
    Tensor & forward(Tensor &x);

protected:
    Conv2d conv1, conv2;
    BN2d bn1, bn2;
    RELU gate;
};

class ResBlock : Module{
public:
    ResBlock(int inplanes, int planes, Conv2d downsample, int stride = 1, int groups = 1, int base_width = 64, int dilation = 1);
    Tensor & forward(Tensor &x);

    static int expansion;

protected:
    Conv2d conv1, conv2, downsample;
    BN2d bn1, bn2;
    RELU gate;
    int stride;
};

class ALnet : Module{
public:
    ALnet(int c1, int c2, int c3, int c4, int dim, bool single_head);
    std::vector<Tensor> & forward(const Tensor &image);

protected:
    RELU gate;
    torch::nn::MaxPool2d pool2, pool4;
    ConvBlock block1;
    ResBlock block2, block3, block4;
    Conv2d conv1, conv2, conv3, conv4, convhead1, convhead2;
    torch::nn::Upsample upsample2, upsample4, upsample8, upsample32;
    bool single_head;
};

class ALike : public ALnet {
public:
    ALike(int c1 = 32, int c2 = 64, int c3 = 128, int c4 = 128, int dim = 128, bool single_head = false,
          int radius_ = 2, int top_k_ = 500, float scores_th_ = 0.5, int n_limit_ = 5000);
    std::vector<Tensor> & forward(const cv::Mat &img, const torch::Device device, bool sub_pixel);

protected:
    std::vector<Tensor> & extract_dense_map(Tensor &image);

    int radius, top_k, n_limit;
    float scores_th;
    ORB_SLAM3::DKD dkd;
    c10::DeviceType device;
};

class ALIKEextractor : public BaseExtractor{
public:
    ALIKEextractor(int _model, int _device, int _topk, float _scores_th, int _nlimit);

    virtual ~ALIKEextractor() = default;

    int operator()( cv::InputArray _image, cv::InputArray _mask,
                     std::vector<cv::KeyPoint>& _keypoints,
                     cv::OutputArray _descriptors, std::vector<int> &vLappingArea);

protected:
    // std::vector<Tensor> extract_dense_map();
    torch::Device device_;
    std::shared_ptr<ALike> model_;
    std::map<std::string, int> configs;
};
}

#endif //ORB_SLAM3_ALIKEEXTRACTOR_H
