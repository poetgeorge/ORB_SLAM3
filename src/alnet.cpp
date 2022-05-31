//
// Created by zhou on 22-5-11.
//

#include "ALIKEextractor.h"

typedef torch::nn::Module Module;
typedef torch::Tensor Tensor;
namespace IDX = torch::indexing;

ORB_SLAM3::ConvBlock::ConvBlock(int in_channels, int out_channels)
:conv1(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1).bias(false)),
 conv2(torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1).bias(false)),
 bn1(torch::nn::BatchNorm2dOptions(out_channels)),
 bn2(torch::nn::BatchNorm2dOptions(out_channels)),
 gate(torch::nn::ReLUOptions().inplace(true)){
}

Tensor & ORB_SLAM3::ConvBlock::forward(Tensor &x) {
    x = gate(bn1(conv1(x)));
    x = gate(bn2(conv2(x)));
    return x;
}

int ORB_SLAM3::ResBlock::expansion = 1;

ORB_SLAM3::ResBlock::ResBlock(int inplanes, int planes, Conv2d downsample, int stride, int groups, int base_width,
                              int dilation)
:conv1(torch::nn::Conv2dOptions(inplanes, planes, 3).stride(stride).padding(1).bias(false)),
conv2(torch::nn::Conv2dOptions(planes, planes, 3).padding(1).bias(false)),
bn1(torch::nn::BatchNorm2dOptions(planes)),
bn2(torch::nn::BatchNorm2dOptions(planes)),
gate(torch::nn::ReLUOptions().inplace(true)),
downsample(downsample),
stride(stride){
    if(groups!=1 || base_width!=1 || dilation>1){
        //std::cerr<<"ResBlock needs groups=1, base_width=64 and dilation<=1"<<std::endl;
        throw std::invalid_argument("ResBlock needs groups=1, base_width=64 and dilation<=1");
    }
}

Tensor & ORB_SLAM3::ResBlock::forward(Tensor &x) {
    Tensor identity = x.clone();
    Tensor out = conv1(x);
    out = bn1(out);
    out = gate(out);
    out = conv2(out);
    out = bn2(out);
    if(downsample){
        identity = downsample(x);
    }
    out += identity;
    out = gate(out);
    return out;
}

ORB_SLAM3::ALnet::ALnet(int c1, int c2, int c3, int c4, int dim, bool single_head)
:single_head(single_head),
gate(torch::nn::ReLUOptions().inplace(true)),
pool2(torch::nn::MaxPool2dOptions(2).stride(2)),
pool4(torch::nn::MaxPool2dOptions(2).stride(2)),
block1(ORB_SLAM3::ConvBlock(3, c1)),
block2(ORB_SLAM3::ResBlock(c1, c2, Conv2d(c1, c2, 1))),
block3(ORB_SLAM3::ResBlock(c2, c3, Conv2d(c2, c3, 1))),
block4(ORB_SLAM3::ResBlock(c3, c4, Conv2d(c3, c4, 1))),
conv1(torch::nn::Conv2dOptions(c1, dim/4, 1).padding(1).bias(false)),
conv2(torch::nn::Conv2dOptions(c1, dim/4, 1).padding(1).bias(false)),
conv3(torch::nn::Conv2dOptions(c1, dim/4, 1).padding(1).bias(false)),
conv4(torch::nn::Conv2dOptions(c1, dim/4, 1).padding(1).bias(false)),
upsample2(torch::nn::UpsampleOptions().mode(torch::kBilinear).align_corners(true).scale_factor(std::vector<double>({2.}))),
upsample4(torch::nn::UpsampleOptions().mode(torch::kBilinear).align_corners(true).scale_factor(std::vector<double>({4.}))),
upsample8(torch::nn::UpsampleOptions().mode(torch::kBilinear).align_corners(true).scale_factor(std::vector<double>({8.}))),
upsample32(torch::nn::UpsampleOptions().mode(torch::kBilinear).align_corners(true).scale_factor(std::vector<double>({32.}))){
    if(single_head == false){
        convhead1 = Conv2d(torch::nn::Conv2dOptions(dim, dim, 1).padding(1).bias(false));
    }
    convhead2 = Conv2d(torch::nn::Conv2dOptions(dim, dim+1, 1).padding(1).bias(false));
}

std::vector<Tensor> & ORB_SLAM3::ALnet::forward(const Tensor &image) {
    Tensor x, x1, x2, x3, x4, x2_up, x3_up, x4_up, x1234, descriptor_map, scores_map;
    x1 = block1.forward(image);
    x2 = pool2->forward(x1);
    x2 = block2.forward(x2);
    x3 = pool4->forward(x2);
    x3 = block3.forward(x3);
    x4 = pool4->forward(x3);
    x4 = block4.forward(x4);

    x1 = gate(conv1(x1));
    x2 = gate(conv2(x2));
    x3 = gate(conv3(x3));
    x4 = gate(conv4(x4));
    x2_up = upsample2(x2);
    x3_up = upsample8(x3);
    x4_up = upsample32(x4);
    x1234 = torch::cat({x1, x2_up, x3_up, x4_up}, 1);
    if(single_head == false){
        x1234 = gate(convhead1(x1234));
    }
    x = convhead2(x1234);

    descriptor_map = x.index({IDX::Slice(), IDX::Slice(0, -1), IDX::Slice(), IDX::Slice()});
    scores_map = x.index({IDX::Slice(), -1, IDX::Slice(), IDX::Slice()}).unsqueeze(1);
    std::vector<Tensor> res = {scores_map, descriptor_map};
    return res;
}
