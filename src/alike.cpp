//
// Created by zhou on 22-5-14.
//

#include "ALIKEextractor.h"

typedef torch::nn::Module Module;
typedef torch::Tensor Tensor;
namespace IDX = torch::indexing;

ORB_SLAM3::ALike::ALike(int c1, int c2, int c3, int c4, int dim, bool single_head, int radius_, int top_k_,
                        float scores_th_,
                        int n_limit_)
: radius(radius_), top_k(top_k_), n_limit(n_limit_), scores_th(scores_th_),
  ALnet(c1, c2, c3, c4, dim, single_head){
    dkd = ORB_SLAM3::DKD(radius, top_k, scores_th, n_limit);
}

std::vector<Tensor> & ORB_SLAM3::ALike::extract_dense_map(Tensor &image) {
    c10::Device device = image.device();
    int b, c, h, w;
    b = image.size(0);
    c = image.size(1);
    h = image.size(2);
    w = image.size(3);
    int h_ = h % 32 == 0 ? h : static_cast<int>(std::ceil(h/32) * 32);
    int w_ = w % 32 == 0 ? w : static_cast<int>(std::ceil(w/32) * 32);
    if(h_ != h){
        Tensor h_padding = torch::zeros({b, c, h_-h, w});
        h_padding.to(device);
        image = torch::cat({image, h_padding}, 2);
    }
    if(w_ != w){
        Tensor w_padding = torch::zeros({b, c, h, w_-w});
        w_padding.to(device);
        image = torch::cat({image, w_padding}, 3);
    }
    std::vector<Tensor> fmaps = ORB_SLAM3::ALnet::forward(image);
    Tensor scores_map = fmaps[0];
    Tensor descriptor_map = fmaps[1];
    if(h_ != h || w_ != w){
        scores_map = scores_map.index({IDX::Slice(), IDX::Slice(), IDX::Slice(0, h), IDX::Slice(0, w)});
        descriptor_map = descriptor_map.index({IDX::Slice(), IDX::Slice(), IDX::Slice(0, h), IDX::Slice(0, w)});
    }
    descriptor_map = torch::nn::functional::normalize(descriptor_map, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    std::vector<Tensor> res = {descriptor_map, scores_map};
    return res;
}

std::vector<cv::Mat> & ORB_SLAM3::ALike::forward(const cv::Mat &img, int image_size_max, bool sort, bool sub_pixel) {
    int H, W, three;
    H = img.size(0);
    W = img.size(1);
    three = img.size(2);
    assert(three == 3);
    Tensor image = img.clone();
    int maxhw = std::max(H, W);
    if(maxhw > image_size_max){
        float ratio = image_size_max / static_cast<float>(maxhw);
        image
    }
}