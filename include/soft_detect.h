//
// Created by zhou on 22-5-9.
//

#ifndef ORB_SLAM3_SOFT_DETECT_H
#define ORB_SLAM3_SOFT_DETECT_H

#include <vector>
#include <tuple>

#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/nn/functional.h>


namespace ORB_SLAM3{
typedef torch::nn::Module Module;
typedef torch::Tensor Tensor;

class DKD : Module {
public:
    DKD(int radius_ = 2, int top_k_ = 0, float scores_th_ = 0.2, int n_limit_ = 20000);
    std::vector<std::vector<Tensor>> & forward(const Tensor &scores_map, const Tensor &descriptor_map, bool sub_pixel = true);

protected:
    Tensor & simple_nms(const Tensor &scores, int nms_radius);
    std::vector<Tensor> & sample_descriptor(const Tensor &descriptor_map, const std::vector<Tensor> &kpts, bool bilinear_interp = false);
    Tensor & max_pool(const Tensor &x, int r);
    std::vector<std::vector<Tensor>> & detect_keypoints(const Tensor &scores_map, bool sub_pixel = true);

    int radius;
    int top_k;
    float scores_th;
    int n_limit;
    int kernel_size;
    float tdet;
    torch::nn::Unfold _unfold;
    Tensor hw_grid;
};
}


#endif //ORB_SLAM3_SOFT_DETECT_H
