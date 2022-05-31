//
// Created by zhou on 22-5-9.
//

#include "soft_detect.h"
#include <assert.h>

typedef torch::nn::Module Module;
typedef torch::Tensor Tensor;
typedef torch::nn::Unfold Unfold;
namespace F = torch::nn::functional;
namespace IDX = torch::indexing;


ORB_SLAM3::DKD::DKD(int radius_, int top_k_, float scores_th_, int n_limit_)
        : radius(radius_), top_k(top_k_), scores_th(scores_th_), n_limit(n_limit_){
    kernel_size = 2 * radius + 1;
    tdet = 0.1;
    Unfold _unfold = Unfold(torch::nn::UnfoldOptions(kernel_size).padding(radius));
    Tensor x = torch::linspace(-radius, radius, kernel_size);
    hw_grid = torch::stack(torch::meshgrid({x, x})).view({2, -1}).t();
    hw_grid = hw_grid.index({IDX::Slice(), torch::tensor({1, 0})});
}


std::vector<std::vector<Tensor>> &ORB_SLAM3::DKD::detect_keypoints(const Tensor &scores_map, bool sub_pixel) {
    int b, c, h, w;
    b = scores_map.size(0);
    c = scores_map.size(1);
    h = scores_map.size(2);
    w = scores_map.size(3);

    Tensor scores_nograd = scores_map.detach();
    Tensor nms_scores = simple_nms(scores_nograd, 2);
    nms_scores.index_put_({IDX::Slice(), IDX::Slice(), IDX::Slice({0, radius+1}), IDX::Slice()}, 0);
    nms_scores.index_put_({IDX::Slice(), IDX::Slice(), IDX::Slice(), IDX::Slice({0, radius+1})}, 0);
    nms_scores.index_put_({IDX::Slice(), IDX::Slice(), IDX::Slice({ h-radius, h}), IDX::Slice()}, 0);
    nms_scores.index_put_({IDX::Slice(), IDX::Slice(), IDX::Slice(), IDX::Slice({h-radius, h})}, 0);

    std::vector<Tensor> indices_keypoints;
    if(top_k > 0){
        std::tuple<Tensor, Tensor> topk = torch::topk(nms_scores.view({b, -1}), top_k);
        Tensor indices = std::get<1>(topk);
        for(int i=0; i<b; i++){
            indices_keypoints.push_back(indices.select(0, i));
        }
    } else{
        Tensor masks;
        if(scores_th > 0){
            masks = nms_scores > scores_th;
            if(torch::equal(masks.sum(), torch::tensor({0}))){
                auto th = scores_nograd.reshape({b, -1}).mean(1);
                masks = nms_scores > th.reshape({b, 1, 1, 1});
            }
        } else{
            Tensor th = scores_nograd.reshape({b, -1}).mean(1);
            masks = nms_scores > th.reshape({b, 1, 1, 1});
        }
        masks = masks.reshape({b, -1});
        Tensor scores_view = scores_nograd.reshape({b, -1});
        for(int i=0; i<b; i++){
            Tensor mask = masks.select(0, i);
            Tensor scores = scores_view.select(0, i);
            Tensor indices = mask.nonzero().index({IDX::Slice(), 0});
            if(indices.size(0) > n_limit){
                Tensor kpts_sc = scores.index({indices});
                auto sort_idx = torch::sort(kpts_sc, 0, 1);
                auto sel_idx = std::get<1>(sort_idx).index({IDX::Slice({0, n_limit})});
                indices = indices.index({sel_idx});
            }
            indices_keypoints.push_back(indices);
        }
    }

    std::vector<Tensor> keypoints, scoredispersitys, kptscores;
    if(sub_pixel){
        Tensor patches = _unfold(scores_map);
        hw_grid = hw_grid.to(patches.device());
        for(int i=0; i<b; i++){
            Tensor patch = patches.select(0, i).t();
            Tensor indices_kpt = indices_keypoints[i];
            Tensor patch_scores = patch.index({indices_kpt});
            Tensor score_map = scores_map.select(0, i);
            auto max_tuple = torch::max(patch_scores, 1);
            Tensor max_v = std::get<0>(max_tuple).detach().unsqueeze(1);
            Tensor x_exp = (patch_scores - max_v).div(tdet);
            x_exp = x_exp.exp();
            Tensor xy_residual = torch::matmul(x_exp, hw_grid).div(x_exp.sum(1).unsqueeze(1));
            auto hw_grid_dist2 = torch::norm((hw_grid.unsqueeze(0) - xy_residual.unsqueeze(1)).div(radius), 1, -1);
            hw_grid_dist2 = torch::pow(hw_grid_dist2, 2);
            Tensor scoredispersity = x_exp.mul(hw_grid_dist2).sum(1).div(x_exp.sum(1));
            Tensor keypoints_xy_nms =
                    torch::stack({torch::fmod(indices_kpt,w), torch::div(indices_kpt, w, "floor")}, 1);
            Tensor keypoints_xy = keypoints_xy_nms + xy_residual;
            Tensor fact = keypoints_xy.new_zeros({1, 2});
            fact.select(0, 0) = w - 1;
            fact.select(0, 1) = h - 1;
            keypoints_xy = keypoints_xy.div(fact).mul(2) - 1;
            Tensor kptscore = F::grid_sample(
                    score_map.unsqueeze(0), keypoints_xy.view({1, 1, -1, 2}), F::GridSampleFuncOptions().align_corners(true));
            kptscore = kptscore.index({0, 0, 0, IDX::Slice()});
            keypoints.push_back(keypoints_xy);
            scoredispersitys.push_back(scoredispersity);
            kptscores.push_back(kptscore);
        }
    } else{
        for(int i=0; i<b; i++){
            Tensor indices_kpt = indices_keypoints[i];
            Tensor score_map = scores_map.select(0, i);
            Tensor keypoints_xy_nms =
                    torch::stack({torch::fmod(indices_kpt,w), torch::div(indices_kpt, w, "floor")}, 1);
            Tensor fact = keypoints_xy_nms.new_zeros({1, 2});
            fact.select(0, 0) = w - 1;
            fact.select(0, 1) = h - 1;
            Tensor keypoints_xy = keypoints_xy_nms.div(fact).mul(2) - 1;
            Tensor kptscore = F::grid_sample(
                    score_map.unsqueeze(0), keypoints_xy.view({1, 1, -1, 2}), F::GridSampleFuncOptions().align_corners(true));
            keypoints.push_back(keypoints_xy);
            scoredispersitys.push_back(keypoints_xy.new_empty(keypoints_xy.sizes()));
            kptscores.push_back(kptscore);
        }
    }

    std::vector<std::vector<Tensor>> result =
            {keypoints, kptscores, scoredispersitys};
    return result;
}


std::vector<std::vector<Tensor>> &ORB_SLAM3::DKD::forward(const Tensor &scores_map, const Tensor &descriptor_map, bool sub_pixel) {
    std::vector<std::vector<Tensor>> res = detect_keypoints(scores_map, sub_pixel);
    std::vector<Tensor> descriptors = sample_descriptor(descriptor_map, res[0], sub_pixel);
    res.insert(res.begin()+1, descriptors);
    return res;
}


Tensor & ORB_SLAM3::DKD::max_pool(const Tensor &x, int r) {
    Tensor res = F::max_pool2d(x,
F::MaxPool2dFuncOptions(r*2+1).stride(1).padding(r));
    return res;
}


Tensor & ORB_SLAM3::DKD::simple_nms(const Tensor &scores, int nms_radius) {
    assert(nms_radius >= 0);
    Tensor zeros = torch::zeros_like(scores);
    Tensor max_mask = (scores == max_pool(scores, nms_radius));
    for(int i=0; i<2; i++){
        Tensor supp_mask = max_pool(max_mask.toType(torch::kFloat), nms_radius);
        Tensor supp_scores = torch::where(supp_mask, zeros, scores);
        Tensor new_max_mask = (supp_scores == max_pool(supp_scores, nms_radius));
        max_mask = max_mask | (new_max_mask & (~supp_mask));
    }
    Tensor res = torch::where(max_mask, scores, zeros);
    return res;
}


std::vector<Tensor> & ORB_SLAM3::DKD::sample_descriptor(const Tensor &descriptor_map, const std::vector<Tensor> &kpts, bool bilinear_interp) {
    auto batch_size = descriptor_map.size(0);
    auto channel = descriptor_map.size(1);
    auto height = descriptor_map.size(2);
    auto width = descriptor_map.size(3);
    std::vector<Tensor> descriptors;
    for(int i=0; i<batch_size; i++){
        Tensor descriptors_;
        Tensor kptsi = kpts[i];
        if(bilinear_interp)
            descriptors_ = F::grid_sample(descriptor_map[i].unsqueeze(0),
           kptsi.view({1, 1, -1, 2}), F::GridSampleFuncOptions().align_corners(true));
        else{
            Tensor s = kptsi.new_zeros({1, 2});
            s.select(0, 0) = width - 1;
            s.select(0, 0) = height - 1;
            kptsi = (kptsi + 1) / 2 * s;
            kptsi = kptsi.toType(torch::kLong);
            Tensor id2 = kptsi.index({IDX::Slice(), 1});
            Tensor id3 = kptsi.index({IDX::Slice(), 0});
            descriptors_ = descriptor_map.index({i, IDX::Slice(), id2, id3});
        }
        descriptors_ = F::normalize(descriptors_, F::NormalizeFuncOptions().p(2).dim(0));
        descriptors.push_back(descriptors_);
    }
    return descriptors;
}