//
// Created by zhou on 22-4-28.
//

// #include "alike_onnx.h"
#include <torch/torch.h>

namespace IDX = torch::indexing;

int main(int argc, char **argv){
//    ORB_SLAM3::Alike* Alike = new ORB_SLAM3::Alike("/home/zhou/SLAMCODE/ORB_SLAM3/models/alike_n_1000.onnx");
//    cv::Mat img = cv::imread("./test/1.png");
//    (*Alike)(img);
    torch::Tensor t = torch::rand({3, 5});
    auto t1 = t.index({IDX::Slice(), IDX::Slice(1, 3)});
    std::cout<<t<<std::endl;
    std::cout<<t1<<std::endl;
    return 0;
}
