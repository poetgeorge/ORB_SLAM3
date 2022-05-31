//
// Created by zhou on 22-5-3.
//

#ifndef ORB_SLAM3_BASEEXTRACTOR_H
#define ORB_SLAM3_BASEEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM3
{
class BaseExtractor{
public:
    BaseExtractor() = delete; // 禁止使用默认构造函数

    BaseExtractor(int _nfeatures, float _scaleFactor, int _nlevels,
                  int _iniThFAST, int _minThFAST);

    // 编译器自动生成函数体（由于默认构造函数是自定义的，若无default关键字，则析构函数也需要自己定义）
    virtual ~BaseExtractor() = default;

    virtual int operator()( cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::KeyPoint>& _keypoints,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea);

    int inline GetLevels(){
        return nlevels;}

    float inline GetScaleFactor(){
        return scaleFactor;}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;

protected:
    void ComputePyramid(cv::Mat image);

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
};
}

#endif //ORB_SLAM3_BASEEXTRACTOR_H
