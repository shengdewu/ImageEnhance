#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

struct OnnxOutInfo{

};

class ImgEnhance{
    public:
        ImgEnhance(const std::string model_path, size_t ref_size=512, bool map_point_wise=false);

        ~ImgEnhance();

        cv::Mat run(const cv::Mat &img_bgr_normalized);

    private:                      

        void create_onnx_env();

    private:
        std::string _model_path;
        size_t _ref_size;
        bool _map_point_wise;
        Ort::Env *_env;
        Ort::Session *_session;
        std::vector<int64_t> _input_dims;
        std::vector<const char*> _output_names;
};
