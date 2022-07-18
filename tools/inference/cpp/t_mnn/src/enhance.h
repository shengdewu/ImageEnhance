#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>



class ImgEnhance{
    public:
        ImgEnhance(const std::string mnn_path, size_t num_threads=4);

        ~ImgEnhance();

        cv::Mat run(const cv::Mat &img_bgr_normalized, size_t ref_size=256);

    private:
        void create_mnn_env();

        std::vector<float> fetch_cure_param(MNN::Tensor* tensor);

        void print_input_info(std::string flag);

        void transfor_data(const cv::Mat &mat);

        MNN::Session *create_session();

    private:
        std::string _mnn_path;
        size_t _num_threads;
        std::shared_ptr<MNN::Interpreter> _mnn_interpreter;
        std::shared_ptr<MNN::CV::ImageProcess> _pretreat;
        MNN::Session *_mnn_session = nullptr;

    private:
        static const std::vector<std::string> _OUTPUT_NAMES;
        float _mean_vals[3]   = {0.0f, 0.0f, 0.0f};
        float _normal_vals[3] = {1.0f/255.0, 1.0f/255.0, 1.0f/255.0};
};
