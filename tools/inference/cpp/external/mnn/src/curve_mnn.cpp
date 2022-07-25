//
// Created by ts on 2022/1/21.
//
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.h>
#include "curve_mnn.hpp"


const std::string CurveMNN::L_NAME = "L";

cv::Mat CurveMNN::exposure(const cv::Mat &input, size_t ref_size)
{
    // normalization to [0, 1.0]
    cv::Mat input_normal;
    input.convertTo(input_normal, CV_32FC3, 1.0/255.0);

    cv::Size target_size = scale_long_edge(cv::Size(input_normal.cols, input_normal.rows), ref_size);
	cv::Mat map_img, gray_img;
	cv::resize(input_normal, map_img, target_size, 0, 0, cv::INTER_AREA);
    cv::cvtColor(map_img, gray_img, cv::ColorConversionCodes::COLOR_RGB2GRAY);

    fill_input_data(gray_img);

    ort_->mnn_interpreter->runSession(ort_->mnn_session);
    auto output_tensors = ort_->mnn_interpreter->getSessionOutputAll(ort_->mnn_session);

    std::vector<float> l_value = fetch_cure_param(output_tensors.at(L_NAME));
    if(l_value.empty()){
        return {};
    }

    cv::Mat enhance_img = apply_spline_luma(input_normal, gray_img, l_value);

    auto convert_time = std::chrono::system_clock::now();
    enhance_img.convertTo(enhance_img, CV_8UC3);

    return enhance_img;
}

void CurveMNN::fill_input_data(const cv::Mat &in_img) {
    auto input_tensor = ort_->mnn_interpreter->getSessionInput(ort_->mnn_session, nullptr);
    int input_batch = input_tensor->batch();
    int input_channel = input_tensor->channel();
    int input_height = input_tensor->height();
    int input_width = input_tensor->width();

    int d_w = in_img.size().width;
    int d_h = in_img.size().height;
    int d_c = in_img.channels();

    std::vector<int> target_dims{1, d_c, d_h, d_w};
    std::vector<int> input_dims{input_batch, input_channel, input_height, input_width};
    if(input_dims != target_dims){
        ort_->mnn_interpreter->resizeTensor(input_tensor, target_dims);
        ort_->mnn_interpreter->resizeSession(ort_->mnn_session);
    }

    input_tensor = ort_->mnn_interpreter->getSessionInput(ort_->mnn_session, nullptr);
    const MNN::Tensor* in_tensor = new MNN::Tensor(input_tensor, input_tensor->getDimensionType());
    auto img_nchw = cv::dnn::blobFromImage(in_img);
    memccpy(in_tensor->host<float>(), reinterpret_cast<float*>(const_cast<unsigned char*>(img_nchw.data)), 0, 1*d_c*d_h*d_w);
    input_tensor->copyFromHostTensor(in_tensor);
    delete in_tensor;
}

std::vector<float> CurveMNN::fetch_cure_param(MNN::Tensor* tensor){
    MNN::Tensor host_tensor(tensor, tensor->getDimensionType());
    tensor->copyToHostTensor(&host_tensor);

    std::vector<float> cure_param;
    auto output_shape = host_tensor.shape();
    const float *f_data = host_tensor.host<float>();
    for(size_t i=0; i<output_shape.at(1); i++){
        cure_param.push_back(f_data[i]);
    }

    return cure_param;
}


cv::Mat CurveMNN::apply_spline_luma(const cv::Mat &img,
                                    const cv::Mat &gray,
                                    const std::vector<float> &L){

    cv::Mat scale_map = cv::Mat(gray.rows, gray.cols, CV_32F, L[0]);

    auto cure_steps = L.size() - 1;

    std::vector<float> slope = std::vector<float>(cure_steps);

    for(int i=0; i<cure_steps-1; i++){
        slope[i] = L[i+1] - L[i];
    }

    // #pragma omp parallel
    {
        // #pragma omp for
        for(int row=0; row < gray.rows; row++){

            const float* img_fptr = gray.ptr<float>(row);
            float* scale_ptr = scale_map.ptr<float>(row);

            for(size_t col = 0; col < gray.cols; col++){

                float img_cure = (*img_fptr) * static_cast<float>(cure_steps);

                for(int i=0; i<cure_steps-1; ++i){
                    scale_ptr[col] = scale_ptr[col] + slope[i] * (img_cure - static_cast<float>(i));
                }

                img_fptr += 1;
            }
        }
    }

    // 1
    std::vector<cv::Mat> scale_map_vector{scale_map, scale_map, scale_map};
    cv::Mat scale_map_rgb;
    cv::merge(scale_map_vector, scale_map_rgb);

    cv::Mat scale_rgb;
    cv::resize(scale_map_rgb, scale_rgb, cv::Size(img.cols, img.rows));

    return img.mul(scale_rgb)*255.0;
}

cv::Size CurveMNN::scale_long_edge(cv::Size size, size_t ref_size){
    int target_width = size.width;
    int target_height = size.height;
    int max_size = std::max(target_width, target_height);
    float scale = ref_size * 1.0 / max_size;
    if(scale < 1.0){
        if(target_width > target_height){
            target_width = ref_size;
            target_height = target_height * scale;
        }
        else{
            target_width = target_width * scale;
            target_height = ref_size;
        }
    }

    return cv::Size (int(target_width+0.5), int(target_height+0.5));
} 