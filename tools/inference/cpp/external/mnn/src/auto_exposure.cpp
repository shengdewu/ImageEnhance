//
// Created by ts on 2022/1/21.
//
#include <iostream>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "auto_exposure.hpp"

cv::Mat AutoExposure::exposure(const cv::Mat &img_bgr, size_t ref_size)
{
    cv::Mat input_img_bgr_normal, input_img_gray_normal;
    img_bgr.convertTo(input_img_bgr_normal, CV_32FC3, 1.0/255.0);

    cv::Size target_size = scale_longe_edge(cv::Size(input_img_bgr_normal.cols, input_img_bgr_normal.rows), ref_size);
	cv::Mat map_img, gray_img;
	cv::resize(input_img_bgr_normal, map_img, target_size, 0, 0, cv::INTER_AREA);
    cv::cvtColor(map_img, gray_img, cv::ColorConversionCodes::COLOR_BGR2GRAY);
    cv::Mat in_img = cv::dnn::blobFromImage(gray_img, 1.0, gray_img.size(), cv::Scalar(), false);

    // std::cout << "input_img_bgr_normal: "<< input_img_bgr_normal.rows << "," << input_img_bgr_normal.cols << "," << input_img_bgr_normal.channels() << std::endl;
    // std::cout << "map_img: "<< map_img.rows << "," << map_img.cols << "," << map_img.channels() << std::endl;
    // std::cout << "in_img: " << in_img.rows << "," << in_img.cols << "," << in_img.channels() << "," << in_img.dims  << std::endl;
    // std::cout << "gray_img: " << gray_img.rows << "," << gray_img.cols << "," << gray_img.channels() << std::endl;

    MNN::Session* mnn_session = ort_->mnn_interpreter->createSession(ort_->schedule_config);

    auto input_tensor = ort_->mnn_interpreter->getSessionInput(mnn_session, nullptr);
    int input_batch = input_tensor->batch();
    int input_channel = input_tensor->channel();
    int input_height = input_tensor->height();
    int input_width = input_tensor->width();

    int d_w = gray_img.size().width;
    int d_h = gray_img.size().height;
    int d_c = gray_img.channels();

    std::vector<int> target_dims{1, d_c, d_h, d_w};
    std::vector<int> input_dims{input_batch, input_channel, input_height, input_width};
    if(input_dims != target_dims){
        ort_->mnn_interpreter->resizeTensor(input_tensor, target_dims);
        ort_->mnn_interpreter->resizeSession(mnn_session);
    }

    input_tensor = ort_->mnn_interpreter->getSessionInput(mnn_session, nullptr);
    auto nchw_tensor = new MNN::Tensor(input_tensor, input_tensor->getDimensionType());
    memccpy(nchw_tensor->host<float>(), reinterpret_cast<float*>(const_cast<unsigned char*>(in_img.data)), 0, 1*d_c*d_h*d_w);
    input_tensor->copyFromHostTensor(nchw_tensor);
    delete nchw_tensor;

    ort_->mnn_interpreter->runSession(mnn_session);
    auto output_tensors = ort_->mnn_interpreter->getSessionOutputAll(mnn_session);

    std::vector<float> r_tensor_values = fetch_cure_param(output_tensors.at("r"));
    ort_->mnn_interpreter->releaseSession(mnn_session);
    if(r_tensor_values.size() == 0){
        return cv::Mat();
    }

    cv::Mat enhance_img = apply_spline_luma(input_img_bgr_normal, map_img, in_img, r_tensor_values);

    auto convert_time = std::chrono::system_clock::now();
    enhance_img.convertTo(enhance_img, CV_8UC3);

    return enhance_img;
}


std::vector<float> AutoExposure::fetch_cure_param(MNN::Tensor* tensor){
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


cv::Mat AutoExposure::apply_spline_luma(const cv::Mat &img_bgr, 
                                        const cv::Mat &img_bgr_map, 
                                        const cv::Mat &img_gray_map, 
                                        const std::vector<float> &r_tensor_values){

    cv::Mat r_scale_map = cv::Mat(img_bgr_map.rows, img_bgr_map.cols, CV_32F, r_tensor_values[0]);

    auto cure_steps = r_tensor_values.size() - 1;

    std::vector<float> r_slope = std::vector<float>(cure_steps);

    for(int i=0; i<cure_steps-1; i++){
        r_slope[i] = r_tensor_values[i+1] - r_tensor_values[i];
    }

    // #pragma omp parallel
    {
        // #pragma omp for
        for(size_t nrow=0; nrow < img_bgr_map.rows; nrow++){

            const float* img_fptr = img_bgr_map.ptr<float>(nrow);
            float* r_scale_ptr = r_scale_map.ptr<float>(nrow);

            for(size_t ncol = 0; ncol < img_bgr_map.cols; ncol++){

                float b_img_cure = img_fptr[0] * static_cast<float>(cure_steps);
                float g_img_cure = img_fptr[1] * static_cast<float>(cure_steps);
                float r_img_cure = img_fptr[2] * static_cast<float>(cure_steps);


                for(int i=0; i<cure_steps-1; ++i){
                    r_scale_ptr[ncol] = r_scale_ptr[ncol] + r_slope[i] * (r_img_cure - static_cast<float>(i));

                }

                img_fptr += 3;
            }
        }
    }

    // 1
    std::vector<cv::Mat> scale_map{r_scale_map, r_scale_map, r_scale_map};
    cv::Mat bgr_scale_map;
    cv::merge(scale_map, bgr_scale_map);

    cv::Mat bgr_scale;
    cv::resize(bgr_scale_map, bgr_scale, cv::Size(img_bgr.cols, img_bgr.rows));

    return img_bgr.mul(bgr_scale)*255.0;
}

cv::Size AutoExposure::scale_longe_edge(cv::Size size, size_t ref_size){
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