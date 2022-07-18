#include "apply_spline.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>


cv::Mat apply_spline(const cv::Mat &img_bgr,
                     const std::vector<float> &r_tensor_values,
                     const std::vector<float> &g_tensor_values,
                     const std::vector<float> &b_tensor_values){

    cv::Mat b_scale = cv::Mat(img_bgr.rows, img_bgr.cols, CV_32F, b_tensor_values[0]);
    cv::Mat g_scale = cv::Mat(img_bgr.rows, img_bgr.cols, CV_32F, g_tensor_values[0]);
    cv::Mat r_scale = cv::Mat(img_bgr.rows, img_bgr.cols, CV_32F, r_tensor_values[0]);

    auto cure_steps = r_tensor_values.size() - 1;

    std::vector<float> b_slope = std::vector<float>(cure_steps);
    std::vector<float> g_slope = std::vector<float>(cure_steps);
    std::vector<float> r_slope = std::vector<float>(cure_steps);

    for(int i=0; i<cure_steps-1; i++){
        b_slope[i] = b_tensor_values[i+1] - b_tensor_values[i];
        g_slope[i] = g_tensor_values[i+1] - g_tensor_values[i];
        r_slope[i] = r_tensor_values[i+1] - r_tensor_values[i];
    }

    cv::Mat enhance_img = cv::Mat::zeros(img_bgr.rows, img_bgr.cols, CV_32FC3);

    #pragma omp parallel
    {
        #pragma omp for
        for(size_t nrow=0; nrow < img_bgr.rows; nrow++){

            const float* img_fptr = img_bgr.ptr<float>(nrow);
            float* b_scale_ptr = b_scale.ptr<float>(nrow);
            float* g_scale_ptr = g_scale.ptr<float>(nrow);
            float* r_scale_ptr = r_scale.ptr<float>(nrow);

            float*  enhance_img_fptr = enhance_img.ptr<float>(nrow);

            for(size_t ncol = 0; ncol < img_bgr.cols; ncol++){

                float b_img_cure = img_fptr[0] * static_cast<float>(cure_steps);
                float g_img_cure = img_fptr[1] * static_cast<float>(cure_steps);
                float r_img_cure = img_fptr[2] * static_cast<float>(cure_steps);


                for(int i=0; i<cure_steps-1; ++i){
                    b_scale_ptr[ncol] = b_scale_ptr[ncol] + b_slope[i] * (b_img_cure - static_cast<float>(i));
                    g_scale_ptr[ncol] = g_scale_ptr[ncol] + g_slope[i] * (g_img_cure - static_cast<float>(i));
                    r_scale_ptr[ncol] = r_scale_ptr[ncol] + r_slope[i] * (r_img_cure - static_cast<float>(i));

                }

                enhance_img_fptr[0] = img_fptr[0] * b_scale_ptr[ncol] * 2550.0;
                enhance_img_fptr[1] = img_fptr[1] * g_scale_ptr[ncol] * 2550.0;
                enhance_img_fptr[2] = img_fptr[2] * r_scale_ptr[ncol] * 2550.0;

                img_fptr += 3;
                enhance_img_fptr += 3;

            }
        }
    }

    return enhance_img;
}


cv::Mat apply_spline(const cv::Mat &img_bgr_map, const cv::Mat &img_bgr,
                     const std::vector<float> &r_tensor_values,
                     const std::vector<float> &g_tensor_values,
                     const std::vector<float> &b_tensor_values){

    cv::Mat b_scale_map = cv::Mat(img_bgr_map.rows, img_bgr_map.cols, CV_32F, b_tensor_values[0]);
    cv::Mat g_scale_map = cv::Mat(img_bgr_map.rows, img_bgr_map.cols, CV_32F, g_tensor_values[0]);
    cv::Mat r_scale_map = cv::Mat(img_bgr_map.rows, img_bgr_map.cols, CV_32F, r_tensor_values[0]);

    auto cure_steps = r_tensor_values.size() - 1;

    std::vector<float> b_slope = std::vector<float>(cure_steps);
    std::vector<float> g_slope = std::vector<float>(cure_steps);
    std::vector<float> r_slope = std::vector<float>(cure_steps);

    for(int i=0; i<cure_steps-1; i++){
        b_slope[i] = b_tensor_values[i+1] - b_tensor_values[i];
        g_slope[i] = g_tensor_values[i+1] - g_tensor_values[i];
        r_slope[i] = r_tensor_values[i+1] - r_tensor_values[i];
    }

    // #pragma omp parallel
    {
        // #pragma omp for
        for(size_t nrow=0; nrow < img_bgr_map.rows; nrow++){

            const float* img_fptr = img_bgr_map.ptr<float>(nrow);
            float* b_scale_ptr = b_scale_map.ptr<float>(nrow);
            float* g_scale_ptr = g_scale_map.ptr<float>(nrow);
            float* r_scale_ptr = r_scale_map.ptr<float>(nrow);

            for(size_t ncol = 0; ncol < img_bgr_map.cols; ncol++){

                float b_img_cure = img_fptr[0] * static_cast<float>(cure_steps);
                float g_img_cure = img_fptr[1] * static_cast<float>(cure_steps);
                float r_img_cure = img_fptr[2] * static_cast<float>(cure_steps);


                for(int i=0; i<cure_steps-1; ++i){
                    b_scale_ptr[ncol] = b_scale_ptr[ncol] + b_slope[i] * (b_img_cure - static_cast<float>(i));
                    g_scale_ptr[ncol] = g_scale_ptr[ncol] + g_slope[i] * (g_img_cure - static_cast<float>(i));
                    r_scale_ptr[ncol] = r_scale_ptr[ncol] + r_slope[i] * (r_img_cure - static_cast<float>(i));

                }

                img_fptr += 3;
            }
        }
    }

    // 1
    std::vector<cv::Mat> scale_map{b_scale_map, g_scale_map, r_scale_map};
    cv::Mat bgr_scale_map;
    cv::merge(scale_map, bgr_scale_map);

    cv::Mat bgr_scale;
    cv::resize(bgr_scale_map, bgr_scale, cv::Size(img_bgr.cols, img_bgr.rows));

    return img_bgr.mul(bgr_scale)*255.0;

    // 2
    // cv::Mat b_scale, g_scale, r_scale;
    // cv::resize(b_scale_map, b_scale, cv::Size(img_bgr.cols, img_bgr.rows));
    // cv::resize(g_scale_map, g_scale, cv::Size(img_bgr.cols, img_bgr.rows));
    // cv::resize(r_scale_map, r_scale, cv::Size(img_bgr.cols, img_bgr.rows));

    // cv::Mat enhance_img = cv::Mat::zeros(img_bgr.rows, img_bgr.cols, CV_32FC3);
    // #pragma omp parallel
    // {
    //     #pragma omp for
    //     for(size_t nrow=0; nrow < img_bgr.rows; nrow++){

    //         const float* img_fptr = img_bgr.ptr<float>(nrow);
    //         float* b_scale_ptr = b_scale.ptr<float>(nrow);
    //         float* g_scale_ptr = g_scale.ptr<float>(nrow);
    //         float* r_scale_ptr = r_scale.ptr<float>(nrow);
    //         float*  enhance_img_fptr = enhance_img.ptr<float>(nrow);

    //         for(size_t ncol = 0; ncol < img_bgr.cols; ncol++){

    //             enhance_img_fptr[0] = img_fptr[0] * b_scale_ptr[ncol] * 255.0;
    //             enhance_img_fptr[1] = img_fptr[1] * g_scale_ptr[ncol] * 255.0;
    //             enhance_img_fptr[2] = img_fptr[2] * r_scale_ptr[ncol] * 255.0;

    //             img_fptr += 3;
    //             enhance_img_fptr += 3;

    //         }
    //     }
    // }

    // return enhance_img;
}

cv::Mat apply_spline_luma(const cv::Mat &img_bgr, const cv::Mat &img_bgr_map, const cv::Mat &img_gray_map, const std::vector<float> &r_tensor_values){
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

    // cv::Mat r_scale;
    // cv::resize(r_scale_map, r_scale, cv::Size(img_bgr.cols, img_bgr.rows));

    // cv::Mat enhance_img = cv::Mat::zeros(img_bgr.rows, img_bgr.cols, CV_32FC3);
    // #pragma omp parallel
    // {
    //     #pragma omp for
    //     for(size_t nrow=0; nrow < img_bgr.rows; nrow++){

    //         const float* img_fptr = img_bgr.ptr<float>(nrow);
    //         float* r_scale_ptr = r_scale.ptr<float>(nrow);
    //         float*  enhance_img_fptr = enhance_img.ptr<float>(nrow);

    //         for(size_t ncol = 0; ncol < img_bgr.cols; ncol++){

    //             enhance_img_fptr[0] = img_fptr[0] * r_scale_ptr[ncol] * 255.0;
    //             enhance_img_fptr[1] = img_fptr[1] * r_scale_ptr[ncol] * 255.0;
    //             enhance_img_fptr[2] = img_fptr[2] * r_scale_ptr[ncol] * 255.0;

    //             img_fptr += 3;
    //             enhance_img_fptr += 3;

    //         }
    //     }
    // }

    // return enhance_img;

}

// cv::Mat ImgEnhance::apply_spline(const cv::Mat &img_single, const std::vector<float> &cure_single){

//     const int slope_len = cure_single.size() - 1;
//     auto cure_steps = cure_single.size() - 1;

//     std::vector<float> slope = std::vector<float>(slope_len);

//     for(int i=0; i<cure_steps; i++){
//         slope[i] = cure_single[i+1] - cure_single[i];
//     }

//     cv::Mat scale = cv::Mat(img_single.rows, img_single.cols, CV_32F, cure_single[0]);
//     cv::Mat cure_img = img_single * cure_steps;

//     for(int i=0; i<slope.size()-1; i++){
//         scale = scale + slope[i] * (cure_img - i);
//     }

//     return img_single.mul(scale);
// }