#include <mmintrin.h> //mmx
#include <xmmintrin.h> //sse
#include <emmintrin.h> //sse2
#include <pmmintrin.h> //sse3

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <numeric>
#include <time.h>
#include <chrono>
#include <iomanip>
#include <iostream>

#include <opencv2/core/simd_intrinsics.hpp>
#include "common/common.h"


void test_opencv_simd(){

    std::string img_path = "/mnt/sdb/data.set/xintu.data/转档测评/20210510转档评测_tif/儿童/01浅色实景/儿童SJ (41)/16108236268247323435.tif";
    cv::Mat gray_img = cv::imread(img_path, cv::ImreadModes::IMREAD_GRAYSCALE);

    auto binary_time = std::chrono::system_clock::now();
    cv::imwrite("g.jpg", gray_img);
    uchar threshold = 120;
    size_t total_cnt = 0;

    cv::Mat binary_img = cv::Mat::zeros(gray_img.rows, gray_img.cols, gray_img.type());

    // omp_set_num_threads(1);
    // #pragma omp parallel
    {
        // #pragma omp for
        for(size_t row=0; row<gray_img.rows; row++){
            u_int8_t* gray_ptr = gray_img.ptr<u_int8_t>(row);
            u_int8_t* binary_ptr = binary_img.ptr<u_int8_t>(row);
            for(size_t col=0; col<gray_img.cols; col++){
                binary_ptr[col] = gray_ptr[col] > threshold ? 255 : 0;
                total_cnt += 1;
            }
        }
    }

    spend_time("binary_time", binary_time);
    std::cout << "total_cnt: " << total_cnt << std::endl;
    cv::imwrite("binary_img.jpg", binary_img);

    auto binary_simd_time = std::chrono::system_clock::now();

    cv::Mat binary_simd_img = cv::Mat::zeros(gray_img.rows, gray_img.cols, gray_img.type());
    
    size_t v_step = cv::v_uint8::nlanes;
    std::cout << "v_step " << v_step << std::endl;

    cv::v_uint8 v_threshold = cv::vx_setall(threshold);
    cv::v_uint8 v_255 = cv::vx_setall(uchar(255));
    cv::v_uint8 v_0 = cv::vx_setall(uchar(0));
    // #pragma omp parallel
    {
        // #pragma omp for
        for(size_t row=0; row<gray_img.rows; row++){
            u_int8_t* gray_ptr = gray_img.ptr<u_int8_t>(row);
            u_int8_t* binary_ptr = binary_simd_img.ptr<u_int8_t>(row);

            size_t col=0;
            for(; col<gray_img.cols-v_step; col+=v_step){
                cv::v_uint8 gray = cv::vx_load(gray_ptr+col);
                auto condition = gray > v_threshold;
                cv::v_uint8 binary = cv::v_select(condition, v_255, v_0);
                cv::vx_store(binary_ptr+col, binary);
                total_cnt += 1;
            }
            
            // for(; col<gray_img.cols; col++){
            //     binary_ptr[col] = gray_ptr[col] > threshold ? 255 : 0;
            // }

        }    
    }
    spend_time("binary_simd_time", binary_simd_time);
    std::cout << "total_cnt: " << total_cnt << std::endl;
    cv::imwrite("binary_simd_img.jpg", binary_simd_img);    

    cv::Mat bgr_img_float;
    gray_img.convertTo(bgr_img_float, CV_32FC1, 1.0/255.0);
    std::cout << bgr_img_float.size().width << ", " << bgr_img_float.size().height << ", "  << bgr_img_float.channels() << ", " <<bgr_img_float.step  << ", " <<bgr_img_float.step[1] << std::endl;
    
    cv::Mat mul_img = cv::Mat::zeros(bgr_img_float.rows, bgr_img_float.cols, bgr_img_float.type());

    auto mul_time = std::chrono::system_clock::now();
    total_cnt = 0;
    // omp_set_num_threads(1);
    // #pragma omp parallel
    {
        // #pragma omp for
        for(size_t row=0; row<bgr_img_float.rows; row++){
            float* gray_ptr = bgr_img_float.ptr<float>(row);
            float* out_ptr = mul_img.ptr<float>(row);
            for(size_t col=0; col<bgr_img_float.cols; col++){
                out_ptr[col] = gray_ptr[col] + 0.4;
                total_cnt += 1;
            }
        }
    }

    spend_time("mul_time", mul_time);
    std::cout << "total_cnt: " << total_cnt << std::endl;

    cv::Mat mul_img_255 = (mul_img*255);
    cv::Mat mul_img_8uc1;
    mul_img_255.convertTo(mul_img_8uc1, CV_8UC1);
    cv::imwrite("mul_img_8uc1.jpg", mul_img_8uc1);  


    cv::Mat mul_simd_img = cv::Mat::zeros(bgr_img_float.rows, bgr_img_float.cols, bgr_img_float.type());

    auto mul_simd_time = std::chrono::system_clock::now();
    size_t v_fstep = cv::v_float32::nlanes;
    // cv::v_float32 v_05 = cv::vx_setall(float(0.5));
    // cv::v_float32 v_04 = cv::vx_setall(float(0.4));
    std::cout << "v_fstep: " << v_fstep << std::endl;
    total_cnt = 0;
    
    // omp_set_num_threads(1);
    // #pragma omp parallel
    {
        // #pragma omp for
        for(size_t row=0; row<bgr_img_float.rows; row++){
            float* gray_ptr = bgr_img_float.ptr<float>(row);
            float* out_ptr = mul_simd_img.ptr<float>(row);
            
            __m128 v_05 = _mm_set_ps(0.5, 0.5, 0.5, 0.5);
            __m128 v_04 = _mm_set_ps(0.4, 0.4, 0.4, 0.4);

            for(size_t col=0; col<bgr_img_float.cols-4; col+=4){
                __m128 gp = _mm_set_ps(gray_ptr[0], gray_ptr[1], gray_ptr[2], gray_ptr[3]);
                // __m128 r1 = _mm_mul_ps(gp, v_04);
                __m128 r2 = _mm_add_ps(gp, v_04);
                _mm_storeu_ps(out_ptr, r2);

                total_cnt += 1;
                gray_ptr += 4;
                out_ptr += 4;
            }
        }
    }

    spend_time("mul_simd_time", mul_simd_time);
    std::cout << "total_cnt: " << total_cnt << std::endl;

    cv::Mat mul_simd_img_255 = (mul_simd_img*255);
    cv::Mat mul_simd_img_8uc1;
    mul_simd_img_255.convertTo(mul_simd_img_8uc1, CV_8UC1);
    cv::imwrite("mul_simd_img_8uc1.jpg", mul_simd_img_8uc1);  


    // cv::Mat bgr_img = cv::imread(img_path, cv::ImreadModes::IMREAD_COLOR);
    
    // cv::Mat smid_bgr_img_float, bgr_img_float;
    // bgr_img.convertTo(smid_bgr_img_float, CV_32FC3, 1.0/255.0);
    // bgr_img.convertTo(bgr_img_float, CV_32FC3, 1.0/255.0);

    // auto simd_opt_time = std::chrono::system_clock::now();
    // size_t v_fstep = cv::v_float32::nlanes;
    // std::cout << "v_fstep " << v_fstep << std::endl;
    // cv::v_float32 v_05 = cv::vx_setall(float(0.5));
    // cv::v_float32 v_1 = cv::vx_setall(float(1.0));
    // for(size_t nrow=0; nrow < bgr_img.rows; nrow++){
            
    //     float* img_fptr = smid_bgr_img_float.ptr<float>(nrow);

    //     for(size_t ncol=0; ncol < bgr_img.cols-v_fstep; ncol+=v_fstep){
            
    //         size_t index = 0;
    //         cv::v_float32 v_img_b(*(img_fptr+index), *(img_fptr+1*3+index), *(img_fptr+2*3+index), *(img_fptr+3*3+index), *(img_fptr+4*3+index), *(img_fptr+5*3+index), *(img_fptr+6*3+index), *(img_fptr+7*3+index));
    //         index = 1;
    //         cv::v_float32 v_img_g(*(img_fptr+index), *(img_fptr+1*3+index), *(img_fptr+2*3+index), *(img_fptr+3*3+index), *(img_fptr+4*3+index), *(img_fptr+5*3+index), *(img_fptr+6*3+index), *(img_fptr+7*3+index));
    //         index = 2;
    //         cv::v_float32 v_img_r(*(img_fptr+index), *(img_fptr+1*3+index), *(img_fptr+2*3+index), *(img_fptr+3*3+index), *(img_fptr+4*3+index), *(img_fptr+5*3+index), *(img_fptr+6*3+index), *(img_fptr+7*3+index));

    //         // cv::v_float32 v_img_b, v_img_g, v_img_r;
    //         // cv::v_load_deinterleave(img_fptr+ncol*3, v_img_b, v_img_g, v_img_r);

    //         cv::v_float32 v_et_img_b = v_1 - v_img_b * v_05;
    //         cv::v_float32 v_et_img_g = v_1 - v_img_g * v_05;
    //         cv::v_float32 v_et_img_r = v_1 - v_img_r * v_05;  

    //         // cv::v_store_interleave(img_fptr+ncol*3, v_et_img_b, v_et_img_g, v_et_img_r); 

    //         index = 0;
    //         *(img_fptr+index) = v_et_img_b.get0();
    //         *(img_fptr+1*3+index) = v_et_img_b.get0();
    //         *(img_fptr+2*3+index) = v_et_img_b.get0();
    //         *(img_fptr+3*3+index) = v_et_img_b.get0();
    //         *(img_fptr+4*3+index) = v_et_img_b.get0();
    //         *(img_fptr+5*3+index) = v_et_img_b.get0();
    //         *(img_fptr+6*3+index) = v_et_img_b.get0();
    //         *(img_fptr+7*3+index) = v_et_img_b.get0();
    //         index = 1;
    //         *(img_fptr+index) = v_et_img_g.get0();
    //         *(img_fptr+1*3+index) = v_et_img_g.get0();
    //         *(img_fptr+2*3+index) = v_et_img_g.get0();
    //         *(img_fptr+3*3+index) = v_et_img_g.get0();
    //         *(img_fptr+4*3+index) = v_et_img_g.get0();
    //         *(img_fptr+5*3+index) = v_et_img_g.get0();
    //         *(img_fptr+6*3+index) = v_et_img_g.get0();
    //         *(img_fptr+7*3+index) = v_et_img_g.get0();            
    //         index = 2;
    //         *(img_fptr+index) = v_et_img_r.get0();
    //         *(img_fptr+1*3+index) = v_et_img_r.get0();
    //         *(img_fptr+2*3+index) = v_et_img_r.get0();
    //         *(img_fptr+3*3+index) = v_et_img_r.get0();
    //         *(img_fptr+4*3+index) = v_et_img_r.get0();
    //         *(img_fptr+5*3+index) = v_et_img_r.get0();
    //         *(img_fptr+6*3+index) = v_et_img_r.get0();
    //         *(img_fptr+7*3+index) = v_et_img_r.get0();

    //         img_fptr += v_fstep * 3;                   
    //     } 
        // for(; ncol < cols; ncol+=step){
            
        //     cv::v_float32 v_img_b, v_img_g, v_img_r;
        //     cv::v_load_deinterleave(img_fptr+ncol*3, v_img_b, v_img_g, v_img_r);

        //     cv::v_float32 v_scale_b = cv::vx_load(b_scale_ptr+ncol);
        //     cv::v_float32 v_scale_g = cv::vx_load(g_scale_ptr+ncol);
        //     cv::v_float32 v_scale_r = cv::vx_load(r_scale_ptr+ncol);              

        //     // for(int i=0; i<cure_steps-1; i++){
        //     //     cv::v_float32 v_i = cv::vx_setall(float(i));
        //     //     cv::v_float32 v_b_slope = cv::vx_setall(float(b_slope[i]));
        //     //     cv::v_float32 v_g_slope = cv::vx_setall(float(g_slope[i]));
        //     //     cv::v_float32 v_r_slope = cv::vx_setall(float(r_slope[i]));

        //     //     v_scale_b = v_scale_b + v_b_slope * (v_img_b * v_cure_steps - v_i);
        //     //     v_scale_g = v_scale_g + v_g_slope * (v_img_g * v_cure_steps - v_i);
        //     //     v_scale_r = v_scale_r + v_r_slope * (v_img_r * v_cure_steps - v_i);  

        //     // }

        //     cv::v_float32 v_et_img_b = v_img_b * v_scale_b * v_255;
        //     cv::v_float32 v_et_img_g = v_img_g * v_scale_g * v_255;
        //     cv::v_float32 v_et_img_r = v_img_r * v_scale_r * v_255;  

        //     cv::v_store_interleave(enhance_img_fptr+ncol*3, v_et_img_b, v_et_img_g, v_et_img_r);                       
        // }     
    // }
    
    // spend_time("simd_opt_time", simd_opt_time);

    // cv::Mat tmp = smid_bgr_img_float*255;
    // cv::Mat tmp_255;
    // tmp.convertTo(tmp_255, CV_8UC3);
    // cv::imwrite("opt1.jpg", tmp_255);

    // auto opt_time = std::chrono::system_clock::now();
    // for(size_t nrow=0; nrow < bgr_img.rows; nrow++){
            
    //     float* img_fptr = bgr_img_float.ptr<float>(nrow);

    //     for(size_t ncol=0; ncol < bgr_img.cols; ncol+=1){
            
    //         img_fptr[0] = 1.0 - img_fptr[0] * 0.5;
    //         img_fptr[1] = 1.0 - img_fptr[1] * 0.5;
    //         img_fptr[2] = 1.0 - img_fptr[2] * 0.5;   

    //         // (img_fptr+3)[0] = 1.0 - (img_fptr+3)[0] * 0.5;
    //         // (img_fptr+3)[1] = 1.0 - (img_fptr+3)[1] * 0.5;
    //         // (img_fptr+3)[2] = 1.0 - (img_fptr+3)[2] * 0.5;              

    //         // (img_fptr+3*2)[0] = 1.0 - (img_fptr+3*2)[0] * 0.5;
    //         // (img_fptr+3*2)[1] = 1.0 - (img_fptr+3*2)[1] * 0.5;
    //         // (img_fptr+3*2)[2] = 1.0 - (img_fptr+3*2)[2] * 0.5;  

    //         // (img_fptr+3*3)[0] = 1.0 - (img_fptr+3*3)[0] * 0.5;
    //         // (img_fptr+3*3)[1] = 1.0 - (img_fptr+3*3)[1] * 0.5;
    //         // (img_fptr+3*3)[2] = 1.0 - (img_fptr+3*3)[2] * 0.5;  

    //         img_fptr += 3;                    
    //     } 
    // }    
    // spend_time("opt_time", opt_time);

    // cv::Mat tmp2 = bgr_img_float*255;
    // cv::Mat tmp2_255;
    // tmp2.convertTo(tmp2_255, CV_8UC3);
    // cv::imwrite("opt2.jpg", tmp2_255);
}

