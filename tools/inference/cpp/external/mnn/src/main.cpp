#include <iostream>
#include <assert.h>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "auto_exposure.hpp"


int main(int argc, char** argv){

    std::string mnn_path = "/mnt/sda1/wokspace/ImageEnhance/tools/inference/cpp/model/luma_11xy.mnn";

    AutoExposure auto_exposure(mnn_path);
    
    std::vector<std::string> img_path;
    img_path.push_back("/mnt/sdb/data.set/xintu.data/转档测评/20210510转档评测_tif/儿童/01浅色实景/儿童SJ (31)/161082062385739907252.tif");
    img_path.push_back("/mnt/sdb/data.set/xintu.data/转档测评/20210510转档评测_tif/儿童/01浅色实景/儿童SJ (31)/16108206392879900290.tif");
    img_path.push_back("/mnt/sdb/data.set/xintu.data/转档测评/20210510转档评测_tif_3000x2000/婚纱/鲜艳纯底/婚纱CS (50)/IMG_6929.tif");
    
    for(size_t i=0; i < img_path.size(); i++){

        cv::Mat img_bgr = cv::imread(img_path[i], cv::ImreadModes::IMREAD_COLOR);

        cv::Mat enhance_img = auto_exposure.exposure(img_bgr);

        cv::Mat concat;
        cv::hconcat(img_bgr, enhance_img, concat);
        size_t spos = img_path[i].rfind('/')+1;
        size_t epos = img_path[i].find(".tif");
        cv::imwrite(img_path[i].substr(spos, epos-spos)+".jpg", concat);
    }

}