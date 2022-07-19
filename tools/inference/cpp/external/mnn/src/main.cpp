#include <iostream>
#include <fstream>
#include <assert.h>
#include <vector>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "curve_mnn.hpp"

void bat_test(std::string file_list_txt, std::string root_path, std::string out_root){
    std::ifstream ifs;
    ifs.open(file_list_txt, std::ios::in);
    if(!ifs.is_open()){
        std::cout << file_list_txt << std::endl;
        return;
    }

    std::string mnn_path = "/mnt/sda1/wokspace/ImageEnhance/tools/inference/cpp/model/luma_11xy.mnn";
    CurveMNN auto_exposure(mnn_path);

    std::string split("#");
    std::string ctrl_n("\n");
    std::string line;
    while(std::getline(ifs, line)){
        std::cout << line << std::endl;
        size_t pos = line.find(split);
        std::string sub_path(line.substr(pos+1, line.size()-ctrl_n.size()));
        // std::cout << sub_path << std::endl;
        std::string out_path = out_root + "/" + sub_path;
        std::string cmd("mkdir -p " + std::string("\"") + out_path + std::string("\""));
        // std::cout << cmd << std::endl;
        int ret = system(cmd.c_str());
        if(ret){
            std::cout << strerror(errno) << std::endl;
        }
        std::string img_path = root_path + "/" + sub_path;
        struct dirent* ptr;
        DIR *dir = opendir(img_path.c_str());
        while((ptr=readdir(dir)) != NULL){
            if(strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0){
                continue;
            }

            std::string name = ptr->d_name;

            size_t pos = name.rfind('.');
            std::string out_name = out_path + "/" + name.substr(0, pos)+ ".jpg";
            std::cout << out_name << std::endl;
            std::ifstream f(out_name.c_str());

            if (f.good()){
                std::cout << out_name << " has found "<< std::endl;
                continue;
            }

            std::string img_name = img_path + "/" + name ;

            cv::Mat img_bgr = cv::imread(img_name, cv::ImreadModes::IMREAD_COLOR);
            auto enhance_time = std::chrono::system_clock::now();
            cv::Mat enhance_img = auto_exposure.exposure(img_bgr);

            cv::Mat concat;
            cv::hconcat(img_bgr, enhance_img, concat);

            cv::imwrite(out_name, concat);
        }
    }
}


int main(int argc, char** argv){

    // bat_test("/mnt/sda1/wokspace/ImageEnhance/test_miscellaneous/test_dirs/dir_error.txt", "/mnt/sdb/data.set/xintu.data/转档测评/20210510转档评测_tif_3000x2000", "/mnt/sda1/enhance.test/luma_mnn");
    std::string mnn_path = "/mnt/sda1/wokspace/ImageEnhance/tools/inference/cpp/model/luma_11xy.mnn";

    CurveMNN auto_exposure(mnn_path);
    
    std::vector<std::string> img_path;
    img_path.emplace_back("/mnt/sdb/data.set/xintu.data/转档测评/20210510转档评测_tif_3000x2000/婚纱/鲜艳纯底/婚纱CS (50)/IMG_6929.tif");
    img_path.emplace_back("/mnt/sda1/wokspace/ImageEnhance/test_miscellaneous/debug.raw.jpg");
     img_path.emplace_back("/mnt/sdb/data.set/xintu.data/转档测评/20210510转档评测_tif/儿童/01浅色实景/儿童SJ (31)/161082062385739907252.tif");
     img_path.emplace_back("/mnt/sdb/data.set/xintu.data/转档测评/20210510转档评测_tif/儿童/01浅色实景/儿童SJ (31)/16108206392879900290.tif");
     img_path.emplace_back("/mnt/sdb/data.set/xintu.data/转档测评/20210510转档评测_tif_3000x2000/婚纱/鲜艳纯底/婚纱CS (50)/IMG_6929.tif");
    
    for(auto & i : img_path){

        cv::Mat img_bgr = cv::imread(i, cv::ImreadModes::IMREAD_COLOR);
        cv::Mat img_rgb;
        cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

        cv::Mat enhance_rgb = auto_exposure.exposure(img_rgb);

        cv::Mat enhance_bgr;
        cv::cvtColor(enhance_rgb, enhance_bgr, cv::COLOR_RGB2BGR);

        cv::Mat concat;
        cv::hconcat(img_bgr, enhance_bgr, concat);
        size_t spos = i.rfind('/')+1;
        size_t epos = i.find(".tif");
        cv::imwrite(i.substr(spos, epos-spos)+".jpg", concat);
    }

}