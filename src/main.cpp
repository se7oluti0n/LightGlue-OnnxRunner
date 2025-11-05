/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.08.31
    Description: 
*********************************/

#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include "Configuration.h"
#include "BaseOnnxRunner.h"
#include "LightGlueOnnxRunner.h"
#include "LightGlueDecoupleOnnxRunner.h"

namespace fs = std::filesystem;


std::vector<cv::Mat> ReadImage(std::vector<cv::String> image_filelist , bool grayscale = false)
{
    /*
    Func:
        Read an image from path as RGB or grayscale

    */
    int mode = cv::IMREAD_COLOR;
    if (grayscale)
    {
        mode = grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
    }

    std::vector<cv::Mat> image_matlist;
    for (const auto& file : image_filelist)
    {
        std::cout << "[FILE INFO] : " << file << std::endl;
        cv::Mat image = cv::imread(file , mode);
        if (image.empty())
        {
            throw std::runtime_error("[ERROR] Could not read image at " + file);
        }
        if (!grayscale)
        {
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // BGR -> RGB
        }
        image_matlist.emplace_back(image);
    }

    return image_matlist;
}

std::vector<cv::KeyPoint> UnnormalizeKeypoints(const std::vector<cv::Point2f>& normalized_keypoints, int w, int h) {
  std::vector<cv::KeyPoint> keypoints;

    cv::Size size(w, h);
    cv::Point2f shift(static_cast<float>(w) / 2, static_cast<float>(h) / 2);
    float scale = static_cast<float>((std::max)(w, h)) / 2;

    std::vector<cv::Point2f> normalizedKpts;
    for (const cv::Point2f& kpt : normalized_keypoints) {
        cv::Point2f ret = kpt * scale + shift;
        cv::KeyPoint kp;
        kp.pt = ret;
        keypoints.push_back(kp);


        // std::cout << "(" << ret.x << ", " << ret.y << ")  ";
    }

    std::cout << std::endl;

    return keypoints;
}

cv::Mat DrawMatches(const cv::Mat& src_img, const std::vector<cv::Point2f>& src_kps_normalized,
                    const cv::Mat& dst_img, const std::vector<cv::Point2f>& dst_kps_normalized) {

    
    std::cout << "SRC Keypoints: " << std::endl;
    auto src_kps = UnnormalizeKeypoints(src_kps_normalized, src_img.cols, src_img.rows);
    std::cout << "DST Keypoints: " << std::endl;
    auto dst_kps = UnnormalizeKeypoints(dst_kps_normalized, dst_img.cols, dst_img.rows);

    std::vector<cv::DMatch> matches;
    for (size_t i = 0; i < src_kps.size(); i++) {
      matches.emplace_back(i, i, 0);
    }

    cv::Mat outImg;
    cv::drawMatches(src_img, src_kps, dst_img, dst_kps, matches, outImg, 1);
    
    return outImg;
}


int main(int argc , char* argv[])
{
    /* ****** CONFIG START ****** */
    std::string lightglue_path = "${LightGlueOnnxModelPath}";
    std::string extractor_path = "${ExtractorOnnxMOdelPath}";

    // Type of feature extractor. Supported extractors are 'superpoint' and 'disk'.
    std::string extractor_type = "${ModelExtractorType}";
    // Sample image size for ONNX tracing , resize the longer side of the images to this value. Supported image size {512 , 1024 , 2048}
    unsigned int image_size = 512; 
    bool grayscale = false;
    bool end2end;
    std::string device = "${Device}"; // Now support "cpu" / "cuda"
    bool viz = true;
    float matchThresh = 0.0f;

    std::string image_path1 = "${YourImageDirPath1}";
    std::string image_path2 = "${YourImageDirPath2}";
    std::string save_path = "${YourResultSavePath}";

    /* ****** CONFIG END ****** */
    
    /* ****** Usage Example Start ****** */
    image_path1 = "/home/dell/Desktop/2/dir_0";
    image_path2 = "/home/dell/Desktop/2/dir_1";
    // image_path2 = "D:\\OroChiLab\\LightGlue\\data\\dir_1";
    device = "cuda";
    
    // End to End 
    // end2end = true;
    // lightglue_path = "D:\\OroChiLab\\LightGlue-OnnxRunner\\models\\superpoint\\superpoint_lightglue_end2end.onnx";
    // extractor_type = "SuperPoint";
    // lightglue_path = "D:\\OroChiLab\\LightGlue\\weights\\onnx\\disk_lightglue_end2end.onnx";
    // extractor_type = "Disk";

    // Decouple
    end2end = false;
    extractor_path = "/home/dell/Downloads/superpoint.onnx";
    lightglue_path = "/home/dell/Downloads/superpoint_lightglue.onnx";
    extractor_type = "SuperPoint";
    // extractor_path = "D:\\OroChiLab\\LightGlue-OnnxRunner\\models\\disk\\disk.onnx";
    // lightglue_path = "D:\\OroChiLab\\LightGlue-OnnxRunner\\models\\disk\\disk_lightglue.onnx";
    // extractor_type = "Disk";

    /* ****** Usage Example End ****** */

    Configuration cfg;
    cfg.lightgluePath = lightglue_path;
    cfg.extractorPath = extractor_path;

    cfg.extractorType = extractor_type;
    cfg.isEndtoEnd = end2end;
    cfg.grayScale = grayscale;
    cfg.image_size = image_size;
    cfg.threshold = matchThresh;
    cfg.device = device;
    cfg.viz = viz;

    std::transform(cfg.extractorType.begin() , cfg.extractorType.end() , \
            cfg.extractorType.begin() , ::tolower);
    if (cfg.extractorType != "superpoint" && cfg.extractorType != "disk")
    {
        std::cerr << "[ERROR] Unsupported feature extractor type: " << extractor_type << std::endl;

        return EXIT_FAILURE;
    }else
    {
        std::cout << "[INFO] Extractor Type : " << cfg.extractorType << std::endl;
    }


    if (fs::exists(cfg.lightgluePath))
    {
        if (cfg.isEndtoEnd)
        {
            if (!fs::exists(cfg.lightgluePath))
            {
                std::cerr << "[ERROR] The specified LightGlue mode at is not end-to-end. Please pass the extractor_path argument." << extractor_type << std::endl;
                return EXIT_FAILURE;
            }
        }
    }else 
    {
        std::cerr << "[ERROR] LightGlue onnx model Path is not exist : " << cfg.lightgluePath << std::endl;
    }
    
    std::vector<cv::String> image_filelist1;
    std::vector<cv::String> image_filelist2;
    cv::glob(image_path1 , image_filelist1);
    cv::glob(image_path2 , image_filelist2);
    if (image_filelist1.size() != image_filelist2.size())
    {
        std::cout << "[INFO] Image Matlist1 size : " << image_filelist1.size() << std::endl;
        std::cout << "[INFO] Image Matlist2 size : " << image_filelist2.size() << std::endl;
        std::cerr << "[ERROR] The number of images in the source folder and \
                    the destination folder is inconsistent" << std::endl;

        return EXIT_FAILURE;
    }

    std::cout << "[INFO] => Building Image Matlist1" << std::endl;
    std::vector<cv::Mat> image_matlist1 = ReadImage(image_filelist1 , cfg.grayScale);
    std::cout << "[INFO] => Building Image Matlist2" << std::endl;
    std::vector<cv::Mat> image_matlist2 = ReadImage(image_filelist2 , cfg.grayScale);

    /* - * -------- End to End Example -------- * - */
    BaseFeatureMatchOnnxRunner* FeatureMatcher;
    if (cfg.isEndtoEnd)
    {
        FeatureMatcher = new LightGlueOnnxRunner();
        FeatureMatcher->InitOrtEnv(cfg);
        FeatureMatcher->SetMatchThresh(cfg.threshold);
    }else {
        FeatureMatcher = new LightGlueDecoupleOnnxRunner();
        FeatureMatcher->InitOrtEnv(cfg);
        FeatureMatcher->SetMatchThresh(cfg.threshold);
    }
    
    auto iter1 = image_matlist1.begin();
    auto iter2 = image_matlist2.begin();
    std::string mode = cfg.isEndtoEnd ? "LightGlueOnnxRunner" : "LightGlueDecoupleOnnxRunner";

    for (;iter1 != image_matlist1.end() && iter2 !=image_matlist2.end() ; \
            ++iter1, ++iter2)
    {
        auto startTime = std::chrono::steady_clock::now();
        auto kpts_result = FeatureMatcher->InferenceImage(cfg , *iter1 , *iter2);
        auto endTime = std::chrono::steady_clock::now();

        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        std::cout << "[INFO] " << mode << " single picture whole process takes time : " \
                    << elapsedTime << " ms" << std::endl;
        if (cfg.viz)
        {
            std::vector<cv::Mat> imagesPair = {*iter1 , *iter2};
            std::vector<std::string> titlePair = {"srcImage" , "destImage"};
            // cv:Mat figure = plotImages(imagesPair , kpts_result , titlePair);
            
            cv::Mat src_resized, dst_resized;
            cv::resize(imagesPair[0], src_resized, cv::Size(512, 512));
            cv::resize(imagesPair[1], dst_resized, cv::Size(512, 512));
            cv::Mat figure = DrawMatches(src_resized, kpts_result.first, dst_resized, kpts_result.second);
            cv::imshow("Figure", figure);
            cv::waitKey(0);
            cv::destroyAllWindows();

        }
        auto kpts = FeatureMatcher->GetKeypointsResult();
    }
    if (cfg.isEndtoEnd)
    {
        printf("[INFO] End2End model inference %ld images mean cost %.2f ms" , image_filelist1.size() , (FeatureMatcher->GetTimer() / image_filelist1.size()));
    }else
    {
        printf("[INFO] Decouple model extractor inference %ld images mean cost %.2f ms , matcher mean cost %.2f\n" , image_filelist1.size() , \
                    (FeatureMatcher->GetTimer("extractor") / image_filelist1.size()) , (FeatureMatcher->GetTimer() / image_filelist1.size()));
    }
    
    return EXIT_SUCCESS;
}


