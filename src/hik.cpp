#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "MvCameraControl.h"
#include "CameraParams.h"
#include "PixelType.h"
#include "yaml-cpp/yaml.h"  
#include <vector>


class hikConfig {
    private:
        YAML::Node config_;
    public:
        hikConfig(const std::string& filename) {
            loadconfig(filename, config_);
        }
    public:
    // 读取 YAML 配置文件
        bool loadconfig (const std::string& filename, YAML::Node& config) {
            try {
                config = YAML::LoadFile(filename);
                return true;
            } catch (const std::exception& e) {
                std::cerr << "读取文件失败: " << e.what() << std::endl;
                return false;
            }
        }

        // 获取配置项，读取相机的参数
        //创建一个读取参数的函数，返回一个结构体或者类对象，包含相机的参数
        struct CameraParams {
            int exposure_time;
            float gain;
            int width;
            int height;
            int pixel_format;
        };
        CameraParams getCameraParams() {
            CameraParams params;
            try {
                params.exposure_time = config_["exposure_time"].as<int>();
                params.gain = config_["gain"].as<float>();
                params.width = config_["width"].as<int>();
                params.height = config_["height"].as<int>();
                params.pixel_format = config_["pixel_format"].as<int>();
            } catch (const std::exception& e) {
                std::cerr << "读取配置项失败: " << e.what() << std::endl;
            }
            return params;
    }
};


class ImageQueue {
private:
    std::queue<cv::Mat> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    int max_size_;
    
public:
    ImageQueue(int max_size = 5) : max_size_(max_size) {}
    
    void push(const cv::Mat& img) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.size() >= max_size_) {
            queue_.pop();  // 丢弃旧帧
        }
        queue_.push(img.clone());  // 深拷贝，因为回调中的原数据会被释放
        cond_.notify_one();
    }
    
    bool pop(cv::Mat& img, int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (cond_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                           [this] { return !queue_.empty(); })) {
            img = queue_.front();
            queue_.pop();
            return true;
        }
        return false;
    }
    
    void clear() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!queue_.empty()) queue_.pop();
    }
};

// ==================== 海康相机封装类 ====================
class HikCamera {
private:
    void* handle_ = nullptr;
    ImageQueue image_queue_;
    std::thread process_thread_;
    bool is_grabbing_ = false;
    double fps_ = 0.0;
    std::chrono::steady_clock::time_point last_time_;

    
    // 图像像素格式转换（根据实际相机配置）
    //创建三个用来处理；绿色通道的函数，来提高绿色弹丸的识别
    cv::Mat extractgreen( const cv::Mat& image){
        //创建一个容器用来存储分离后的绿色通道
        std::vector<cv::Mat> green_chan;
        cv::split(image , green_chan);
        return green_chan[1];
    }

    cv::Mat convertToBGR(unsigned char* pData, MV_FRAME_OUT_INFO_EX* pFrameInfo) {
        cv::Mat bgrMat;
        
        // 根据像素类型选择转换方式
        switch (pFrameInfo->enPixelType) {
            case PixelType_Gvsp_Mono8: {
                cv::Mat rawMat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC1, pData);
                cv::cvtColor(rawMat, bgrMat, cv::COLOR_GRAY2BGR);
                break;
            }
            case PixelType_Gvsp_BayerRG8: {
                cv::Mat rawMat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC1, pData);
                cv::cvtColor(rawMat, bgrMat, cv::COLOR_BayerRG2BGR);
                break;
            }
            case PixelType_Gvsp_BayerGB8: {
                cv::Mat rawMat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC1, pData);
                cv::cvtColor(rawMat, bgrMat, cv::COLOR_BayerGB2BGR);
                break;
            }
            case PixelType_Gvsp_RGB8_Packed: {
                // 【修复1】RGB8_Packed 是 3 通道，不是单通道
                cv::Mat rawMat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC3, pData);
                cv::cvtColor(rawMat, bgrMat, cv::COLOR_RGB2BGR);
                break;
            }
            default:
                std::cerr << "Unsupported pixel format: " << pFrameInfo->enPixelType << std::endl;
        }
        return bgrMat;
    }
    
    // 【关键】SDK 回调函数（在 SDK 内部线程中执行，必须尽快返回！）
    static void __stdcall ImageCallback(unsigned char* pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser) {
        HikCamera* pThis = static_cast<HikCamera*>(pUser);
        
        if (!pThis->is_grabbing_ || pData == nullptr) return;
        
        // 转换为 BGR 格式
        cv::Mat bgrMat = pThis->convertToBGR(pData, pFrameInfo);
        
        // 放入队列（内部会 clone，因为 pData 即将被 SDK 回收）
        if (!bgrMat.empty()) {
            pThis->image_queue_.push(bgrMat);
        }
        
        // 回调结束，SDK 自动释放 pData 指向的内存
    }
    
    // 视觉计算线程
    void processLoop() {
        cv::Mat frame;
        int frame_count = 0;
        last_time_ = std::chrono::steady_clock::now();
    
        while (is_grabbing_) {
            if (image_queue_.pop(frame, 100)) {
                // ========== 计算帧率 ==========
                frame_count++;
            auto current_time = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(current_time - last_time_).count();
            
            if (elapsed >= 1.0) {  // 每秒更新一次帧率
                fps_ = frame_count / elapsed;
                frame_count = 0;
                last_time_ = current_time;
            }
                // ========== 图像增强处理 ==========
                cv::Mat blurred, sharpened, gray, edges, green;
                green = extractgreen(frame);
                // 【修复2】你之前的锐化逻辑有问题，修正如下：
                // 高斯模糊
                cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 0, 0);
                // 锐化：原图 + (原图 - 模糊图) * 权重
                cv::addWeighted(frame, 1.5, blurred, -0.5, 0, sharpened);
                
                // 可选：CLAHE 增强对比度（光照不均时使用）
                cv::Mat lab;
                cv::cvtColor(sharpened, lab, cv::COLOR_BGR2Lab);
                std::vector<cv::Mat> channels;
                cv::split(lab, channels);
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
                clahe->apply(channels[0], channels[0]);
                cv::merge(channels, lab);
                cv::cvtColor(lab, sharpened, cv::COLOR_Lab2BGR);
                
                // 灰度化 + 边缘检测（用于演示）
                cv::cvtColor(sharpened, gray, cv::COLOR_BGR2GRAY);
                cv::Canny(gray, edges, 100, 200);
               // ========== 在图像上绘制帧率 ==========
            cv::Mat display;
            sharpened.copyTo(display);
            
            // 准备帧率文本
            char fps_text[64];
            snprintf(fps_text, sizeof(fps_text), "FPS: %.2f", fps_);
            
            // 设置文本样式
            int font_face = cv::FONT_HERSHEY_SIMPLEX;
            double font_scale = 0.8;
            int thickness = 2;
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(fps_text, font_face, font_scale, thickness, &baseline);
            
            // 文本位置（左上角，带边距）
            cv::Point text_pos(10, text_size.height + 10);
            
            // 绘制黑色背景（可选，让文字更清晰）
            cv::rectangle(display, 
                          cv::Point(5, 5), 
                          cv::Point(text_size.width + 15, text_size.height + 20), 
                          cv::Scalar(0, 0, 0), 
                          -1);
            
            // 绘制绿色文字
            cv::putText(display, fps_text, text_pos, font_face, font_scale, 
                        cv::Scalar(0, 255, 0), thickness);
            
            // 显示带帧率的图像
            cv::imshow("green",green);
            cv::imshow("Sharpened (FPS: " + std::to_string((int)fps_) + ")", gray);
            cv::imshow("Edges", edges);
            cv::waitKey(1);
        }
    }
}
    
public:
    HikCamera() : handle_(nullptr), is_grabbing_(false) {}
    
    ~HikCamera() { stop(); }
    
    bool init() {
        // 1. SDK 初始化
        int nRet = MV_CC_Initialize();
        if (nRet != MV_OK) {
            std::cerr << "MV_CC_Initialize failed" << std::endl;
            return false;
        }
        
        // 2. 枚举设备
        MV_CC_DEVICE_INFO_LIST stDeviceList;
        memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
        nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
        if (nRet != MV_OK || stDeviceList.nDeviceNum == 0) {
            std::cerr << "No camera found" << std::endl;
            MV_CC_Finalize();
            return false;
        }
        
        std::cout << "Found " << stDeviceList.nDeviceNum << " camera(s)" << std::endl;
        
        // 3. 创建句柄并打开设备
        nRet = MV_CC_CreateHandle(&handle_, stDeviceList.pDeviceInfo[0]);
        if (nRet != MV_OK) {
            std::cerr << "Create handle failed" << std::endl;
            MV_CC_Finalize();
            return false;
        }
        
        nRet = MV_CC_OpenDevice(handle_);
        if (nRet != MV_OK) {
            std::cerr << "Open device failed" << std::endl;
            MV_CC_DestroyHandle(handle_);
            MV_CC_Finalize();
            return false;
        }
        
        // ========== 【新增】相机参数优化设置 ==========
        // 设置曝光时间（微秒）
        MV_CC_SetFloatValue(handle_, "ExposureTime", 8000.0f);
        // 设置增益
        MV_CC_SetFloatValue(handle_, "Gain", 5.0f);
        // 开启伽马校正
        MV_CC_SetBoolValue(handle_, "GammaEnable", true);
        MV_CC_SetFloatValue(handle_, "Gamma", 0.7f);
        
        // 4. 设置网络包大小（仅 GigE 相机需要）
        if (stDeviceList.pDeviceInfo[0]->nTLayerType == MV_GIGE_DEVICE) {
            unsigned int nPacketSize = MV_CC_GetOptimalPacketSize(handle_);
            if (nPacketSize > 0) {
                MV_CC_SetIntValue(handle_, "GevSCPSPacketSize", nPacketSize);
            }
        }
        
        // 5. 【关键】注册图像回调
        nRet = MV_CC_RegisterImageCallBackEx(handle_, ImageCallback, this);
        if (nRet != MV_OK) {
            std::cerr << "Register callback failed" << std::endl;
            MV_CC_CloseDevice(handle_);
            MV_CC_DestroyHandle(handle_);
            MV_CC_Finalize();
            return false;
        }
        
        return true;
    }
    
    bool start() {
        if (handle_ == nullptr) return false;
        
        int nRet = MV_CC_StartGrabbing(handle_);
        if (nRet != MV_OK) {
            std::cerr << "Start grabbing failed" << std::endl;
            return false;
        }
        
        is_grabbing_ = true;
        process_thread_ = std::thread(&HikCamera::processLoop, this);
        
        std::cout << "Camera started" << std::endl;
        return true;
    }
    
    void stop() {
        is_grabbing_ = false;
        
        if (process_thread_.joinable()) {
            process_thread_.join();
        }
        
        if (handle_ != nullptr) {
            MV_CC_StopGrabbing(handle_);
            MV_CC_CloseDevice(handle_);
            MV_CC_DestroyHandle(handle_);
            handle_ = nullptr;
        }
        
        MV_CC_Finalize();
        image_queue_.clear();
        
        cv::destroyAllWindows();  // 【新增】关闭所有 OpenCV 窗口
        
        std::cout << "Camera stopped" << std::endl;
    }
    
    bool isRunning() const { return is_grabbing_; }
};

// ==================== 主程序 ====================
int main() {
    HikCamera camera;
    
    if (!camera.init()) {
        std::cerr << "Camera initialization failed" << std::endl;
        return -1;
    }
    
    if (!camera.start()) {
        std::cerr << "Camera start failed" << std::endl;
        return -1;
    }
    
    std::cout << "Press Enter to stop..." << std::endl;
    std::cin.get();
    
    camera.stop();
    
    return 0;
}