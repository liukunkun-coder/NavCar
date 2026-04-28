//创建一个线程池的文件，线程池的功能是处理图像增强的任务，线程池中有一个线程专门负责从队列中取出图像进行处理，处理完成后将结果显示出来。
//通过纯CPP文件实现
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

// OpenCV 头文件
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
//首先定一个结构体 包含线程以及的装态 id
struct Mythread {
    std::thread thread;
    bool isRunning;
    int id;
};

//创建一个线程池类

class ThreadPool {
private:
    std::vector<Mythread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_ = false;
public:
    ThreadPool(size_t threads) {
        for (size_t i = 0; i < threads; ++i) {
            //创建线程并将其加入线程池
            workers_.emplace_back(Mythread{std::thread(&ThreadPool::workers_, this), true, static_cast<int>(i)});
        }
    }
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        //通知所有线程停止工作
        condition_.notify_all();
        //等待所有线程完成工作
        for (auto& worker : workers_) {
            if (worker.thread.joinable()) {
                worker.thread.join();
            }
        }
    }
    //添加任务到线程池
    template<class F, class... Args>
    void enqueue(F&& f, Args&&... args) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            //将任务添加到队列中
            tasks_.emplace(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        }
        //通知一个线程有新的任务
        condition_.notify_one();
    }
private:
    //工作线程函数
    void workers_func() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                //等待任务队列中有任务或者线程池停止
                condition_.wait(lock, [this] { return !tasks_.empty() || stop_; });
                if (stop_ && tasks_.empty()) {
                    return; //线程池停止且没有任务，退出线程
                }
                //取出一个任务
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            //执行任务
            task();
        }   
        }
    //清理调所用的资源。包括清空任务队列，等待所有线程完成工作等。
    void clear() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            while (!tasks_.empty()) tasks_.pop();
        }
        condition_.notify_all();
        for (auto& worker : workers_) {
            if (worker.thread.joinable()) {
                worker.thread.join();
            }
        }
    }
};