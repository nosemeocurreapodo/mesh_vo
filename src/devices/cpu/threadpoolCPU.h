#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

template <int numThreads>
class ThreadPool
{
public:
    ThreadPool() : stop(false), activeTasks(0)
    {
        for (size_t i = 0; i < numThreads; ++i)
        {
            workers.emplace_back([this]
                                 { this->workerThread(); });
        }
    }

    static constexpr int getNumThreads()
    {
        return numThreads;
    }

    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers)
        {
            worker.join();
        }
    }

    void enqueue(std::function<void()> task)
    {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace(task);
            ++activeTasks;
        }
        condition.notify_one();
    }

    void waitUntilDone()
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        doneCondition.wait(lock, [this]
                           { return this->tasks.empty() && activeTasks == 0; });
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    std::condition_variable doneCondition;
    bool stop;
    size_t activeTasks;

    void workerThread()
    {
        while (true)
        {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                condition.wait(lock, [this]
                               { return this->stop || !this->tasks.empty(); });
                if (this->stop && this->tasks.empty())
                {
                    return;
                }
                task = std::move(this->tasks.front());
                this->tasks.pop();
            }
            task();
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                --activeTasks;
                if (activeTasks == 0 && tasks.empty())
                {
                    doneCondition.notify_all();
                }
            }
        }
    }
};

/*
int main() {
    ThreadPool pool(4);
    for (int i = 0; i < 8; ++i) {
        pool.enqueue([i] { std::cout << "Task " << i << " executed\n"; });
    }
    return 0;
}
*/
