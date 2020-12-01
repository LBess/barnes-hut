// Liam Bessell, 10/6/20

#pragma once
#include <chrono>

class MyTimer
{
public:
    void start();
    int elapsedMS();
    void reset();

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
};
