// Liam Bessell, 10/6/20

#include "MyTimer.h"

void MyTimer::start()
{
    startTime = std::chrono::high_resolution_clock::now();
}

int MyTimer::elapsedMS()
{
    std::chrono::duration<float> elapsed = std::chrono::high_resolution_clock::now() - startTime;
    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
}

void MyTimer::reset()
{
    startTime = std::chrono::high_resolution_clock::now();
}
