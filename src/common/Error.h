#pragma once

#include <Eigen/Core>

//struct HJPose{
//    float J[6];
//    float H[21];
//    float error;
//    int cout;
//};

class Error
{
public:
    Error()
    {
        error = 0.0;
        count = 0;
    }

    void setZero()
    {
        error = 0.0;
        count = 0.0;
    }

    Error operator+(Error a)
    {
        Error sum;
        sum.error = error + a.error;
        sum.count = count + a.count;

        return sum;
    }

    void operator+=(Error a)
    {
        error += a.error;
        count += a.count;
    }
    
    /*
    void operator=(Error _error)
    {
        error = _error.error;
        count = _error.count;
    }
    */

    float error;
    float count;

private:

};
