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

    Error operator+(Error _error)
    {
        Error _p;
        _p.error = error + _error.error;
        _p.count = count + _error.count;

        return _p;
    }

    void operator+=(Error _error)
    {
        error += _error.error;
        count += _error.count;
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
