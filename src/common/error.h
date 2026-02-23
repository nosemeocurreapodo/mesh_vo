#pragma once

#include "mpdr/common/huber.h"

class Error
{
public:
    Error()
    {
        e_ = 0.0;
        c_ = 0;
    }

    void setZero()
    {
        e_ = 0.0;
        c_ = 0;
    }

    Error operator+(Error a)
    {
        Error sum;

        /*
        assert(a.count > 0);

        if (count > 0)
        {
            sum.error = error / count + a.error / a.count;
        }
        else
        {
            sum.error = a.error / a.count;
        }

        sum.count = 1;
        */

        sum.e_ = e_ + a.e_;
        sum.c_ = c_ + a.c_;

        return sum;
    }

    void operator+=(Error a)
    {
        /*
        assert(a.count > 0);

        if (count > 0)
        {
            error = error / count + a.error / a.count;
        }
        else
        {
            error = a.error / a.count;
        }
        count = 1;
        */

        e_ += a.e_;
        c_ += a.c_;
    }

    template <typename type>
    void operator+=(type a)
    {
        e_ += a;
        c_++;
    }

    template <typename type>
    void operator*=(type a)
    {
        e_ *= a;
    }

    float operator()() const
    {
        return e_;
    }

    float getError() const
    {
        return e_;
    }

    int getCount() const
    {
        return c_;
    }

    /*
    void operator=(Error _error)
    {
        error = _error.error;
        count = _error.count;
    }
    */

private:
    float e_;
    float c_;
};
