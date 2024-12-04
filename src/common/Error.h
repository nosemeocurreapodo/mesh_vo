#pragma once

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
        assert(a.count > 0);

        Error sum;

        if (count > 0)
        {
            sum.error = error / count + a.error / a.count;
        }
        else
        {
            sum.error = a.error / a.count;
        }

        sum.count = 1;

        return sum;
    }

    void operator+=(Error a)
    {
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
    }

    template <typename type>
    void operator+=(type a)
    {
        error += a;
        count++;
    }

    template <typename type>
    void operator*=(type a)
    {
        error *= a;
    }

    float getError()
    {
        assert(count > 0);

        // if(count == 0)
        //     return std::numeric_limits<float>::max();

        return error / count;
    }

    /*
    void operator=(Error _error)
    {
        error = _error.error;
        count = _error.count;
    }
    */

private:
    float error;
    float count;
};
