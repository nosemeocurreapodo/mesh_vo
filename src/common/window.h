#pragma once

#include "common/types.h"

template <typename Type>
class window
{
public:
    window(Type minx, Type maxx, Type miny, Type maxy)
    {
        min_x = minx;
        min_y = miny;
        max_x = maxx;
        max_y = maxy;
    }

    bool isPixInWindow(Type x, Type y)
    {
        if (x < min_x || x > max_x || y < min_y || y > max_y)
            return false;
        return true;
    }

    bool isPixInWindow(vec2f pix)
    {
        if (pix(0) < min_x || pix(0) > max_x || pix(1) < min_y || pix(1) > max_y)
            return false;
        return true;
    }

    void intersect(window win)
    {
        min_x = std::max(min_x, win.min_x);
        max_x = std::min(max_x, win.max_x);
        min_y = std::max(min_y, win.min_y);
        max_y = std::min(max_y, win.max_y);
    }

    Type min_x;
    Type max_x;
    Type min_y;
    Type max_y;
};
