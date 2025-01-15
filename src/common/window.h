#pragma once

#include "common/types.h"

class window
{
public:
    window(int minx, int maxx, int miny, int maxy)
    {
        min_x = minx;
        min_y = miny;
        max_x = maxx;
        max_y = maxy;
    }

    bool isPixInWindow(vec2f pix)
    {
        if (pix(0) > min_x && pix(0) < max_x && pix(1) > min_y && pix(1) < max_y)
            return true;
        return false;
    }

    void intersect(window win)
    {
        min_x = std::max(min_x, win.min_x);
        max_x = std::min(max_x, win.max_x);
        min_y = std::max(min_y, win.min_y);
        max_y = std::min(max_y, win.max_y);
    }

    int min_x;
    int max_x;
    int min_y;
    int max_y;
};

