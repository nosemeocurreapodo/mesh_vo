#pragma once

#include "common/types.h"

template <typename Type>
class dataGPU
{
public:
dataGPU()
    {
        width = 0;
        height = 0;
    }

    dataGPU(int _width, int _height, Type _nodata_value)
    {
        nodata = _nodata_value;
        width = _width;
        height = _height;
        m_data = new Type[width * height];
        /*
        if (m_data == nullptr)
        {
            throw std::bad_alloc();
        }
        */
        set(_nodata_value);
    }

    dataGPU(const dataGPU &other)
    {
        nodata = other.nodata;
        width = other.width;
        height = other.height;
        m_data = new Type[width * height];
        std::memcpy(m_data, other.m_data, sizeof(Type) * width * height);
    }

    dataGPU &operator=(const dataGPU &other)
    {
        if (this != &other)
        {
            delete m_data;

            nodata = other.nodata;
            width = other.width;
            height = other.height;
            m_data = new Type[width * height];

            std::memcpy(m_data, other.m_data, sizeof(Type) * width * height);
        }
        return *this;
    }

    ~dataGPU()
    {
        delete[] m_data;
        // m_data = nullptr;
    }

    void setTexel(const Type value, int y, int x)
    {
        assert(y >= 0 && x >= 0 && y < height && x < width);

        int address = x + y * width;
        m_data[address] = value;
    }

    void set(const Type value)
    {
        std::fill_n(m_data, width * height, value);
    }

    void setToNoData()
    {
        set(nodata);
    }

    void set(Type *data)
    {
        std::memcpy(m_data, data, sizeof(Type) * width * height);
    }

    Type getTexel(int y, int x) const
    {
        assert(y >= 0 && x >= 0 && y < height && x < width);

        int address = x + y * width;
        return m_data[address];
    }

    Type get(float norm_y, float norm_x) const
    {
        float wrapped_y = norm_y;
        float wrapped_x = norm_x;
        if (wrapped_y < 0.0)
            // wrapped_y = std::fabs(wrapped_y);
            wrapped_y = -wrapped_y;
        if (wrapped_x < 0.0)
            // wrapped_x = std::fabs(wrapped_x);
            wrapped_x = -wrapped_x;
        if (wrapped_y > 1.0)
            wrapped_y = 1.0 - (wrapped_y - 1.0);
        if (wrapped_x > 1.0)
            wrapped_x = 1.0 - (wrapped_x - 1.0);
        float x = wrapped_x * (width - 1);
        float y = wrapped_y * (height - 1);
        return bilinear(y, x);
    }

    Type *get()
    {
        return m_data;
    }

    void invert()
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                Type d = getTexel(y, x);
                assert(d != 0);
                setTexel(1.0 / d, y, x);
            }
        }
    }

    float getPercentNoData()
    {
        int nodatacount = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                Type d = getTexel(y, x);
                if (d == nodata)
                    nodatacount++;
            }
        }
        return float(nodatacount) / (width * height);
    }

    dataCPU<Type> generateMipmap()
    {
        dataCPU<Type> mipmap(width / 2, height / 2, nodata);

        for (int y = 0; y < height / 2; y++)
        {
            for (int x = 0; x < width / 2; x++)
            {
                Type pixel = area(y * 2, x * 2);
                mipmap.setTexel(pixel, y, x);
            }
        }
        return mipmap;
    }

    const dataCPU<vec2f> computeFrameDerivative()
    {
        dataCPU<vec2f> dIdpix_image(width, height, vec2f(0.0, 0.0));

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                if (y == 0 || y == height - 1 || x == 0 || x == width - 1)
                {
                    // dx.set(0.0, y, x, lvl);
                    // dy.set(0.0, y, x, lvl);
                    dIdpix_image.setTexel(vec2f(0.0f, 0.0f), y, x);
                    continue;
                }

                float _dx = (float(getTexel(y, x + 1)) - float(getTexel(y, x - 1))) * width / 2.0;
                float _dy = (float(getTexel(y + 1, x)) - float(getTexel(y - 1, x))) * height / 2.0;

                dIdpix_image.setTexel(vec2f(_dx, _dy), y, x);
            }
        return dIdpix_image;
    }

    vec2f getMinMax()
    {
        Type min = nodata;
        Type max = nodata;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                Type d = getTexel(y, x);
                if (d == nodata)
                    continue;
                if (min == nodata || d < min)
                    min = d;
                if (max == nodata || d > max)
                    max = d;
            }
        }
        return {min, max};
    }

    vec2f getMeanStd()
    {
        int n = 0;
        float old_m = 0.0;
        float new_m = 0.0;
        float old_s = 0.0;
        float new_s = 0.0;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                Type d = getTexel(y, x);
                if (d == nodata)
                    continue;

                n += 1;

                if (n == 1)
                {
                    old_m = x;
                    new_m = x;
                    old_s = 0;
                }
                else
                {
                    new_m = old_m + (x - old_m) / n;
                    new_s = old_s + (x - old_m) * (x - new_m);

                    old_m = new_m;
                    old_s = new_s;
                }
            }
        }
        return {new_m, sqrt(new_s / (n - 1))};
    }

    template <typename type2>
    dataCPU<Type> operator*(type2 c)
    {
        dataGPU<Type> result(width, height, nodata);
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                Type d = getTexel(y, x);
                if (d == nodata)
                    continue;
                Type res = d * c;
                result.setTexel(res, y, x);
            }
        return result;
    }

    template <typename type2>
    dataCPU<Type> operator+(type2 c)
    {
        dataCPU<Type> result(width, height, nodata);
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                Type d = getTexel(y, x);
                if (d == nodata)
                    continue;
                Type res = d + c;
                result.setTexel(res, y, x);
            }
        return result;
    }

    void normalize(float min, float max)
    {
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                Type d = getTexel(y, x);
                if (d == nodata)
                    continue;
                Type res = (d - min) / (max - min);
                setTexel(res, y, x);
            }
    }

    template <typename type2>
    dataCPU<type2> convert()
    {
        dataCPU<type2> result(width, height, type2(nodata));
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                Type d = getTexel(y, x);
                if (d == nodata)
                    continue;
                type2 res = type2(d);
                result.setTexel(res, y, x);
            }
        return result;
    }

    /*
    dataCPU add(dataCPU &other, int lvl)
    {
        dataCPU<Type> result(lvlWidths[lvl], lvlHeights[lvl], nodata);
        for (int y = 0; y < sizes[lvl][1]; y++)
            for (int x = 0; x < sizes[lvl][0]; x++)
            {
                Type p1 = get(y, x, lvl);
                Type p2 = other.get(y, x, lvl);
                if (p1 == nodata || p2 == other.nodata)
                    continue;
                Type res = p1 + p2;
                result.set(res, y, x, lvl);
            }
        return result;
    }

    dataCPU sub(dataCPU &other, int lvl)
    {
        dataCPU<Type> result(sizes[0][0], sizes[0][1], nodata);
        for (int y = 0; y < sizes[lvl][1]; y++)
            for (int x = 0; x < sizes[lvl][0]; x++)
            {
                Type p1 = get(y, x, lvl);
                Type p2 = other.get(y, x, lvl);
                if (p1 == nodata || p2 == other.nodata)
                    continue;
                Type res = p1 - p2;
                result.set(res, y, x, lvl);
            }
        return result;
    }
    */

    Type nodata;
    int width;
    int height;

private:
    Type bilinear(float y, float x) const
    {
        // bilinear interpolation (-2 because the read the next pixel)
        // int _x = std::min(std::max(int(x), 0), texture[lvl].cols-2);
        // int _y = std::min(std::max(int(y), 0), texture[lvl].rows-2);
        if (y > height - 2 || x > width - 2)
            return getTexel(int(y), int(x));

        int _x = int(x);
        int _y = int(y);
        float dx = x - _x;
        float dy = y - _y;

        float weight_tl = (1.0 - dx) * (1.0 - dy);
        float weight_tr = (dx) * (1.0 - dy);
        float weight_bl = (1.0 - dx) * (dy);
        float weight_br = (dx) * (dy);

        Type tl = getTexel(_y, _x);
        Type tr = getTexel(_y, _x + 1);
        Type bl = getTexel(_y + 1, _x);
        Type br = getTexel(_y + 1, _x + 1);

        if (tl == nodata || tr == nodata || bl == nodata || br == nodata)
            return nodata;

        Type pix = Type(tl * weight_tl +
                        tr * weight_tr +
                        bl * weight_bl +
                        br * weight_br);

        return pix;
    }

    Type area(int y, int x) const
    {
        // bilinear interpolation (-2 because the read the next pixel)
        // int _x = std::min(std::max(int(x), 0), texture[lvl].cols-2);
        // int _y = std::min(std::max(int(y), 0), texture[lvl].rows-2);
        if (y > height - 2 || x > width - 2)
            return getTexel(y, x);

        Type tl = getTexel(y, x);
        Type tr = getTexel(y, x + 1);
        Type bl = getTexel(y + 1, x);
        Type br = getTexel(y + 1, x + 1);

        if (tl == nodata || tr == nodata || bl == nodata || br == nodata)
            return nodata;

        Type pix = Type((tl + tr + bl + br) / 4.0f);

        return pix;
    }

    Type *m_data;
};

template <typename Type>
class dataMipMapCPU
{
public:
    dataMipMapCPU()
    {
    }

    dataMipMapCPU(int _width, int _height, Type _nodata_value)
    {
        int width = _width;
        int height = _height;
        nodata = _nodata_value;

        while (true)
        {
            dataCPU<Type> lvlData(width, height, nodata);
            data.push_back(lvlData);

            width = int(width / 2);
            height = int(height / 2);

            if (width <= 1 || height <= 1)
                break;
        }
    }

    dataMipMapCPU(const dataCPU<Type> image)
    {
        dataCPU<Type> imageLoD = image;
        nodata = image.nodata;

        while (true)
        {
            data.push_back(imageLoD);
            imageLoD = imageLoD.generateMipmap();

            if (imageLoD.width <= 1 || imageLoD.height <= 1)
                break;
        }
    }

    dataMipMapCPU(const dataMipMapCPU &other)
    {
        nodata = other.nodata;
        for (size_t lvl = 0; lvl < other.data.size(); lvl++)
        {
            data.push_back(other.data[lvl]);
        }
    }

    dataMipMapCPU &operator=(const dataMipMapCPU &other)
    {
        if (this != &other)
        {
            nodata = other.nodata;
            data.clear();

            for (size_t lvl = 0; lvl < other.data.size(); lvl++)
            {
                data.push_back(other.data[lvl]);
            }
        }
        return *this;
    }

    void setToNoData(int lvl)
    {
        data[lvl].set(nodata);
    }

    dataCPU<Type> &get(int lvl)
    {
        return data[lvl];
    }

    void generateMipmaps(int baselvl = 0)
    {
        for (size_t lvl = baselvl + 1; lvl < data.size(); lvl++)
        {
            dataCPU<Type> d = data[lvl - 1].generateMipmap();
            data[lvl] = d;
        }
    }

    int getLvls()
    {
        return data.size();
    }

    Type nodata;

private:
    std::vector<dataCPU<Type>> data;
};