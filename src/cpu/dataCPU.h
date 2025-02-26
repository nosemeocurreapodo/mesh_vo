#pragma once

template <typename Type>
class dataCPU
{
public:

    dataCPU()
    {
        width = 0;
        height = 0;
    }

    dataCPU(int _width, int _height, Type _nodata_value)
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

    dataCPU(const dataCPU &other)
    {
        nodata = other.nodata;
        width = other.width;
        height = other.height;
        m_data = new Type[width * height];
        std::memcpy(m_data, other.m_data, sizeof(Type) * width * height);
        // for (int y = 0; y < height; y++)
        //     for (int x = 0; x < width; x++)
        //         setTexel(other.getTexel(y, x), y, x);
    }

    dataCPU &operator=(const dataCPU &other)
    {
        if (this != &other)
        {
            delete m_data;

            nodata = other.nodata;
            width = other.width;
            height = other.height;
            m_data = new Type[width * height];

            std::memcpy(m_data, other.m_data, sizeof(Type) * width * height);
            // for (int y = 0; y < height; y++)
            //     for (int x = 0; x < width; x++)
            //         setTexel(other.getTexel(y, x), y, x);
        }
        return *this;
    }

    ~dataCPU()
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

    /*
    void setTexel(const Type value, float y, float x)
    {
        setTexel(value, int(round(y)), int(round(x)));
    }
    */

    /*
    void set(const Type value, float norm_y, float norm_x)
    {
        float y = norm_y * (height - 1);
        float x = norm_x * (width - 1);
        setTexel(value, y, x);
    }
    */

    void set(const Type value)
    {
        std::fill_n(m_data, width * height, value);

        // for(int y = 0; y < height; y++)
        //     for(int x = 0; x < width; x++)
        //         set(value, y, x);
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

    /*
    Type getTexel(float y, float x) const
    {
        return getTexel(int(round(y)), int(round(x)));
    }
    */

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

        Type pix = tl * weight_tl +
                   tr * weight_tr +
                   bl * weight_bl +
                   br * weight_br;

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

        Type pix = (tl + tr + bl + br) / 4.0;

        return pix;
    }

    Type *m_data;
};

template <typename Type>
class dataMipMapCPU
{
public:
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

    dataMipMapCPU(const dataMipMapCPU &other)
    {
        for (size_t lvl = 0; lvl < other.data.size(); lvl++)
        {
            dataCPU<Type> lvlData = other.data[lvl];
            data.push_back(lvlData);
        }
    }

    dataMipMapCPU &operator=(const dataMipMapCPU &other)
    {
        if (this != &other)
        {
            assert(data.size() == other.data.size());

            for (size_t lvl = 0; lvl < other.data.size(); lvl++)
            {
                data[lvl] = other.data[lvl];
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