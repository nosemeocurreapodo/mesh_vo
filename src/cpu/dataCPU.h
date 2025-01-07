#pragma once

template <typename Type>
class dataCPU
{
public:

    dataCPU()
    {

    }

    dataCPU(int _width, int _height, Type _nodata_value)
    {
        nodata = _nodata_value;
        width = _width;
        height = _height;
        m_data = new (std::nothrow) Type[width * height];
        assert(m_data != nullptr);
    }

    dataCPU(const dataCPU &other)
    {
        nodata = other.nodata;
        width = other.width;
        height = other.height;
        m_data = new (std::nothrow) Type[width * height];
        std::memcpy(m_data, other.m_data, sizeof(Type) * width * height);
    }

    dataCPU &operator=(const dataCPU &other)
    {
        if (this != &other)
        {
            assert(width == other.width && height == other.height);

            nodata = other.nodata;
            width = other.width;
            height = other.height;

            std::memcpy(m_data, other.m_data, sizeof(Type) * width * height);
        }
        return *this;
    }

    ~dataCPU()
    {
        delete m_data;
    }

    void set(Type value, int y, int x)
    {
        assert(y >= 0 && x >= 0 && y < height && x < width);

        int address = x + y * width;
        m_data[address] = value;
    }

    void set(Type value, float y, float x)
    {
        set(value, int(y), int(x));
    }

    void setNormalized(Type value, float norm_y, float norm_x)
    {
        float y = norm_y * height;
        float x = norm_x * width;
        set(value, int(y), int(x));
    }

    void set(Type value)
    {
        std::fill_n(m_data, width * height, value);
    }

    void set(Type *data)
    {
        std::memcpy(m_data, data, sizeof(Type) * width * height);
    }

    Type get(int y, int x)
    {
        assert(y >= 0 && x >= 0 && y < height && x < width);

        int address = x + y * width;
        return m_data[address];
    }

    Type get(float y, float x)
    {
        return bilinear(y, x);
    }

    Type getNormalized(float norm_y, float norm_x)
    {
        float x = norm_x * width;
        float y = norm_y * height;
        return get(y, x);
    }

    Type *get()
    {
        return m_data;
    }

    /*
    void invert(int lvl = 0)
    {
        for (int y = 0; y < sizes[lvl][1]; y++)
        {
            for (int x = 0; x < sizes[lvl][0]; x++)
            {
                Type pixel = get(y, x, lvl);
                set(1.0 / pixel, y, x, lvl);
            }
        }
    }
    */

    float getPercentNoData()
    {
        int nodatacount = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (get(y, x) == nodata)
                    nodatacount++;
            }
        }
        return float(nodatacount) / (width * height);
    }

    dataCPU<Type> generateMipmap(int baselvl = 0)
    {
        dataCPU<Type> mipmap(width/2, height/2, nodata);

        for (int y = 0; y < height/2; y++)
        {
            for (int x = 0; x < width/2; x++)
            {
                Type pixel = area(y * 2, x * 2);
                mipmap.set(pixel, y, x);
            }
        }
        return mipmap;
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
    inline Type bilinear(float y, float x)
    {
        // bilinear interpolation (-2 because the read the next pixel)
        // int _x = std::min(std::max(int(x), 0), texture[lvl].cols-2);
        // int _y = std::min(std::max(int(y), 0), texture[lvl].rows-2);
        if (y > height - 2 || x > width - 2)
            return get(int(y), int(x));

        int _x = int(x);
        int _y = int(y);
        float dx = x - _x;
        float dy = y - _y;

        float weight_tl = (1.0 - dx) * (1.0 - dy);
        float weight_tr = (dx) * (1.0 - dy);
        float weight_bl = (1.0 - dx) * (dy);
        float weight_br = (dx) * (dy);

        Type pix = get(_y, _x) * weight_tl +
                   get(_y, _x + 1) * weight_tr +
                   get(_y + 1, _x) * weight_bl +
                   get(_y + 1, _x + 1) * weight_br;

        return pix;
    }

    Type area(int y, int x)
    {
        // bilinear interpolation (-2 because the read the next pixel)
        // int _x = std::min(std::max(int(x), 0), texture[lvl].cols-2);
        // int _y = std::min(std::max(int(y), 0), texture[lvl].rows-2);
        if (y > height - 2 || x > width - 2)
            return get(y, x);

        Type pix = (get(y, x) +
                    get(y, x + 1) +
                    get(y + 1, x) +
                    get(y + 1, x + 1)) /
                   4.0;

        return pix;
    }

    Type* m_data;
};


template <typename Type>
class dataMipMapCPU
{
public:
    dataMipMapCPU(int _width, int _height, Type _nodata_value)
    {
        lvls = 0;

        int width = _width;
        int height = _height;

        while (true)
        {
            dataCPU<Type> lvlData(width, height, _nodata_value);
            data.push_back(lvlData);

            lvls++;

            _width = int(_width / 2);
            _height = int(_height / 2);

            if (_width == 0 || _height == 0)
                break;
        }
    }

    dataMipMapCPU(const dataMipMapCPU &other)
    {
        lvls = other.lvls;

        for (int lvl = 0; lvl < lvls; lvl++)
        {
            dataCPU<Type> lvlData = other.data[lvl];
            data.push_back(lvlData);
        }
    }

    dataMipMapCPU &operator=(const dataMipMapCPU &other)
    {
        if (this != &other)
        {
            assert(lvls == other.lvls);

            for (int lvl = 0; lvl < lvls; lvl++)
            {
                data[lvl] = other.data[lvl];
            }
        }
        return *this;
    }

    void set(Type value, int y, int x, int lvl)
    {
        data[lvl].set(value, y, x);
    }

    void set(Type value, float y, float x, int lvl)
    {
        data[lvl].set(value, y, x);
    }

    void setNormalized(Type value, float norm_y, float norm_x, int lvl)
    {
        data[lvl].setNormalized(value, norm_y, norm_x);
    }

    void set(Type value, int lvl)
    {
        data[lvl].set(value);
    }

    void set(Type *data, int lvl)
    {
        data[lvl].set(data, lvl);
    }

    void set(Type *data)
    {
        set(data, 0);
        generateMipmaps();
    }

    Type get(int y, int x, int lvl)
    {
        return data[lvl].get(y, x);
    }

    Type get(float y, float x, int lvl)
    {
        return data[lvl].get(y, x);
    }

    Type getNormalized(float norm_y, float norm_x, int lvl)
    {
        return data[lvl].getNormalized(lvl);
    }

    dataCPU<Type> &get(int lvl)
    {
        return data[lvl];
    }

    void generateMipmaps(int baselvl = 0)
    {
        for (int lvl = baselvl + 1; lvl < lvls; lvl++)
        {
            data[lvl] = data[lvl-1].generateMipmap();
        }
    }

    /*
    void invert(int lvl = 0)
    {
        for (int y = 0; y < sizes[lvl][1]; y++)
        {
            for (int x = 0; x < sizes[lvl][0]; x++)
            {
                Type pixel = get(y, x, lvl);
                set(1.0 / pixel, y, x, lvl);
            }
        }
    }
    */

    float getPercentNoData(int lvl)
    {
        return data[lvl].getPercentNoData(lvl);
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
    int lvls;

private:

    std::vector<dataCPU<Type>> data;
};