#pragma once

template <typename Type>
class dataCPU
{
public:
    dataCPU(int width, int height, Type nodata_value)
    {
        nodata = nodata_value;

        int _width = width;
        int _height = height;
        lvls = 0;

        while (true)
        {
            Type* tex = new (std::nothrow) Type[_width * _height];
            if (!tex)
                break;

            std::fill_n(tex, _width * _height, nodata);

            lvlWidths.push_back(_width);
            lvlHeights.push_back(_height);
            texture.push_back(tex);
            lvls++;

            _width = int(_width / 2);
            _height = int(_height / 2);

            if (_width == 0 || _height == 0)
                break;
        }
    }

    dataCPU(const dataCPU &other)
    {
        // std::fill(std::begin(texture), std::end(texture), nullptr);
        for (size_t lvl = 0; lvl < texture.size(); lvl++)
        {
            delete[] texture[lvl];
            //texture[lvl] = nullptr;
        }
        texture.clear();

        nodata = other.nodata;
        lvlWidths = other.lvlWidths;
        lvlHeights = other.lvlHeights;
        lvls = other.lvls;

        for (int lvl = 0; lvl < lvls; lvl++)
        {
            Type* tex = new (std::nothrow) Type[lvlWidths[lvl] * lvlHeights[lvl]];
            std::memcpy(tex, other.texture[lvl], sizeof(Type) * lvlWidths[lvl] * lvlHeights[lvl]);
            texture.push_back(tex);
        }
    }

    dataCPU &operator=(const dataCPU &other)
    {
        if (this != &other)
        {
            // Free existing resources
            for (size_t lvl = 0; lvl < texture.size(); lvl++)
            {
                delete[] texture[lvl];
                //texture[lvl] = nullptr;
            }
            texture.clear();

            nodata = other.nodata;
            lvlWidths = other.lvlWidths;
            lvlHeights = other.lvlHeights;
            lvls = other.lvls;

            //std::fill(std::begin(texture), std::end(texture), nullptr);

            for (int lvl = 0; lvl < lvls; lvl++)
            {
                //texture[lvl] = new Type[sizes[lvl][0] * sizes[lvl][1]];
                Type* tex = new (std::nothrow) Type[lvlWidths[lvl] * lvlHeights[lvl]];
                std::memcpy(tex, other.texture[lvl], sizeof(Type) * lvlWidths[lvl] * lvlHeights[lvl]);
                texture.push_back(tex);
            }
        }
        return *this;
    }

    ~dataCPU()
    {
        for (int lvl = 0; lvl < lvls; lvl++)
        {
            delete[] texture[lvl];
            // texture[lvl] = nullptr;
        }
    }

    void set(Type value, int y, int x, int lvl)
    {
        assert(y >= 0 && x >= 0 && y < lvlHeights[lvl] && x < lvlWidths[lvl]);

        int address = x + y * lvlWidths[lvl];
        texture[lvl][address] = value;
    }

    void set(Type value, float y, float x, int lvl)
    {
        set(value, int(y), int(x), lvl);
    }

    void setNormalized(Type value, float norm_y, float norm_x, int lvl)
    {
        float y = norm_y * lvlHeights[lvl];
        float x = norm_x * lvlWidths[lvl];
        set(value, int(y), int(x), lvl);
    }

    void set(Type value, int lvl)
    {
        std::fill_n(texture[lvl], lvlWidths[lvl] * lvlHeights[lvl], value);
    }

    void set(Type *data, int lvl)
    {
        std::memcpy(texture[lvl], data, sizeof(Type) * lvlWidths[lvl] * lvlHeights[lvl]);
    }

    void set(Type *data)
    {
        set(data, 0);
        generateMipmaps();
    }

    inline Type get(int y, int x, int lvl)
    {
        assert(y >= 0 && x >= 0 && y < lvlHeights[lvl] && x < lvlWidths[lvl]);

        int address = x + y * lvlWidths[lvl];
        return texture[lvl][address];
    }

    inline Type get(float y, float x, int lvl)
    {
        return bilinear(y, x, lvl);
    }

    Type getNormalized(float norm_y, float norm_x, int lvl)
    {
        float x = norm_x * lvlWidths[lvl];
        float y = norm_y * lvlHeights[lvl];
        return get(y, x, lvl);
    }

    Type *get(int lvl)
    {
        return texture[lvl];
    }

    void generateMipmaps(int baselvl = 0)
    {
        for (int lvl = baselvl + 1; lvl < lvls; lvl++)
        {
            for (int y = 0; y < lvlHeights[lvl]; y++)
            {
                for (int x = 0; x < lvlWidths[lvl]; x++)
                {
                    Type pixel = area(y * 2, x * 2, lvl - 1);
                    set(pixel, y, x, lvl);
                }
            }
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
        int nodatacount = 0;
        for (int y = 0; y < lvlHeights[lvl]; y++)
        {
            for (int x = 0; x < lvlWidths[lvl]; x++)
            {
                if (get(y, x, lvl) == nodata)
                    nodatacount++;
            }
        }
        return float(nodatacount) / (lvlWidths[lvl] * lvlHeights[lvl]);
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
    std::vector<int> lvlWidths;
    std::vector<int> lvlHeights;

private:
    inline Type bilinear(float y, float x, int lvl)
    {
        // bilinear interpolation (-2 because the read the next pixel)
        // int _x = std::min(std::max(int(x), 0), texture[lvl].cols-2);
        // int _y = std::min(std::max(int(y), 0), texture[lvl].rows-2);
        if (y > lvlHeights[lvl] - 2 || x > lvlWidths[lvl] - 2)
            return get(int(y), int(x), lvl);

        int _x = int(x);
        int _y = int(y);
        float dx = x - _x;
        float dy = y - _y;

        float weight_tl = (1.0 - dx) * (1.0 - dy);
        float weight_tr = (dx) * (1.0 - dy);
        float weight_bl = (1.0 - dx) * (dy);
        float weight_br = (dx) * (dy);

        Type pix = get(_y, _x, lvl) * weight_tl +
                   get(_y, _x + 1, lvl) * weight_tr +
                   get(_y + 1, _x, lvl) * weight_bl +
                   get(_y + 1, _x + 1, lvl) * weight_br;

        return pix;
    }

    Type area(int y, int x, int lvl)
    {
        // bilinear interpolation (-2 because the read the next pixel)
        // int _x = std::min(std::max(int(x), 0), texture[lvl].cols-2);
        // int _y = std::min(std::max(int(y), 0), texture[lvl].rows-2);
        if (y > lvlHeights[lvl] - 2 || x > lvlWidths[lvl] - 2)
            return get(y, x, lvl);

        Type pix = (get(y, x, lvl) +
                    get(y, x + 1, lvl) +
                    get(y + 1, x, lvl) +
                    get(y + 1, x + 1, lvl)) /
                   4.0;

        return pix;
    }

    std::vector<Type*> texture;
};
