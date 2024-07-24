#pragma once

template <typename Type>
class dataCPU
{
public:
    dataCPU(int width, int height, Type nodata_value)
    {
        nodata = nodata_value;

        int lvl = 0;
        while (true)
        {
            float scale = std::pow(2.0f, float(lvl));

            int width_s = int(width / scale);
            int height_s = int(height / scale);

            if (width_s <= 0 || height_s <= 0)
                break;

            texture[lvl] = new (std::nothrow) Type[width_s * height_s];
            if (!texture[lvl])
                break;

            sizes[lvl] = {width_s, height_s};
            std::fill_n(texture[lvl], width_s * height_s, nodata);
            lvl++;
        }

        max_lvl = lvl;
    }

    dataCPU(const dataCPU &other)
    {
        nodata = other.nodata;
        max_lvl = other.max_lvl;
        // std::fill(std::begin(texture), std::end(texture), nullptr);
        for (int lvl = 0; lvl < max_lvl; lvl++)
        {
            sizes[lvl] = other.sizes[lvl];
            texture[lvl] = new Type[sizes[lvl][0] * sizes[lvl][1]];
            std::memcpy(texture[lvl], other.texture[lvl], sizeof(Type) * sizes[lvl][0] * sizes[lvl][1]);
        }
    }

    dataCPU &operator=(const dataCPU &other)
    {
        if (this != &other)
        {
            // Free existing resources
            for (int lvl = 0; lvl < max_lvl; lvl++)
            {
                delete[] texture[lvl];
                texture[lvl] = nullptr;
            }

            nodata = other.nodata;
            max_lvl = other.max_lvl;
            std::fill(std::begin(texture), std::end(texture), nullptr);

            for (int lvl = 0; lvl < max_lvl; lvl++)
            {
                sizes[lvl] = other.sizes[lvl];
                texture[lvl] = new Type[sizes[lvl][0] * sizes[lvl][1]];
                std::memcpy(texture[lvl], other.texture[lvl], sizeof(Type) * sizes[lvl][0] * sizes[lvl][1]);
            }
        }
        return *this;
    }

    ~dataCPU()
    {
        for (int lvl = 0; lvl < max_lvl; lvl++)
        {
            delete[] texture[lvl];
            // texture[lvl] = nullptr;
        }
    }

    std::array<int, 2> getSize(int lvl)
    {
        return sizes[lvl];
    }

    void set(Type value, int y, int x, int lvl)
    {
        int address = x + y * sizes[lvl][0];
        if (address < 0 || address >= sizes[lvl][1]*sizes[lvl][0])
            throw std::out_of_range("set invalid address");
        texture[lvl][address] = value;
    }

    void set(Type value, float y, float x, int lvl)
    {
        set(value, int(y), int(x), lvl);
    }

    void setNormalized(Type value, float norm_y, float norm_x, int lvl)
    {
        float y = norm_y * texture[lvl].rows;
        float x = norm_x * texture[lvl].cols;
        set(value, int(y), int(x), lvl);
    }

    void set(Type value, int lvl)
    {
        std::fill_n(texture[lvl], sizes[lvl][0] * sizes[lvl][1], value);
    }

    void set(Type *data, int lvl)
    {
        std::memcpy(texture[lvl], data, sizeof(Type) * sizes[lvl][0] * sizes[lvl][1]);
    }

    void set(Type *data)
    {
        set(data, 0);
        generateMipmaps();
    }

    void setSmooth(int lvl, float start = 0.5, float end = 1.0)
    {
        for (int y = 0; y < sizes[lvl][1]; y++)
        {
            for (int x = 0; x < sizes[lvl][0]; x++)
            {
                float val = start + (end - start) * float(y) / (sizes[lvl][1] - 1.0);
                set(val, y, x, lvl);
            }
        }
    }

    void setRandom(int lvl, float min = 0.5, float max = 1.0)
    {
        for (int y = 0; y < sizes[lvl][1]; y++)
        {
            for (int x = 0; x < sizes[lvl][0]; x++)
            {
                float val = (max - min) * float(rand() % 1000) / 1000.0 + min;
                set(val, y, x, lvl);
            }
        }
    }

    inline Type get(int y, int x, int lvl)
    {
        int address = x + y * sizes[lvl][0];
        if (address < 0 || address >= sizes[lvl][1]*sizes[lvl][0])
            throw std::out_of_range("get invalid address");
        return texture[lvl][address];
    }

    inline Type get(float y, float x, int lvl)
    {
        return bilinear(y, x, lvl);
    }

    Type getNormalized(float norm_y, float norm_x, int lvl)
    {
        float x = norm_x * sizes[lvl][0];
        float y = norm_y * sizes[lvl][1];
        return get(y, x, lvl);
    }

    Type *get(int lvl)
    {
        return texture[lvl];
    }

    void generateMipmaps(int baselvl = 0)
    {
        for (int lvl = baselvl + 1; lvl < max_lvl; lvl++)
        {
            for (int y = 0; y < sizes[lvl][1]; y++)
            {
                for (int x = 0; x < sizes[lvl][0]; x++)
                {
                    Type pixel = area(y * 2, x * 2, lvl - 1);
                    set(pixel, y, x, lvl);
                }
            }
        }
    }

    float getPercentNoData(int lvl)
    {
        int nodatacount = 0;
        for (int y = 0; y < sizes[lvl][1]; y++)
        {
            for (int x = 0; x < sizes[lvl][0]; x++)
            {
                if (get(y, x, lvl) == nodata)
                    nodatacount++;
            }
        }
        return float(nodatacount) / (sizes[lvl][0] * sizes[lvl][1]);
    }

    dataCPU add(dataCPU &other, int lvl)
    {
        dataCPU<Type> result(sizes[0][1], sizes[0][0], nodata);
        for (int y = 0; y < sizes[lvl][1]; y++)
            for (int x = 0; x < sizes[lvl][0]; x++)
            {
                Type p1 = get(y, x, lvl);
                Type p2 = other.get(y, x, lvl);
                if(p1 == nodata || p2 == other.nodata)
                    continue;
                Type res = p1 + p2;
                result.set(res, y, x, lvl);
            }
        return result;
    }

    dataCPU sub(dataCPU &other, int lvl)
    {
        dataCPU<Type> result(sizes[0][1], sizes[0][0], nodata);
        for (int y = 0; y < sizes[lvl][1]; y++)
            for (int x = 0; x < sizes[lvl][0]; x++)
            {
                Type p1 = get(y, x, lvl);
                Type p2 = other.get(y, x, lvl);
                if(p1 == nodata || p2 == other.nodata)
                    continue;
                Type res = p1 - p2;
                result.set(res, y, x, lvl);
            }
        return result;
    }

    Type nodata;

private:
    inline Type bilinear(float y, float x, int lvl)
    {
        // bilinear interpolation (-2 because the read the next pixel)
        // int _x = std::min(std::max(int(x), 0), texture[lvl].cols-2);
        // int _y = std::min(std::max(int(y), 0), texture[lvl].rows-2);
        if (y > sizes[lvl][1] - 2 || x > sizes[lvl][0] - 2)
            return get(int(y), int(x), lvl);

        int _x = int(x);
        int _y = int(y);
        float dx = x - _x;
        float dy = y - _y;

        float weight_tl = (1.0 - dx) * (1.0 - dy);
        float weight_tr = (dx) * (1.0 - dy);
        float weight_bl = (1.0 - dx) * (dy);
        float weight_br = (dx) * (dy);

        Type pix = weight_tl * get(_y, _x, lvl) +
                   weight_tr * get(_y, _x + 1, lvl) +
                   weight_bl * get(_y + 1, _x, lvl) +
                   weight_br * get(_y + 1, _x + 1, lvl);

        return pix;
    }

    Type area(int y, int x, int lvl)
    {
        // bilinear interpolation (-2 because the read the next pixel)
        // int _x = std::min(std::max(int(x), 0), texture[lvl].cols-2);
        // int _y = std::min(std::max(int(y), 0), texture[lvl].rows-2);
        if (y > sizes[lvl][1] - 2 || x > sizes[lvl][0] - 2)
            return get(y, x, lvl);

        Type pix = (get(y, x, lvl) +
                    get(y, x + 1, lvl) +
                    get(y + 1, x, lvl) +
                    get(y + 1, x + 1, lvl)) /
                   4.0;

        return pix;
    }

    Type *texture[MAX_LEVELS];
    int max_lvl;
    std::array<int, 2> sizes[MAX_LEVELS];
};
