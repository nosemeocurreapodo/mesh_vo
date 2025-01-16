#pragma once

#include <cmath>
#include <cassert>

template <typename type, int _rows, int _cols>
struct mat;

template <typename type, int _rows>
struct vec
{
    vec()
    {
        setZero();
    }

    vec(const vec &other)
    {
        for (int i = 0; i < _rows; i++)
            data[i] = other.data[i];
    }

    void setZero()
    {
        for (int i = 0; i < _rows; i++)
            data[i] = 0;
    }

    static constexpr vec Zero()
    {
        vec v;
        for (int i = 0; i < _rows; i++)
            v(i) = 0;
        return v;
    }

    static constexpr int rows()
    {
        return _rows;
    }

    static constexpr int cols()
    {
        return 1;
    }

    static constexpr int size()
    {
        return _rows;
    }

    type norm()
    {
        type sum = 0;
        for (int i = 0; i < _rows; i++)
            sum += data[i] * data[i];
        return sqrt(sum);
    }

    template <typename type2>
    vec operator*(type2 c)
    {
        vec result;
        for (int i = 0; i < _rows; i++)
            result(i) = data[i] * c;
        return result;
    }

    template <typename type2>
    vec operator/(type2 c)
    {
        vec result;
        for (int i = 0; i < _rows; i++)
            result(i) = data[i] / c;
        return result;
    }

    vec operator+(vec c)
    {
        vec result;
        for (int i = 0; i < _rows; i++)
            result(i) = data[i] + c(i);
        return result;
    }

    vec operator-(vec c)
    {
        vec result;
        for (int i = 0; i < _rows; i++)
            result(i) = data[i] - c(i);
        return result;
    }

    type dot(vec c)
    {
        type sum = 0;
        for (int i = 0; i < _rows; i++)
            sum += data[i] * c(i);
        return sum;
    }

    mat<type, _rows, _rows> outer(vec a)
    {
        mat<type, _rows, _rows> result;
        for (int y = 0; y < _rows; y++)
            for (int x = 0; x < _rows; x++)
                result(y, x) = data[y] * a(x);

        return result;
    }

    bool operator==(vec c)
    {
        bool equal = true;
        for (int i = 0; i < _rows; i++)
            if (data[i] != c(i))
                equal = false;
        return equal;
    }

    vec &operator=(const vec &other)
    {
        if (this != &other)
        {
            for (int i = 0; i < _rows; i++)
                data[i] = other.data[i];
        }
        return *this;
    }

    type &operator()(int c)
    {
        return data[c];
    }

    type operator()(int c) const
    {
        return data[c];
    }

protected:
    type data[_rows];
};

template <typename type>
struct vecx
{
    vecx()
    {
        data = nullptr;
        _rows = 0;
    }

    vecx(int size)
    {
        _rows = size;
        data = new type[_rows];

        setZero();
    }

    vecx(const vecx &other)
    {
        _rows = other.rows();
        data = new type[_rows];

        for (int i = 0; i < _rows; i++)
            data[i] = other.data[i];
    }

    void setZero()
    {
        for (int i = 0; i < _rows; i++)
            data[i] = 0;
    }

    static constexpr vecx Zero(int _rows)
    {
        vecx v(_rows);
        for (int i = 0; i < _rows; i++)
            v(i) = 0;
        return v;
    }

    int rows() const
    {
        return _rows;
    }

    static constexpr int cols()
    {
        return 1;
    }

    int size()
    {
        return _rows;
    }

    type dot(vecx c)
    {
        type sum = 0;
        for (int i = 0; i < _rows; i++)
            sum += data[i] * c(i);
        return sum;
    }

    vecx &operator=(const vecx &other)
    {
        if (this != &other)
        {
            if (data != nullptr)
                delete[] data;

            _rows = other.rows();
            data = new type[_rows];

            for (int i = 0; i < _rows; i++)
                data[i] = other.data[i];
        }
        return *this;
    }

    template <typename type2>
    vecx operator*(type2 c)
    {
        vecx result = vecx(_rows);

        for (int i = 0; i < _rows; i++)
        {
            result(i) = data[i] * c;
        }
        return result;
    }

    type &operator()(int c)
    {
        return data[c];
    }

    type operator()(int c) const
    {
        return data[c];
    }

private:
    type *data;
    int _rows;
};

template <typename type, int _rows, int _cols>
struct mat
{
    mat()
    {
        setZero();
    }

    void setZero()
    {
        for (int y = 0; y < _rows; y++)
            for (int x = 0; x < _cols; x++)
            {
                data[y][x] = 0.0;
            }
    }

    static constexpr mat<type, _rows, _cols> Zero()
    {
        mat<type, _rows, _cols> result;

        for (int y = 0; y < _rows; y++)
            for (int x = 0; x < _cols; x++)
            {
                result.data[y][x] = 0.0;
            }
        return result;
    }

    static constexpr int rows()
    {
        return _rows;
    }

    static constexpr int cols()
    {
        return _cols;
    }

    static constexpr int size()
    {
        return _rows * _cols;
    }

    static constexpr mat<type, _rows, _cols> identity()
    {
        mat<type, _rows, _cols> result;

        for (int y = 0; y < _rows; y++)
            for (int x = 0; x < _cols; x++)
            {
                if (x == y)
                    result.data[y][x] = 1.0;
                else
                    result.data[y][x] = 0.0;
            }

        return result;
    }

    mat<type, _cols, _rows> transpose()
    {
        mat<type, _cols, _rows> result;

        for (int y = 0; y < _rows; y++)
        {
            for (int x = 0; x < _cols; x++)
            {
                result.data[x][y] = data[y][x];
            }
        }

        return result;
    }

    mat operator/(type c)
    {
        mat result;

        for (int y = 0; y < _rows; y++)
            for (int x = 0; x < _cols; x++)
            {
                result.data[y][x] = data[y][x] / c;
            }

        return result;
    }

    mat<type, _rows, _cols> operator*(type c)
    {
        mat<type, _rows, _cols> result;

        for (int y = 0; y < _rows; y++)
            for (int x = 0; x < _cols; x++)
            {
                result.data[y][x] = data[y][x] * c;
            }

        return result;
    }

    mat<type, _rows, _cols> operator*(mat<type, _rows, _cols> c)
    {
        mat<type, _rows, _cols> result;

        for (int y = 0; y < _rows; y++)
            for (int x = 0; x < _cols; x++)
            {
                for (int z = 0; z < _cols; z++)
                {
                    result.data[y][x] += data[y][z] * c(z, x);
                }
            }

        return result;
    }

    vec<type, _rows> operator*(vec<type, _rows> c)
    {
        vec<type, _rows> result;

        for (int y = 0; y < _rows; y++)
            for (int z = 0; z < _cols; z++)
                result(y) += data[y][z] * c(z);

        return result;
    }

    void operator=(mat c)
    {
        for (int i = 0; i < _rows; i++)
        {
            for (int j = 0; j < _cols; j++)
            {
                data[i][j] = c(i, j);
            }
        }
    }

    type &operator()(int b, int c)
    {
        return data[b][c];
    }

    type operator()(int b, int c) const
    {
        return data[b][c];
    }

protected:
    type data[_rows][_cols];
};

template <typename type>
struct matx
{
    matx()
    {
        data = nullptr;
        _rows = 0;
        _cols = 0;
    }

    matx(int __rows, int __cols)
    {
        _rows = __rows;
        _cols = __cols;
        data = new type[_rows * _cols];

        setZero();
    }

    matx(const matx &other)
    {
        _rows = other.rows();
        _cols = other.cols();

        data = new type[_rows * _cols];

        for (int y = 0; y < _rows; y++)
            for (int x = 0; x < _cols; x++)
                data[y + _rows * x] = other(y, x);
    }

    void setZero()
    {
        for (int y = 0; y < _rows; y++)
            for (int x = 0; x < _cols; x++)
            {
                data[y + _rows * x] = 0.0;
            }
    }

    static constexpr matx Zero(int __rows, int __cols)
    {
        matx v(__rows, __cols);
        for (int y = 0; y < __rows; y++)
            for (int x = 0; x < __cols; x++)
                v(y, x) = 0;
        return v;
    }

    int rows() const
    {
        return _rows;
    }

    int cols() const
    {
        return _cols;
    }

    int size()
    {
        return _rows * _cols;
    }

    matx transpose()
    {
        matx result(_cols, _rows);

        for (int y = 0; y < _rows; y++)
        {
            for (int x = 0; x < _cols; x++)
            {
                result(x, y) = data[y + _rows * x];
            }
        }

        return result;
    }

    matx operator*(matx c)
    {
        assert(_cols == c.rows());
        assert(_rows == c.cols());

        matx result(_rows, c.cols());

        for (int y = 0; y < _rows; y++)
            for (int x = 0; x < c.cols(); x++)
            {
                for (int z = 0; z < _cols; z++)
                {
                    result(y, x) += data[y + _rows * z] * c(z, x);
                }
            }

        return result;
    }

    vecx<type> operator*(vecx<type> c)
    {
        vecx<type> result(_rows);

        for (int y = 0; y < _rows; y++)
            for (int z = 0; z < _cols; z++)
                result(y) += data[y + _rows * z] * c(z);

        return result;
    }

    matx &operator=(const matx &other)
    {
        if (this != &other)
        {
            if (data != nullptr)
                delete[] data;

            _rows = other.rows();
            _cols = other.cols();

            data = new type[_rows * _cols];

            for (int y = 0; y < _rows; y++)
                for (int x = 0; x < _cols; x++)
                    data[y + _rows * x] = other(y, x);
        }
        return *this;
    }

    matx operator*(type c)
    {
        matx result = matx(_rows, _cols);

        for (int y = 0; y < _rows; y++)
            for (int x = 0; x < _cols; x++)
                result(y, x) = data[y + _rows * x] * c;

        return result;
    }

    type &operator()(int c, int d)
    {
        return data[c + _rows * d];
    }

    type operator()(int c, int d) const
    {
        return data[c + _rows * d];
    }

private:
    type *data;
    int _rows;
    int _cols;
};

template <typename type>
struct vec1 : public vec<type, 1>
{
    vec1() : vec<type, 1>()
    {
    }

    vec1(type x)
    {
        vec<type, 1>::data[0] = x;
    }
};

template <typename type>
struct vec2 : public vec<type, 2>
{
    vec2() : vec<type, 2>()
    {
    }

    vec2(const vec<type, 2> &other) : vec<type, 2>(other)
    {
    }

    vec2(type x, type y)
    {
        vec<type, 2>::data[0] = x;
        vec<type, 2>::data[1] = y;
    }
};

template <typename type>
struct mat3;

template <typename type>
struct vec3 : public vec<type, 3>
{
    vec3() : vec<type, 3>()
    {
    }

    vec3(const vec<type, 3> &other) : vec<type, 3>(other)
    {
    }

    vec3(type x, type y, type z)
    {
        vec<type, 3>::data[0] = x;
        vec<type, 3>::data[1] = y;
        vec<type, 3>::data[2] = z;
    }

    vec3<type> cross(vec3<type> a)
    {
        vec3<type> result;
        result(0) = vec<type, 3>::data[1] * a(2) - vec<type, 3>::data[2] * a(1);
        result(1) = vec<type, 3>::data[2] * a(0) - vec<type, 3>::data[0] * a(2);
        result(2) = vec<type, 3>::data[0] * a(1) - vec<type, 3>::data[1] * a(0);
        return result;
    }
};

template <typename type>
struct mat6;

template <typename type>
struct vec6 : public vec<type, 6>
{
    vec6() : vec<type, 6>()
    {
    }

    vec6(const vec<type, 6> &other) : vec<type, 6>(other)
    {
    }

    vec6(type x, type y, type z, type a, type b, type c)
    {
        vec<type, 6>::data[0] = x;
        vec<type, 6>::data[1] = y;
        vec<type, 6>::data[2] = z;
        vec<type, 6>::data[3] = a;
        vec<type, 6>::data[4] = b;
        vec<type, 6>::data[5] = c;
    }
};

template <typename type>
struct mat8;

template <typename type>
struct vec8 : public vec<type, 8>
{
    vec8() : vec<type, 8>()
    {
    }

    vec8(const vec<type, 8> &other) : vec<type, 8>(other)
    {
    }

    vec8(type x, type y, type z, type a, type b, type c, type d, type e)
    {
        vec<type, 8>::data[0] = x;
        vec<type, 8>::data[1] = y;
        vec<type, 8>::data[2] = z;
        vec<type, 8>::data[3] = a;
        vec<type, 8>::data[4] = b;
        vec<type, 8>::data[5] = c;
        vec<type, 8>::data[6] = d;
        vec<type, 8>::data[7] = e;
    }
};

template <typename type>
struct mat3 : public mat<type, 3, 3>
{
    mat3() : mat<type, 3, 3>()
    {
    }

    mat3(const mat<type, 3, 3> &other) : mat<type, 3, 3>(other)
    {
    }

    type determinant()
    {
        return mat<type, 3, 3>::data[0][0] * (mat<type, 3, 3>::data[1][1] * mat<type, 3, 3>::data[2][2] - mat<type, 3, 3>::data[1][2] * mat<type, 3, 3>::data[2][1]) - mat<type, 3, 3>::data[0][1] * (mat<type, 3, 3>::data[1][0] * mat<type, 3, 3>::data[2][2] - mat<type, 3, 3>::data[1][2] * mat<type, 3, 3>::data[2][0]) + mat<type, 3, 3>::data[0][2] * (mat<type, 3, 3>::data[1][0] * mat<type, 3, 3>::data[2][1] - mat<type, 3, 3>::data[1][1] * mat<type, 3, 3>::data[2][0]);
    }

    mat3<type> inverse()
    {
        mat3<type> inv;

        type det = determinant();

        if(std::fabs(det) < 1e-12)
        {
            throw std::runtime_error("Encountered near-zero determinant.");
        }

        double invDet = 1.0 / det;

        inv.data[0][0] = (mat<type, 3, 3>::data[1][1] * mat<type, 3, 3>::data[2][2] - mat<type, 3, 3>::data[1][2] * mat<type, 3, 3>::data[2][1]) * invDet;
        inv.data[0][1] = -(mat<type, 3, 3>::data[0][1] * mat<type, 3, 3>::data[2][2] - mat<type, 3, 3>::data[0][2] * mat<type, 3, 3>::data[2][1]) * invDet;
        inv.data[0][2] = (mat<type, 3, 3>::data[0][1] * mat<type, 3, 3>::data[1][2] - mat<type, 3, 3>::data[0][2] * mat<type, 3, 3>::data[1][1]) * invDet;

        inv.data[1][0] = -(mat<type, 3, 3>::data[1][0] * mat<type, 3, 3>::data[2][2] - mat<type, 3, 3>::data[1][2] * mat<type, 3, 3>::data[2][0]) * invDet;
        inv.data[1][1] = (mat<type, 3, 3>::data[0][0] * mat<type, 3, 3>::data[2][2] - mat<type, 3, 3>::data[0][2] * mat<type, 3, 3>::data[2][0]) * invDet;
        inv.data[1][2] = -(mat<type, 3, 3>::data[0][0] * mat<type, 3, 3>::data[1][2] - mat<type, 3, 3>::data[0][2] * mat<type, 3, 3>::data[1][0]) * invDet;

        inv.data[2][0] = (mat<type, 3, 3>::data[1][0] * mat<type, 3, 3>::data[2][1] - mat<type, 3, 3>::data[1][1] * mat<type, 3, 3>::data[2][0]) * invDet;
        inv.data[2][1] = -(mat<type, 3, 3>::data[0][0] * mat<type, 3, 3>::data[2][1] - mat<type, 3, 3>::data[0][1] * mat<type, 3, 3>::data[2][0]) * invDet;
        inv.data[2][2] = (mat<type, 3, 3>::data[0][0] * mat<type, 3, 3>::data[1][1] - mat<type, 3, 3>::data[0][1] * mat<type, 3, 3>::data[1][0]) * invDet;

        return inv;
    }
};

template <typename type>
struct mat6 : public mat<type, 6, 6>
{
    mat6() : mat<type, 6, 6>()
    {
    }

    mat6(const mat<type, 6, 6> &other) : mat<type, 6, 6>(other)
    {
    }
};

template <typename type>
struct mat8 : public mat<type, 8, 8>
{
    mat8() : mat<type, 8, 8>()
    {
    }

    mat8(const mat<type, 8, 8> &other) : mat<type, 8, 8>(other)
    {
    }
};

template <typename type>
struct SO3
{
    SO3()
    {
        identity();
    }

    SO3(type qw, type qx, type qy, type qz)
    {
        fromQuaternion(qw, qx, qy, qz);
    }

    SO3(mat3<type> mat)
    {
        matrix = mat;
    }

    void fromQuaternion(type qw, type qx, type qy, type qz)
    {
        const type x = 2 * qx;
        const type y = 2 * qy;
        const type z = 2 * qz;
        const type wx = x * qw;
        const type wy = y * qw;
        const type wz = z * qw;
        const type xx = x * qx;
        const type xy = y * qx;
        const type xz = z * qx;
        const type yy = y * qy;
        const type yz = z * qy;
        const type zz = z * qz;

        matrix(0, 0) = 1 - (yy + zz);
        matrix(0, 1) = xy - wz;
        matrix(0, 2) = xz + wy;
        matrix(1, 0) = xy + wz;
        matrix(1, 1) = 1 - (xx + zz);
        matrix(1, 2) = yz - wx;
        matrix(2, 0) = xz - wy;
        matrix(2, 1) = yz + wx;
        matrix(2, 2) = 1 - (xx + yy);
    }

    /*
    SE3(cv::Mat_<float> r, cv::vec3f t)
    {
      data(0,0)=r(0,0); data(0,1)=r(0,1); data(0,2)=r(0,2); data(0,3)=t.x;
      data(1,0)=r(1,0); data(1,1)=r(1,1); data(1,2)=r(1,2); data(1,3)=t.y;
      data(2,0)=r(2,0); data(2,1)=r(2,1); data(2,2)=r(2,2); data(2,3)=t.z;
    }
    */

    void identity()
    {
        matrix.identity();
    }

    SO3<type> inverse() const
    {
        return matrix.transpose();
    }

    void operator=(SO3<type> c)
    {
        matrix = c.matrix;
    }

    type operator()(int r, int c) const
    {
        return matrix(r, c);
    }

    type &operator()(int r, int c)
    {
        return matrix(r, c);
    }

    vec3<type> dot(const vec3<type> &p) const
    {
        vec3<type> result = matrix.dot(p);
        return result;
    }

private:
    mat3<type> matrix;
};

template <typename type>
SO3<type> exp(vec3<type> phi)
{
    type angle = phi.norm();

    // # Near phi==0, use first order Taylor expansion
    //   if np.isclose(angle, 0.):
    //       return cls(np.identity(cls.dim) + cls.wedge(phi))

    vec3<type> axis = phi / angle;
    type s = sin(angle);
    type c = cos(angle);
    mat3<type> mat_result = c * mat3<type>::identity() + (1 - c) * axis.outer(axis) + s * wedge(axis);
    return SO3<type>(mat_result);
}

template <typename type>
mat3<type> wedge(vec3<type> phi)
{
    SO3<type> r;
    r(0, 0) = 0.0;
    r(0, 1) = -phi(2);
    r(0, 2) = phi(1);
    r(1, 0) = phi(2);
    r(1, 1) = 0.0;
    r(1, 2) = -phi(0);
    r(2, 0) = -phi(1);
    r(2, 1) = phi(0);
    r(2, 2) = 0.0;
    return r;
}

template <typename type>
SO3<type> left_jacobian(vec3<type> phi)
{
    type angle = phi.norm();

    // # Near |phi|==0, use first order Taylor expansion
    //  if np.isclose(angle, 0.):
    //     return np.identity(cls.dof) + 0.5 * cls.wedge(phi)

    vec3<type> axis = phi / angle;
    type s = sin(angle);
    type c = cos(angle);

    mat3<type> mat = (s / angle) * mat3<type>::identity() +
                     (1 - s / angle) * axis.outer(axis) +
                     ((1 - c) / angle) * wedge(axis);

    return SO3<type>(mat);
}

template <typename type>
struct SE3
{
    SE3()
    {
        identity();
    }

    /// Constructor from a normalized quaternion and a translation vector
    SE3(type qw, type qx, type qy, type qz, type tx, type ty, type tz)
    {
        rotation.fromQuaternion(qw, qx, qy, qz);
        translation = vec3<float>(tx, ty, tz);
    }

    /// Construct from C arrays
    /// r is rotation matrix row major
    /// t is the translation vector (x y z)
    SE3(type *r, type *t)
    {
        rotation(0, 0) = r[0];
        rotation(0, 1) = r[1];
        rotation(0, 2) = r[2];

        rotation(1, 0) = r[3];
        rotation(1, 1) = r[4];
        rotation(1, 2) = r[5];

        rotation(2, 0) = r[6];
        rotation(2, 1) = r[7];
        rotation(2, 2) = r[8];

        translation(0) = t[0];
        translation(1) = t[1];
        translation(2) = t[2];
    }

    SE3(SO3<type> _rotation, vec3<type> _translation)
    {
        rotation = _rotation;
        translation = _translation;
    }

    /*
    SE3(cv::Mat_<float> r, cv::vec3f t)
    {
      data(0,0)=r(0,0); data(0,1)=r(0,1); data(0,2)=r(0,2); data(0,3)=t.x;
      data(1,0)=r(1,0); data(1,1)=r(1,1); data(1,2)=r(1,2); data(1,3)=t.y;
      data(2,0)=r(2,0); data(2,1)=r(2,1); data(2,2)=r(2,2); data(2,3)=t.z;
    }
    */

    void identity()
    {
        rotation.identity();
        translation = vec3<float>(0.0, 0.0, 0.0);
    }

    SE3<type> inv() const
    {
        SE3<type> result;
        result.rotation = rotation.inv();
        result.translation = -result.rotation.dot(translation);
        return result;
    }

    /*
        type this->operator()()(int r, int c) const
        {
          return data[r][c];
        }

        type &this->operator()()(int r, int c)
        {
          return data[r][c];
        }
        */

    SE3<type> dot(const SE3<type> &rhs)
    {
        SE3<type> result;
        result.rotation = rotation.dot(rhs.rotation);
        result.translation = rotation.dot(rhs.translation) + translation;
        return result;
    }

    vec3<type> dot(const vec3<type> &rhs)
    {
        vec3<type> result = rotation.dot(rhs) + translation;
        return result;
    }
    /*
        void operator=(SE3<type> c)
        {
            rotation = c.rotation;
            translation = c.translation;
        }
        */

private:
    SO3<type> rotation;
    vec3<type> translation;
};

template <typename type>
SE3<type> exp(vec6<type> xi)
{
    vec3<type> rho(xi(0), xi(1), xi(2));
    vec3<type> phi(xi(3), xi(4), xi(5));

    mat3<type> rotation = exp(phi);
    vec3<type> translation = left_jacobian(phi).dot(rho);

    return SE3<type>(rotation, translation);
}
