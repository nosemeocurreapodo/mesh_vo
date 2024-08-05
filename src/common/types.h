#pragma once

#include <Eigen/Core>
#include "sophus/se3.hpp"

template <typename type>
struct vec2
{
    vec2()
    {
    }

    template <typename type2>
    vec2(type2 &x, type2 &y)
    {
        this->operator()(0) = (type)x;
        this->operator()(1) = (type)y;
    }

    vec2<type> operator+(vec2<type> c)
    {
        vec2<type> result;
        result(0) = type(this->operator()(0) + c(0));
        result(1) = type(this->operator()(1) + c(1));
        return result;
    }

    vec2<type> operator-(vec2<type> c)
    {
        vec2<type> result;
        result(0) = type(this->operator()(0) - c(0));
        result(1) = type(this->operator()(1) - c(1));
        return result;
    }

    type dot(vec2<type> c)
    {
        return this->operator()(0)*c(0) + this->operator()(1)*c(1);
    }

    void operator=(vec2<type> c)
    {
        this->operator()(0) = c(0);
        this->operator()(1) = c(1);
    }

    inline type &operator()(int c)
    {
        return data[c];
    }

    inline type operator()(int c) const
    {
        return data[c];
    }

private:
    type data[2];
};

template <typename type>
struct mat3;

template <typename type>
struct vec3
{
    vec3()
    {
    }

    vec3(type x, type y, type z)
    {
        set(x, y, z);
    }

    void set(type x, type y, type z)
    {
        this->operator()(0) = x;
        this->operator()(1) = y;
        this->operator()(2) = z;
    }

    type dot(vec3<type> &a)
    {
        type result = this->operator()(0) * a(0) + this->operator()(1) * a(1) + this->operator()(2) * a(2);
        return result;
    }

    vec3<type> cross(vec3<type> a)
    {
        vec3<type> result;
        result(0) = this->operator()(1)*a(2) - this->operator()(2)*a(1);
        result(1) = this->operator()(2)*a(0) - this->operator()(0)*a(2);
        result(2) = this->operator()(0)*a(1) - this->operator()(1)*a(0);
        return result;
    }

    mat3<type> outer(vec3<type> a)
    {
        mat3<type> result;
        // for (int y = 0; y < 3; y++)
        //   for (int x = 0; x < 3; x++)
        //     result(y, x) = data[y] * a(x);

        result(0, 0) = this->operator()(0) * a(0);
        result(0, 1) = this->operator()(0) * a(1);
        result(0, 2) = this->operator()(0) * a(2);

        result(1, 0) = this->operator()(1) * a(0);
        result(1, 1) = this->operator()(1) * a(1);
        result(1, 2) = this->operator()(1) * a(2);

        result(2, 0) = this->operator()(2) * a(0);
        result(2, 1) = this->operator()(2) * a(1);
        result(2, 2) = this->operator()(2) * a(2);

        return result;
    }

    type norm()
    {
        return sqrt(this->operator()(0) * this->operator()(0) + this->operator()(1) * this->operator()(1) + this->operator()(2) * this->operator()(2));
    }

    template <typename type2>
    vec3<type> operator*(type2 c)
    {
        vec3<type> result;
        result(0) = type(this->operator()(0) * c);
        result(1) = type(this->operator()(1) * c);
        result(2) = type(this->operator()(2) * c);
        return result;
    }

    template <typename type2>
    vec3<type> operator/(type2 c)
    {
        vec3<type> result;
        result(0) = type(this->operator()(0) / c);
        result(1) = type(this->operator()(1) / c);
        result(2) = type(this->operator()(2) / c);
        return result;
    }

    vec3<type> operator+(vec3<type> c)
    {
        vec3<type> result;
        result(0) = type(this->operator()(0) + c(0));
        result(1) = type(this->operator()(1) + c(1));
        result(2) = type(this->operator()(2) + c(2));
        return result;
    }

    vec3<type> operator-(vec3<type> c)
    {
        vec3<type> result;
        result(0) = type(this->operator()(0) - c(0));
        result(1) = type(this->operator()(1) - c(1));
        result(2) = type(this->operator()(2) - c(2));
        return result;
    }

    void operator=(vec3<type> c)
    {
        this->operator()(0) = c(0);
        this->operator()(1) = c(1);
        this->operator()(2) = c(2);
    }

    inline type &operator()(int c)
    {
        return data[c];
    }

    inline type operator()(int c) const
    {
        return data[c];
    }

private:
    type data[3];
};

template <typename type>
struct mat6;

template <typename type>
struct vec6
{
    vec6()
    {
    }

    vec6(type x, type y, type z, type a, type b, type c)
    {
        this->operator()(0) = x;
        this->operator()(1) = y;
        this->operator()(2) = z;
        this->operator()(3) = a;
        this->operator()(4) = b;
        this->operator()(5) = c;
    }

    void zero()
    {
        this->operator()(0) = type(0.0);
        this->operator()(1) = type(0.0);
        this->operator()(2) = type(0.0);
        this->operator()(3) = type(0.0);
        this->operator()(4) = type(0.0);
        this->operator()(5) = type(0.0);
    }

    type dot(vec6<type> b)
    {
        type result;
        /*
        for (int x = 0; x < 6; x++)
        {
          result += data[x] * b(x);
        }
        */
        result = this->operator()(0) * b(0) + this->operator()(1) * b(1) + this->operator()(2) * b(2) + this->operator()(3) * b(3) + this->operator()(4) * b(4) + this->operator()(5) * b(5);
        return result;
    }

    vec6<type> operator+(vec6<type> c)
    {
        vec6<type> result;
        result(0) = this->operator()(0) + c(0);
        result(1) = this->operator()(1) + c(1);
        result(2) = this->operator()(2) + c(2);
        result(3) = this->operator()(3) + c(3);
        result(4) = this->operator()(4) + c(4);
        result(5) = this->operator()(5) + c(5);
        return result;
    }

    template <typename type2>
    vec6 operator/(type2 c)
    {
        vec6 result;
        result(0) = type(this->operator()(0) / c);
        result(1) = type(this->operator()(1) / c);
        result(2) = type(this->operator()(2) / c);
        result(3) = type(this->operator()(3) / c);
        result(4) = type(this->operator()(4) / c);
        result(5) = type(this->operator()(5) / c);
        return result;
    }

    template <typename type2>
    vec6 operator*(type2 c)
    {
        vec6 result;
        result(0) = this->operator()(0) * c;
        result(1) = this->operator()(1) * c;
        result(2) = this->operator()(2) * c;
        result(3) = this->operator()(3) * c;
        result(4) = this->operator()(4) * c;
        result(5) = this->operator()(5) * c;
        return result;
    }

    mat6<type> outer(mat6<type> a)
    {
        mat6<type> result;

        for (int y = 0; y < 6; y++)
        {
            for (int x = 0; x < 6; x++)
                result(y, x) = this->operator()(y) * a(x);
        }
        return result;
    }

    void operator=(vec6<type> c)
    {
        this->operator()(0) = c(0);
        this->operator()(1) = c(1);
        this->operator()(2) = c(2);
        this->operator()(3) = c(3);
        this->operator()(4) = c(4);
        this->operator()(5) = c(5);
    }

    inline type &operator()(int c)
    {
        return data[c];
    }

    inline type operator()(int c) const
    {
        return data[c];
    }

private:
    type data[6];
};

template <typename type>
struct mat3
{
    mat3()
    {
    }

    mat3(type d00, type d01, type d02, type d10, type d11, type d12, type d20, type d21, type d22)
    {
        this->operator()(0, 0) = d00;
        this->operator()(0, 1) = d01;
        this->operator()(0, 2) = d02;
        this->operator()(1, 0) = d10;
        this->operator()(1, 1) = d11;
        this->operator()(1, 2) = d12;
        this->operator()(2, 0) = d20;
        this->operator()(2, 1) = d21;
        this->operator()(2, 2) = d22;
    }

    void zero()
    {
        /*
        for (int y = 0; y < 3; y++)
          for (int x = 0; x < 3; x++)
          {
            data[y][x] = 0.0;
          }
          */

        this->operator()(0, 0) = 0.0;
        this->operator()(0, 1) = 0.0;
        this->operator()(0, 2) = 0.0;

        this->operator()(1, 0) = 0.0;
        this->operator()(1, 1) = 0.0;
        this->operator()(1, 2) = 0.0;

        this->operator()(2, 0) = 0.0;
        this->operator()(2, 1) = 0.0;
        this->operator()(2, 2) = 0.0;
    }

    void identity()
    {
        /*
        for (int y = 0; y < 3; y++)
          for (int x = 0; x < 3; x++)
          {
            if (x == y)
              data[y][x] = 1.0;
            else
              data[y][x] = 0.0;
          }
          */

        this->operator()(0, 0) = 1.0;
        this->operator()(0, 1) = 0.0;
        this->operator()(0, 2) = 0.0;

        this->operator()(1, 0) = 0.0;
        this->operator()(1, 1) = 1.0;
        this->operator()(1, 2) = 0.0;

        this->operator()(2, 0) = 0.0;
        this->operator()(2, 1) = 0.0;
        this->operator()(2, 2) = 1.0;
    }

    mat3 transpose()
    {
        mat3<type> result;
        result(0, 0) = this->operator()(0, 0);
        result(0, 1) = this->operator()(1, 0);
        result(0, 2) = this->operator()(2, 0);
        result(1, 0) = this->operator()(0, 1);
        result(1, 1) = this->operator()(1, 1);
        result(1, 2) = this->operator()(2, 1);
        result(2, 0) = this->operator()(0, 2);
        result(2, 1) = this->operator()(1, 2);
        result(2, 2) = this->operator()(2, 2);
        return result;
    }

    /*
        mat3 inverse()
        {
        double determinant =    +A(0,0)*(A(1,1)*A(2,2)-A(2,1)*A(1,2))
                            -A(0,1)*(A(1,0)*A(2,2)-A(1,2)*A(2,0))
                            +A(0,2)*(A(1,0)*A(2,1)-A(1,1)*A(2,0));
    double invdet = 1/determinant;
    result(0,0) =  (A(1,1)*A(2,2)-A(2,1)*A(1,2))*invdet;
    result(1,0) = -(A(0,1)*A(2,2)-A(0,2)*A(2,1))*invdet;
    result(2,0) =  (A(0,1)*A(1,2)-A(0,2)*A(1,1))*invdet;
    result(0,1) = -(A(1,0)*A(2,2)-A(1,2)*A(2,0))*invdet;
    result(1,1) =  (A(0,0)*A(2,2)-A(0,2)*A(2,0))*invdet;
    result(2,1) = -(A(0,0)*A(1,2)-A(1,0)*A(0,2))*invdet;
    result(0,2) =  (A(1,0)*A(2,1)-A(2,0)*A(1,1))*invdet;
    result(1,2) = -(A(0,0)*A(2,1)-A(2,0)*A(0,1))*invdet;
    result(2,2) =  (A(0,0)*A(1,1)-A(1,0)*A(0,1))*invdet;
        }
        */

    template <typename type2>
    mat3 operator/(type2 c)
    {
        mat3<type> result;
        /*
        for (int y = 0; y < 3; y++)
          for (int x = 0; x < 3; x++)
          {
            result.data[y][x] = data[y][x] / c;
          }
          */
        result(0, 0) = this->operator()(0, 0) / c;
        result(0, 1) = this->operator()(0, 1) / c;
        result(0, 2) = this->operator()(0, 2) / c;

        result(1, 0) = this->operator()(1, 0) / c;
        result(1, 1) = this->operator()(1, 1) / c;
        result(1, 2) = this->operator()(1, 2) / c;

        result(2, 0) = this->operator()(2, 0) / c;
        result(2, 1) = this->operator()(2, 1) / c;
        result(2, 2) = this->operator()(2, 2) / c;

        return result;
    }

    template <typename type2>
    mat3 operator*(type2 c)
    {
        mat3<type> result;
        /*
        for (int y = 0; y < 3; y++)
          for (int x = 0; x < 3; x++)
          {
            result.data[y][x] = data[y][x] * c;
          }
          */

        result(0, 0) = this->operator()(0, 0) * c;
        result(0, 1) = this->operator()(0, 1) * c;
        result(0, 2) = this->operator()(0, 2) * c;

        result(1, 0) = this->operator()(1, 0) * c;
        result(1, 1) = this->operator()(1, 1) * c;
        result(1, 2) = this->operator()(1, 2) * c;

        result(2, 0) = this->operator()(2, 0) * c;
        result(2, 1) = this->operator()(2, 1) * c;
        result(2, 2) = this->operator()(2, 2) * c;

        return result;
    }

    mat3 dot(mat3 c)
    {
        mat3<type> result;
        /*
        for (int y = 0; y < 3; y++)
          for (int x = 0; x < 3; x++)
          {
            result.data[y][x] = data[y][0] * c(0, x) + data[y][1] * c(1, x) + data[y][2] * c(2, x);
          }
          */

        result(0, 0) = this->operator()(0, 0) * c(0, 0) + this->operator()(0, 1) * c(1, 0) + this->operator()(0, 2) * c(2, 0);
        result(0, 1) = this->operator()(0, 0) * c(0, 1) + this->operator()(0, 1) * c(1, 1) + this->operator()(0, 2) * c(2, 1);
        result(0, 2) = this->operator()(0, 0) * c(0, 2) + this->operator()(0, 1) * c(1, 2) + this->operator()(0, 2) * c(2, 2);

        result(1, 0) = this->operator()(1, 0) * c(0, 0) + this->operator()(1, 1) * c(1, 0) + this->operator()(1, 2) * c(2, 0);
        result(1, 1) = this->operator()(1, 0) * c(0, 1) + this->operator()(1, 1) * c(1, 1) + this->operator()(1, 2) * c(2, 1);
        result(1, 2) = this->operator()(1, 0) * c(0, 2) + this->operator()(1, 1) * c(1, 2) + this->operator()(1, 2) * c(2, 2);

        result(2, 0) = this->operator()(2, 0) * c(0, 0) + this->operator()(2, 1) * c(1, 0) + this->operator()(2, 2) * c(2, 0);
        result(2, 1) = this->operator()(2, 0) * c(0, 1) + this->operator()(2, 1) * c(1, 1) + this->operator()(2, 2) * c(2, 1);
        result(2, 2) = this->operator()(2, 0) * c(0, 2) + this->operator()(2, 1) * c(1, 2) + this->operator()(2, 2) * c(2, 2);

        return result;
    }

    vec3<type> dot(vec3<type> c) const
    {
        vec3<type> result;
        /*
        for (int y = 0; y < 3; y++)
          result(y) = data[y][0] * c(0) + data[y][1] * c(1) + data[y][2] * c(2);
          */
        result(0) = this->operator()(0, 0) * c(0) + this->operator()(0, 1) * c(1) + this->operator()(0, 2) * c(2);
        result(1) = this->operator()(1, 0) * c(0) + this->operator()(1, 1) * c(1) + this->operator()(1, 2) * c(2);
        result(2) = this->operator()(2, 0) * c(0) + this->operator()(2, 1) * c(1) + this->operator()(2, 2) * c(2);

        return result;
    }

    void operator=(mat3<type> c)
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                this->operator()(i, j) = c(i, j);
            }
        }
    }

    inline type &operator()(int b, int c)
    {
        return data[b][c];
    }

    inline type operator()(int b, int c) const
    {
        return data[b][c];
    }

private:
    type data[3][3];
};

template <typename type>
struct mat6
{
    mat6()
    {
    }

    mat6(type *data)
    {
        this->operator()(0, 0) = data[0];
        this->operator()(0, 1) = data[1];
        this->operator()(0, 2) = data[2];
        this->operator()(0, 3) = data[3];
        this->operator()(0, 4) = data[4];
        this->operator()(0, 5) = data[5];

        this->operator()(1, 0) = data[6];
        this->operator()(1, 1) = data[7];
        this->operator()(1, 2) = data[8];
        this->operator()(1, 3) = data[9];
        this->operator()(1, 4) = data[10];
        this->operator()(1, 5) = data[11];

        this->operator()(2, 0) = data[12];
        this->operator()(2, 1) = data[13];
        this->operator()(2, 2) = data[14];
        this->operator()(2, 3) = data[15];
        this->operator()(2, 4) = data[16];
        this->operator()(2, 5) = data[17];

        this->operator()(3, 0) = data[18];
        this->operator()(3, 1) = data[19];
        this->operator()(3, 2) = data[20];
        this->operator()(3, 3) = data[21];
        this->operator()(3, 4) = data[22];
        this->operator()(3, 5) = data[23];

        this->operator()(4, 0) = data[24];
        this->operator()(4, 1) = data[25];
        this->operator()(4, 2) = data[26];
        this->operator()(4, 3) = data[27];
        this->operator()(4, 4) = data[28];
        this->operator()(4, 5) = data[29];

        this->operator()(5, 0) = data[30];
        this->operator()(5, 1) = data[31];
        this->operator()(5, 2) = data[32];
        this->operator()(5, 3) = data[33];
        this->operator()(5, 4) = data[34];
        this->operator()(5, 5) = data[35];
    }

    void zero()
    {

        for (int y = 0; y < 6; y++)
        {

            for (int x = 0; x < 6; x++)
            {
                this->operator()(y, x) = 0.0;
            }
        }
    }

    void identity()
    {

        for (int y = 0; y < 6; y++)
        {

            for (int x = 0; x < 6; x++)
            {
                if (x == y)
                    this->operator()(y, x) = 1.0;
                else
                    this->operator()(y, x) = 0.0;
            }
        }
    }

    template <typename type2>
    mat6 operator/(type2 c)
    {
        mat6<type> result;
        for (int y = 0; y < 6; y++)
            for (int x = 0; x < 6; x++)
            {
                result(y, x) = this->operator()(y, x) / c;
            }
        return result;
    }

    template <typename type2>
    mat6 operator*(type2 c)
    {
        mat6<type> result;
        for (int y = 0; y < 6; y++)
            for (int x = 0; x < 6; x++)
            {
                result(y, x) = type(this->operator()(y, x) * c);
            }
        return result;
    }

    mat6 operator+(mat6 c)
    {
        mat6<type> result;

        for (int y = 0; y < 6; y++)
        {

            for (int x = 0; x < 6; x++)
            {
                result(y, x) = this->operator()(y, x) + c(y, x);
            }
        }
        return result;
    }

    mat6 dot(mat6 c)
    {
        mat6<type> result;
        for (int y = 0; y < 6; y++)
            for (int x = 0; x < 6; x++)
            {
                for (int z = 0; z < 6; z++)
                    result(y, x) += this->operator()(y, z) * c(z, y);
            }

        return result;
    }

    vec6<type> dot(vec6<type> c)
    {
        vec6<type> result;

        for (int y = 0; y < 6; y++)
        {

            for (int x = 0; x < 6; x++)
            {
                result(y) += this->operator()(y, x) * c(x);
            }
        }
        return result;
    }

    void operator=(mat6<type> c)
    {

        for (int i = 0; i < 6; i++)
        {

            for (int j = 0; j < 6; j++)
            {
                this->operator()(i, j) = c(i, j);
            }
        }
    }

    inline type &operator()(int b, int c)
    {
        return data[b][c];
    }

    inline type operator()(int b, int c) const
    {
        return data[b][c];
    }

private:
    type data[6][6];
};

template <typename type>
struct SO3
{
    SO3()
    {
        identity();
    }

    inline SO3(type qw, type qx, type qy, type qz)
    {
        fromQuaternion(qw, qx, qy, qz);
    }

    inline SO3(mat3<type> mat)
    {
        matrix = mat;
    }

    inline SO3(type *r)
    {
        matrix(0, 0) = r[0];
        matrix(0, 1) = r[1];
        matrix(0, 2) = r[2];
        matrix(1, 0) = r[3];
        matrix(1, 1) = r[4];
        matrix(1, 2) = r[5];
        matrix(2, 0) = r[6];
        matrix(2, 1) = r[7];
        matrix(2, 2) = r[8];
    }

    inline void fromQuaternion(type qw, type qx, type qy, type qz)
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
    inline SE3(cv::Mat_<float> r, cv::vec3f t)
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

    inline SO3<type> inverse() const
    {
        return matrix.transpose();
    }

    void operator=(SO3<type> c)
    {
        matrix = c.matrix;
    }

    inline type operator()(int r, int c) const
    {
        return matrix(r, c);
    }

    inline type &operator()(int r, int c)
    {
        return matrix(r, c);
    }

    inline vec3<type> dot(const vec3<type> &p) const
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
    inline SE3(type qw, type qx, type qy, type qz, type tx, type ty, type tz)
    {
        rotation.fromQuaternion(qw, qx, qy, qz);
        translation.set(tx, ty, tz);
    }

    /// Construct from C arrays
    /// r is rotation matrix row major
    /// t is the translation vector (x y z)
    inline SE3(type *r, type *t)
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

    inline SE3(SO3<type> _rotation, vec3<type> _translation)
    {
        rotation = _rotation;
        translation = _translation;
    }

    /*
    inline SE3(cv::Mat_<float> r, cv::vec3f t)
    {
      data(0,0)=r(0,0); data(0,1)=r(0,1); data(0,2)=r(0,2); data(0,3)=t.x;
      data(1,0)=r(1,0); data(1,1)=r(1,1); data(1,2)=r(1,2); data(1,3)=t.y;
      data(2,0)=r(2,0); data(2,1)=r(2,1); data(2,2)=r(2,2); data(2,3)=t.z;
    }
    */

    void identity()
    {
        rotation.identity();
        translation.set(0.0, 0.0, 0.0);
    }

    inline SE3<type> inv() const
    {
        SE3<type> result;
        result.rotation = rotation.inv();
        result.translation = -result.rotation.dot(translation);
        return result;
    }

    /*
        inline type this->operator()()(int r, int c) const
        {
          return data[r][c];
        }

        inline type &this->operator()()(int r, int c)
        {
          return data[r][c];
        }
        */

    inline SE3<type> dot(const SE3<type> &rhs)
    {
        SE3<type> result;
        result.rotation = rotation.dot(rhs.rotation);
        result.translation = rotation.dot(rhs.translation) + translation;
        return result;
    }

    inline vec3<type> dot(const vec3<type> &rhs)
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

