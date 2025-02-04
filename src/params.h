#pragma once

namespace mesh_vo
{
    inline constexpr int image_width = 512;
    inline constexpr int image_height = 512;
    inline constexpr int mesh_width = 32;
    inline constexpr int mesh_height = 32;
    inline constexpr int max_vertex_size = mesh_width * mesh_height;
    inline constexpr int max_triangle_size = (mesh_width - 1) * (mesh_height - 1) * 2;
    inline constexpr int num_frames = 3;
    inline constexpr int renderer_nthreads = 1;
    inline constexpr int reducer_nthreads = 1;

    extern float last_min_angle;
    extern float key_max_angle;
    extern float min_view_perc;
    extern float min_lambda;
    extern float huber_thresh_pix;
    extern float regu_weight;
    extern float prior_weight;
    // for a max depth of 10, using log(depth) as param
    // #define MAX_PARAM fromDepthToParam(1.0 + 0.5)
    // #define MIN_PARAM fromDepthToParam(1.0 - 0.5)
    extern float tracking_pose_var;
    extern float mapping_pose_var;
    // #define INITIAL_PARAM_STD fromDepthToParam(0.01)
    // for a param = log(depth) and a max depth of 10, the max uncertanty should be ln(3) = 1.098
    extern float initial_param_var;
    // lets suppouse that by the optimization we reduce the uncertainty by 10x, that means a depth uncertainty of exp(0.1*1.098) = 1.116, or for 2x reduction, that means 1.731 uncertanty
    extern float good_param_var;
}
