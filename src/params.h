#pragma once

namespace mesh_vo
{
    //inline constexpr int image_width = 512;
    //inline constexpr int image_height = 512;
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

    extern float tracking_prior_weight;
    extern float tracking_convergence_p;
    extern float tracking_convergence_v;
    extern float tracking_pose_initial_var;
    extern float tracking_ini_lvl;
    extern float tracking_fin_lvl;

    extern float mapping_regu_weight;
    extern float mapping_prior_weight;
    extern float mapping_convergence_p;
    extern float mapping_convergence_v;
    extern float mapping_intrinsic_initial_var;
    extern float mapping_intrinsic_good_var;
    extern float mapping_pose_initial_var;
    extern float mapping_pose_good_var;
    extern float mapping_param_initial_var;
    extern float mapping_param_good_var;
}
