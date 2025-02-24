#include "params.h"
#include "math.h"

namespace mesh_vo
{
    float last_min_angle = M_PI / 64.0;
    float key_max_angle = M_PI / 15.0;
    float min_view_perc = 0.7;
    float min_lambda = 0.00001f;
    float huber_thresh_pix = 3.0;

    int tracking_ini_lvl = 4;
    int tracking_fin_lvl = 2;
    float tracking_prior_weight = 0.0;
    float tracking_convergence_p = 0.999;
    float tracking_convergence_v = 1e-16;
    float tracking_pose_initial_var = 100.0 * 100.0;

    int mapping_ini_lvl = 2;
    int mapping_fin_lvl = 2;
    float mapping_regu_weight = 10.0;
    float mapping_prior_weight = 0.0;
    float mapping_convergence_p = 0.9999;
    float mapping_convergence_v = 1e-16;
    float mapping_intrinsic_initial_var = 10.0 * 10.0;
    float mapping_intrinsic_good_var = 0.01 * 0.01;
    float mapping_pose_initial_var = 10.0 * 10.0;
    float mapping_pose_good_var = 10.0 * 10.0;
    float mapping_param_initial_var = 0.5 * 0.5;
    float mapping_param_good_var = 0.26 * 0.26;
}
