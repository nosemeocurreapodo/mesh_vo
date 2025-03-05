#include "params.h"
#include "math.h"

namespace mesh_vo
{
    float last_min_angle = 0.0;
    float key_max_angle = M_PI / 20.0;
    float min_view_perc = 0.8;
    float min_lambda = 0.00001f;
    float huber_thresh_pix = 9.0; //9.0 taken from dso

    int tracking_ini_lvl = 3;
    int tracking_fin_lvl = 1;
    float tracking_prior_weight = 0.0;
    float tracking_convergence_p = 0.999;
    float tracking_convergence_v = 1e-8;
    float tracking_pose_initial_var = 100.0 * 100.0;

    float mapping_mean_depth = 1.0;
    int mapping_ini_lvl = 1;
    int mapping_fin_lvl = 1;
    float mapping_regu_weight = 10.0;
    float mapping_prior_weight = 0.0;
    float mapping_convergence_p = 0.999;
    float mapping_convergence_p_v = 0.0;
    float mapping_convergence_m_v = 1e-16;
    float mapping_intrinsic_initial_var = 1.0 * 1.0;
    float mapping_intrinsic_good_var = 0.1 * 0.1;
    float mapping_pose_initial_var = 10.0 * 10.0;
    float mapping_pose_good_var = 10.0 * 10.0;
    float mapping_param_initial_var = 0.5 * 0.5;
    float mapping_param_good_var = 0.26 * 0.26;
}
