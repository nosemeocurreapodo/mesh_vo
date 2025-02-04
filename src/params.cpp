#include "params.h"
#include "math.h"

namespace mesh_vo
{
    float last_min_angle = M_PI / 128.0;
    float key_max_angle = M_PI / 10.0;
    float min_view_perc = 0.7;
    float min_lambda = 0.00001f;
    float huber_thresh_pix = 3.0;

    float tracking_prior_weight = 0.0;
    float tracking_convergence_p = 0.999;
    float tracking_convergence_v = 1e-16;
    float tracking_pose_initial_var = 10000.0 * 10000.0;

    float mapping_regu_weight = 10.0;
    float mapping_prior_weight = 0.0;
    float mapping_convergence_p = 0.9999;
    float mapping_convergence_v = 1e-16;
    float mapping_pose_initial_var = 100000.0 * 100000.0;
    float mapping_pose_good_var = 0.01 * 0.01;
    float mapping_param_initial_var = 10000.0 * 10000.0; //0.26 * 0.26;
    float mapping_param_good_var = 0.5 * 0.5;
}
