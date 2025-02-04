#include "params.h"
#include "math.h"

namespace mesh_vo
{
    float last_min_angle = M_PI / 128.0;
    float key_max_angle = M_PI / 10.0;
    float min_view_perc = 0.7;
    float min_lambda = 0.00001f;
    float huber_thresh_pix = 3.0;
    float regu_weight = 10.0;
    float prior_weight = 1.0;
    // for a max depth of 10, using log(depth) as param
    // #define MAX_PARAM fromDepthToParam(1.0 + 0.5)
    // #define MIN_PARAM fromDepthToParam(1.0 - 0.5)
    float tracking_pose_var = 10000.0 * 10000.0;
    float mapping_pose_var = 0.01 * 0.01;
    // #define INITIAL_PARAM_STD fromDepthToParam(0.01)
    // for a param = log(depth) and a max depth of 10, the max uncertanty should be ln(3) = 1.098
    float initial_param_var = 0.26 * 0.26;
    // lets suppouse that by the optimization we reduce the uncertainty by 10x, that means a depth uncertainty of exp(0.1*1.098) = 1.116, or for 2x reduction, that means 1.731 uncertanty
    float good_param_var = initial_param_var * 0.5 * 0.5;
}
