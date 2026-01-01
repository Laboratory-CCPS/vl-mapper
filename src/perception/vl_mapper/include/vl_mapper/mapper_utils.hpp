#ifndef VL_MAPPER_UTILS_HPP
#define VL_MAPPER_UTILS_HPP

#include <vector>
#include <array>
#include <cmath>
#include <optional>
#include <algorithm>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <opencv2/core.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/common/common.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <vision_msgs/msg/bounding_box2_d.hpp>


#define STATE_DIM 7
#define MEAS_DIM 7
#define EPSILON 1e-6
#define GATING_THRESHOLD 7.815 // Chi-squared value for 95% confidence in position (3 DOF)


namespace vl_mapper
{


// ====== Usings ======
using StateVector = Eigen::Matrix<double, STATE_DIM, 1>;
using StateMatrix = Eigen::Matrix<double, STATE_DIM, STATE_DIM>;
using MeasurementVector = Eigen::Matrix<double, MEAS_DIM, 1>;
using MeasurementMatrix = Eigen::Matrix<double, MEAS_DIM, MEAS_DIM>;


// ====== Structs ======
struct TrackedObject {
    int id;
    std::string label;

    StateVector x;      // Statesvektor
    StateMatrix P;      // Covarianzmatrix

    // Metadata
    int time_since_last_update = 0;
    int hits = 1;
};


struct BboxData {
    Eigen::Vector3d center;
    Eigen::Quaterniond orientation;
    Eigen::Vector3d size;
};


// ====== Kalman Filter ======
/**
 * @brief Performs the prediction step of the Kalman filter.
 * @param track The object whose state should be predicted.
 * @param F The state transition matrix.
 * @param Q The process noise covariance.
 */
inline void kalman_predict(TrackedObject& track, const StateMatrix& F, const StateMatrix& Q) {
    track.x = F * track.x;
    track.P = F * track.P * F.transpose() + Q;
}


/**
 * @brief Updates the Kalman filter state with a new measurement based on the Joseph form.
 * @param track The object whose state should be updated.
 * @param z The new measurement.
 * @param R The measurement noise covariance.
 * @param H The measurement matrix.
 */
inline void kalman_update(TrackedObject& track, const MeasurementVector& z, const MeasurementMatrix& R, const StateMatrix& H) {
    // Innovation (wrap yaw)
    MeasurementVector innovation = z - H * track.x;
    innovation(6) = std::atan2(std::sin(innovation(6)), std::cos(innovation(6)));

    // S = HPH' + R
    const MeasurementMatrix S = H * track.P * H.transpose() + R;

    // K = P H' S^-1 (solve, don't invert)
    const StateMatrix PHt = track.P * H.transpose();
    const StateMatrix K = PHt * S.ldlt().solve(MeasurementMatrix::Identity());

    // x = x + K * innovation
    track.x = track.x + K * innovation;

    // Joseph form:
    // P = (I - K H) P (I - K H)' + K R K'
    const StateMatrix I = StateMatrix::Identity();
    const StateMatrix I_KH = I - K * H;
    track.P = I_KH * track.P * I_KH.transpose() + K * R * K.transpose();

    // Optional: enforce symmetry (guards numerical drift)
    track.P = 0.5 * (track.P + track.P.transpose());

    // Keep yaw bounded
    track.x(6) = std::atan2(std::sin(track.x(6)), std::cos(track.x(6)));
}


/**
 * @brief Checks if a point is inside the 3D bounding box of a tracked object.
 * @param point The 3D point to check.
 * @param track The tracked object.
 * @return True if the point is inside the bounding box, false otherwise.
 */
inline bool is_inside(const Eigen::Vector3d& point, const TrackedObject& track) {
    const Eigen::Vector3d track_center = track.x.head<3>();
    const Eigen::Vector3d track_size = track.x.segment<3>(3);
    const Eigen::Quaterniond track_orientation(Eigen::AngleAxisd(track.x(6), Eigen::Vector3d::UnitZ()));

    // Transform point to the local coordinate system of the track
    Eigen::Vector3d point_relative = track_orientation.inverse() * (point - track_center);

    // Check if the relative point is inside the axis-aligned bounding box
    return (std::abs(point_relative.x()) <= track_size.x() / 2.0) &&
           (std::abs(point_relative.y()) <= track_size.y() / 2.0) &&
           (std::abs(point_relative.z()) <= track_size.z() / 2.0);
}


/**
 * @brief Performs Mahalanobis gating to check if a measurement is a likely match for a track position.
 * @param measurement The new measurement data.
 * @param track The track to compare against.
 * @param R The measurement noise covariance matrix for this specific measurement.
 * @param gating_threshold The chi-squared gating threshold.
 * @return True if the measurement is within the gate, false otherwise.
 */
inline bool mahalanobis_gating(
        const BboxData& measurement,
        const TrackedObject& track,
        const MeasurementMatrix& R,
        const double gating_threshold)
{
    const Eigen::Vector3d z_pos = measurement.center;
    const Eigen::Vector3d x_pos_predicted = track.x.head<3>();
    const Eigen::Vector3d innovation_pos = z_pos - x_pos_predicted;

    const Eigen::Matrix3d P_pos = track.P.block<3,3>(0, 0);
    const Eigen::Matrix3d R_pos = R.block<3,3>(0, 0);
    const Eigen::Matrix3d S_pos = P_pos + R_pos;

    // dist^2 = innov' S^-1 innov  (solve, don't invert)
    const Eigen::Vector3d y = S_pos.ldlt().solve(innovation_pos);
    const double dist_sq = innovation_pos.dot(y);

    return dist_sq < gating_threshold;
}


inline double get_yaw_from_quaternion(const Eigen::Quaterniond& q) {
    // Yaw (z-axis rotation)
    double siny_cosp = 2 * (q.w() * q.z() + q.x() * q.y());
    double cosy_cosp = 1 - 2 * (q.y() * q.y() + q.z() * q.z());
    return std::atan2(siny_cosp, cosy_cosp);
}


inline MeasurementMatrix get_R_dynamic(const float new_score, const MeasurementMatrix& R_) {
    const double s = std::max<double>(0.0, new_score);
    const double scaling = 1.0 / (s * s + EPSILON);
    const double scaling_clamped = std::clamp(scaling, 0.5, 100.0);
    return R_ * scaling_clamped;
}



// ====== Bounding Box Helpers ======
inline std::array<int, 4> get_bbox_in_img(
        const vision_msgs::msg::BoundingBox2D& bbox, const cv::Mat& img)
{
    double cx_n = bbox.center.position.x; // normalized [0,1]
    double cy_n = bbox.center.position.y;
    double w_n = bbox.size_x;
    double h_n = bbox.size_y;
    int img_w = static_cast<int>(img.cols);
    int img_h = static_cast<int>(img.rows);
    int w = std::max(1, static_cast<int>(w_n * img_w));
    int h = std::max(1, static_cast<int>(h_n * img_h));
    int cx = static_cast<int>(cx_n * img_w);
    int cy = static_cast<int>(cy_n * img_h);
    int x_min = std::clamp(cx - w / 2, 0, img_w - 1);
    int y_min = std::clamp(cy - h / 2, 0, img_h - 1);
    int x_max = std::clamp(cx + w / 2, 0, img_w - 1);
    int y_max = std::clamp(cy + h / 2, 0, img_h - 1);
    return {x_min, y_min, x_max, y_max};
}


inline BboxData cloud_to_bbox(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
    BboxData result;

    if (!cloud || cloud->points.empty()) {
        return result;
    }

    pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
    feature_extractor.setInputCloud(cloud);
    feature_extractor.compute();

    pcl::PointXYZ position;
    Eigen::Matrix3f rotational_matrix;
    pcl::PointXYZ min_point_OBB, max_point_OBB;

    feature_extractor.getOBB(min_point_OBB, max_point_OBB, position, rotational_matrix);

    result.center = Eigen::Vector3d(position.x, position.y, position.z);
    Eigen::Quaternionf quat_f(rotational_matrix);
    result.orientation = quat_f.cast<double>();
    
    result.size = Eigen::Vector3d(
        max_point_OBB.x - min_point_OBB.x,
        max_point_OBB.y - min_point_OBB.y,
        max_point_OBB.z - min_point_OBB.z
    );

    return result;
}


/**
 * @brief Computes IQR-based bounds for robust outlier rejection.
 * @param sorted_values Sorted vector of values.
 * @param iqr_multiplier Multiplier for IQR (typically 1.5 for outliers, 1.0 for tighter bounds).
 * @return Pair of (lower_bound, upper_bound) indices.
 */
inline std::pair<size_t, size_t> compute_iqr_bounds(
        const std::vector<float>& sorted_values,
        float iqr_multiplier = 1.0f)
{
    const size_t n = sorted_values.size();
    if (n < 4) return {0, n > 0 ? n - 1 : 0};

    const size_t q1_idx = n / 4;
    const size_t q3_idx = (3 * n) / 4;
    const float q1 = sorted_values[q1_idx];
    const float q3 = sorted_values[q3_idx];
    const float iqr = q3 - q1;

    const float lower_fence = q1 - iqr_multiplier * iqr;
    const float upper_fence = q3 + iqr_multiplier * iqr;

    // Find indices within fences
    size_t lo = 0;
    while (lo < n && sorted_values[lo] < lower_fence) ++lo;
    
    size_t hi = n - 1;
    while (hi > lo && sorted_values[hi] > upper_fence) --hi;

    return {lo, hi};
}


inline BboxData compute_aabb_in_map_frame(
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud,
        const Eigen::Isometry3d& T_map_camera)
{
    BboxData result;

    if (!cloud || cloud->points.empty()) {
        return result;
    }

    if (T_map_camera.matrix().hasNaN()) {
        return result;
    }

    // Transform all points and collect coordinates
    std::vector<float> xs, ys, zs;
    xs.reserve(cloud->points.size());
    ys.reserve(cloud->points.size());
    zs.reserve(cloud->points.size());

    for (const auto& pt : cloud->points) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
            continue;
        }
        
        // Transform point manually
        Eigen::Vector3d pt_cam(pt.x, pt.y, pt.z);
        Eigen::Vector3d pt_map = T_map_camera * pt_cam;

        xs.push_back(static_cast<float>(pt_map.x()));
        ys.push_back(static_cast<float>(pt_map.y()));
        zs.push_back(static_cast<float>(pt_map.z()));
    }

    if (xs.size() < 10) {
        return result;
    }

    // Sort for percentile/IQR calculations
    std::sort(xs.begin(), xs.end());
    std::sort(ys.begin(), ys.end());
    std::sort(zs.begin(), zs.end());

    // Use IQR-based outlier rejection for bounds
    auto [x_lo, x_hi] = compute_iqr_bounds(xs, 1.0f);
    auto [y_lo, y_hi] = compute_iqr_bounds(ys, 1.0f);
    auto [z_lo, z_hi] = compute_iqr_bounds(zs, 1.0f);

    const float min_x = xs[x_lo], max_x = xs[x_hi];
    const float min_y = ys[y_lo], max_y = ys[y_hi];
    const float min_z = zs[z_lo], max_z = zs[z_hi];

    // Use MEDIAN for center (more robust than midpoint)
    const size_t n = xs.size();
    const float median_x = xs[n / 2];
    const float median_y = ys[n / 2];
    const float median_z = zs[n / 2];

    // Blend median with geometric center (70% median, 30% midpoint)
    // This provides stability while still respecting actual geometry
    constexpr float median_weight = 0.7f;
    constexpr float midpoint_weight = 1.0f - median_weight;

    result.center = Eigen::Vector3d(
        median_weight * median_x + midpoint_weight * (min_x + max_x) / 2.0f,
        median_weight * median_y + midpoint_weight * (min_y + max_y) / 2.0f,
        median_weight * median_z + midpoint_weight * (min_z + max_z) / 2.0f
    );
    
    result.size = Eigen::Vector3d(
        static_cast<double>(max_x - min_x),
        static_cast<double>(max_y - min_y),
        static_cast<double>(max_z - min_z)
    );

    result.orientation = Eigen::Quaterniond::Identity();

    return result;
}



// ====== TF Helpers ======
inline std::optional<geometry_msgs::msg::TransformStamped> lookup_transform_with_fallback(
        const std::shared_ptr<tf2_ros::Buffer> & tf_buffer,
        const std::string & target_frame,
        const std::string & source_frame,
        const rclcpp::Time & stamp)
{
    if (!tf_buffer) return std::nullopt;

    try {
        auto current_tf = tf_buffer->lookupTransform(target_frame, source_frame, stamp, rclcpp::Duration::from_nanoseconds(1000));
        return current_tf;
    } catch (const tf2::TransformException & e) {
        // fallback to latest available
        try {
            auto current_tf = tf_buffer->lookupTransform(target_frame, source_frame, tf2::TimePointZero);
            return current_tf;
        } catch (const tf2::TransformException & e2) {
            return std::nullopt;
        }
    }
}


inline std::optional<Eigen::Isometry3d> lookup_transform_eigen(
        const std::shared_ptr<tf2_ros::Buffer>& tf_buffer,
        const std::string& target_frame,
        const std::string& source_frame,
        const rclcpp::Time& stamp)
{
    std::optional<geometry_msgs::msg::TransformStamped> tf_msg_opt = 
        lookup_transform_with_fallback(tf_buffer, target_frame, source_frame, stamp);

    if (!tf_msg_opt) {
        return std::nullopt;
    }
    return tf2::transformToEigen(*tf_msg_opt);
}


inline BboxData transform_bbox(
    const Eigen::Isometry3d& transform, 
    const BboxData& bbox_in)
{
    BboxData bbox_out;

    bbox_out.center = transform * bbox_in.center;
    bbox_out.orientation = transform.rotation() * bbox_in.orientation;
    
    bbox_out.size = bbox_in.size;

    return bbox_out;
}


} // namespace vl_mapper

#endif // VL_MAPPER_UTILS_HPP