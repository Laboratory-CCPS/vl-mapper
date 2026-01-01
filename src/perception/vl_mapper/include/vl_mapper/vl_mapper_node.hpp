#ifndef VL_MAPPER_NODE_HPP
#define VL_MAPPER_NODE_HPP

#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <thread>
#include <queue>
#include <condition_variable>
#include <future>
#include <cmath>
#include <omp.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>

// ROS2
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>

// Msgs
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vision_msgs/msg/bounding_box2_d_array.hpp>
#include <vision_msgs/msg/bounding_box3_d_array.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection3_d.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <vl_mapper_interface/msg/labels.hpp>
#include <vl_mapper_interface/srv/detect_at_stamp.hpp>

// Internal
#include "vl_mapper/mapper_utils.hpp"


namespace vl_mapper
{

// --- Usings ---
using Image = sensor_msgs::msg::Image;
using CameraInfo = sensor_msgs::msg::CameraInfo;
using Odometry = nav_msgs::msg::Odometry;
using TransformStamped = geometry_msgs::msg::TransformStamped;
using Labels = vl_mapper_interface::msg::Labels;
using ApproxPolicy = message_filters::sync_policies::ApproximateTime<Image, Image, Odometry>;
using DetectAtStamp = vl_mapper_interface::srv::DetectAtStamp;
using Detection2DArray = vision_msgs::msg::Detection2DArray;
using Detection3DArray = vision_msgs::msg::Detection3DArray;

// --- Structs ---
struct PointCloudLabelled {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    std::string label = "unknown";
    double score = 0.0;
};

struct YoloTask {
    Image::ConstSharedPtr rgb;
    Image::ConstSharedPtr depth;
    Odometry::ConstSharedPtr odom;
    std::vector<std::string> prompts;
};


// --- Mapper class ---
class VlMapperNode : public rclcpp::Node 
{
private:
    // --- Parameters --- 
    std::string rgb_topic_;
    std::string depth_topic_;
    std::string camera_info_topic_;
    std::string odom_topic_;
    std::string prompts_topic_;
    bool pub_debug_topics_ {false};
    std::string viz_rgb_topic_;
    std::string viz_depth_topic_;
    std::string viz_pc_topic_;
    std::string detections_topic_;
    std::string yolo_service_name_;
    int queue_size_ {5};
    int worker_count_ {2};
    std::atomic<bool> worker_shutdown_{false};
    bool unsubscribe_cinfo_after_first_ {true};

    // --- Subscriber ---
    std::shared_ptr<message_filters::Subscriber<Image>> rgb_sub_;
    std::shared_ptr<message_filters::Subscriber<Image>> depth_sub_;
    std::shared_ptr<message_filters::Subscriber<Odometry>> odom_sub_;
    std::shared_ptr<message_filters::Synchronizer<ApproxPolicy>> approx_sync_;

    rclcpp::Subscription<Labels>::SharedPtr prompts_sub_;
    rclcpp::Subscription<CameraInfo>::SharedPtr cinfo_sub_;

    // --- Publisher ---
    rclcpp::Publisher<Image>::SharedPtr viz_rgb_pub_;
    rclcpp::Publisher<Image>::SharedPtr viz_depth_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr viz_pc_pub_;
    rclcpp::Publisher<Detection3DArray>::SharedPtr detections_pub_;

    // --- Service ---
    rclcpp::Client<DetectAtStamp>::SharedPtr yolo_client_;

    // --- TF ---
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // --- Callback groups (for multithreaded executor) ---
    rclcpp::CallbackGroup::SharedPtr cbg_realtime_;
    rclcpp::CallbackGroup::SharedPtr cbg_processing_;

    // --- Mutex ---
    std::mutex prompts_mutex_;
    std::mutex tracked_objects_mutex_;
    std::mutex cinfo_mutex_;
    std::mutex queue_mutex_;
    
    // --- Cached Data ---
    std::atomic<bool> request_inflight_ {false};
    std::vector<std::string> prompts_;
    CameraInfo::ConstSharedPtr cinfo_;
    std::vector<TrackedObject> tracked_objects_;
    int next_object_id_ {0};
    
    // --- Task Queue ---
    std::queue<YoloTask> task_queue_;
    std::condition_variable queue_cv_;
    std::vector<std::thread> worker_threads_;

    // --- Kalman Filter ---
    StateMatrix F_; // State Transition
    StateMatrix Q_; // Process Noise
    StateMatrix H_; // Measurement Matrix
    MeasurementMatrix R_; // Measurement Noise

public:
    explicit VlMapperNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~VlMapperNode();
    void run();

private:
    // --- Setup ---
    void declare_and_get_params();
    void setup_subscriptions();
    void setup_publishers();
    void setup_kalman_filter();
    void setup_workers();

    // --- ROS Callbacks ---
    void sync_callback(
        const Image::ConstSharedPtr & rgb, 
        const Image::ConstSharedPtr & depth,
        const Odometry::ConstSharedPtr & odom);
    void prompts_callback(const Labels::SharedPtr msg);
    void cinfo_callback(const CameraInfo::SharedPtr msg);

    // --- Processing ---
    void process_new_detections(
        const YoloTask & task,
        const Detection2DArray & detections);

    // --- Kalman Filter ---
    void kalman_tracking(
        const std::vector<BboxData>& new_measurements,
        const std::vector<std::string>& new_labels,
        const std::vector<float>& new_scores);

    // --- Helpers ---
    std::vector<PointCloudLabelled> compute_detection_point_clouds(
        const Image::ConstSharedPtr & depth,
        const Detection2DArray & detections);

    // --- Publishing ---
    void publish_tracked_objects();
    void publish_point_clouds(
        const std::vector<PointCloudLabelled> & clouds,
        const std::string & frame_id);
    void publish_overlay_rgb(
        const Image::ConstSharedPtr & rgb,
        const Detection2DArray & detections);
    void publish_overlay_depth(
        const Image::ConstSharedPtr & depth,
        const Detection2DArray & detections);
};

} // namespace vl_mapper

#endif // VL_MAPPER_NODE_HPP