#include "vl_mapper/vl_mapper_node.hpp"


namespace vl_mapper
{


VlMapperNode::VlMapperNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("vl_mapper_node", options) 
{
    RCLCPP_INFO(this->get_logger(), "Starting VL Mapper Node.");
    this->declare_and_get_params();

    // Create callback groups for concurrency: realtime (mutually exclusive) and processing (reentrant)
    cbg_realtime_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    cbg_processing_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    this->setup_subscriptions();
    this->setup_publishers();

    // Initialize TF2 buffer and listener for mapping odom -> map
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Create YOLO World service client and attach to processing callback group
    yolo_client_ = this->create_client<DetectAtStamp>(yolo_service_name_, rmw_qos_profile_services_default, cbg_processing_);

    this->setup_kalman_filter();
}


VlMapperNode::~VlMapperNode() {
    worker_shutdown_.store(true);
    queue_cv_.notify_all();

    for (auto &t : worker_threads_) {
        if (t.joinable()) t.join();
    }
}

void VlMapperNode::run() {
    this->setup_workers();
    RCLCPP_INFO(this->get_logger(), "VL Mapper Node is running.");
}


// ###############################################################################
// ----- Setup -----
// ###############################################################################
void VlMapperNode::declare_and_get_params() {
    rgb_topic_ = this->declare_parameter<std::string>("rgb_topic", "/camera/color/image_raw");
    depth_topic_ = this->declare_parameter<std::string>("depth_topic", "/camera/depth/image_rect_raw");
    camera_info_topic_ = this->declare_parameter<std::string>("camera_info_topic", "/camera/color/camera_info");
    odom_topic_ = this->declare_parameter<std::string>("odom_topic", "/odom");
    prompts_topic_ = this->declare_parameter<std::string>("labels_topic", "/vl_mapper/labels");
    pub_debug_topics_ = this->declare_parameter<bool>("publish_debug_topics", true);
    viz_rgb_topic_ = this->declare_parameter<std::string>("viz_rgb_topic", "/vl_mapper/yolo_overlay");
    viz_depth_topic_ = this->declare_parameter<std::string>("viz_depth_topic", "/vl_mapper/yolo_overlay_depth");
    viz_pc_topic_ = this->declare_parameter<std::string>("viz_point_cloud_topic", "/vl_mapper/point_cloud");
    detections_topic_ = this->declare_parameter<std::string>("detections_topic", "/vl_mapper/detections_3d");
    yolo_service_name_ = this->declare_parameter<std::string>("yolo_service", "/yolo_service_server/detect_labels");
    queue_size_ = this->declare_parameter<int>("queue_size", 10);
    unsubscribe_cinfo_after_first_ = this->declare_parameter<bool>("unsubscribe_camera_info_after_first", true);
}


void VlMapperNode::setup_subscriptions() {
    if (rgb_topic_.empty() || depth_topic_.empty()) {
        RCLCPP_FATAL(this->get_logger(), "Required topic names are empty. Shutting down.");
        rclcpp::shutdown();
    }

    rgb_sub_ = std::make_shared<message_filters::Subscriber<Image>>(this, rgb_topic_);
    depth_sub_ = std::make_shared<message_filters::Subscriber<Image>>(this, depth_topic_);
    odom_sub_ = std::make_shared<message_filters::Subscriber<Odometry>>(this, odom_topic_);

    approx_sync_ = std::make_shared<message_filters::Synchronizer<ApproxPolicy>>(
        ApproxPolicy(queue_size_), *rgb_sub_, *depth_sub_, *odom_sub_);
    approx_sync_->registerCallback(std::bind(
        &VlMapperNode::sync_callback, this,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

    rclcpp::SubscriptionOptions so_realtime;
    so_realtime.callback_group = cbg_realtime_;

    prompts_sub_ = this->create_subscription<Labels>(
        prompts_topic_, rclcpp::QoS(10), std::bind(&VlMapperNode::prompts_callback, this, std::placeholders::_1),
        so_realtime);

    cinfo_sub_ = this->create_subscription<CameraInfo>(
        camera_info_topic_, rclcpp::QoS(10), std::bind(&VlMapperNode::cinfo_callback, this, std::placeholders::_1),
        so_realtime);
}


void VlMapperNode::setup_publishers() {
    if (!viz_rgb_topic_.empty() && pub_debug_topics_) {
        viz_rgb_pub_ = this->create_publisher<sensor_msgs::msg::Image>(viz_rgb_topic_, rclcpp::QoS(10));
    }
    if (!viz_depth_topic_.empty() && pub_debug_topics_) {
        viz_depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>(viz_depth_topic_, rclcpp::QoS(10));
    }
    if (!viz_pc_topic_.empty() && pub_debug_topics_) {
        viz_pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(viz_pc_topic_, rclcpp::QoS(10));
    }
    if (!detections_topic_.empty()) {
        detections_pub_ = this->create_publisher<Detection3DArray>(detections_topic_, rclcpp::QoS(10));
    }
}


void VlMapperNode::setup_kalman_filter() {
    F_ = StateMatrix::Identity();
    H_ = StateMatrix::Identity();

    Q_ = StateMatrix::Identity();
    Q_(0,0) = 0.01; Q_(1,1) = 0.01; Q_(2,2) = 0.01; // Position (m^2)
    Q_(3,3) = 0.001; Q_(4,4) = 0.001; Q_(5,5) = 0.001;// Size (m^2)
    Q_(6,6) = 1e-9;                                 // Yaw (rad^2) - Fixed Yaw

    R_ = MeasurementMatrix::Identity();
    R_(0,0) = 0.2; R_(1,1) = 0.2; R_(2,2) = 0.2; // Position uncertainty (m^2)
    R_(3,3) = 0.1; R_(4,4) = 0.1; R_(5,5) = 0.1; // Size uncertainty (m^2)
    R_(6,6) = 1e-9;                               // Yaw uncertainty (rad^2) - Trust Fixed Yaw
}


void VlMapperNode::setup_workers() {
    // determine default worker count if not set
    if (worker_count_ <= 0) {
        unsigned int hw = std::thread::hardware_concurrency();
        worker_count_ = (hw > 1) ? std::max(1u, hw - 1u) : 1;
    }

    for (int i = 0; i < worker_count_; ++i) {
        worker_threads_.emplace_back([this, i]() {
            RCLCPP_INFO(this->get_logger(), "YOLO worker thread %d started (id %zu)", i, std::hash<std::thread::id>{}(std::this_thread::get_id()));
            while (!worker_shutdown_.load()) {
                YoloTask task;
                {
                    std::unique_lock<std::mutex> lk(queue_mutex_);
                    queue_cv_.wait(lk, [this]() { return worker_shutdown_.load() || !task_queue_.empty(); });
                    if (worker_shutdown_.load() && task_queue_.empty()) break;
                    task = std::move(task_queue_.front());
                    task_queue_.pop();
                }

                // No task
                if (!task.rgb || !task.depth || !task.odom || task.prompts.empty()) {
                    request_inflight_.store(false);
                    continue;
                }

                // Wait for YOLO service to be available
                if (!yolo_client_->wait_for_service(std::chrono::seconds(1))) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "YOLO service not available yet.");
                    request_inflight_.store(false);
                    continue;
                }

                // Prepare request
                auto req = std::make_shared<DetectAtStamp::Request>();
                req->stamp = task.rgb->header.stamp;
                req->labels.labels = task.prompts;

                // Prepare promise/future to receive response from executor callback
                auto resp_promise = std::make_shared<std::promise<std::shared_ptr<DetectAtStamp::Response>>>();
                std::future<std::shared_ptr<DetectAtStamp::Response>> resp_future = resp_promise->get_future();

                // send async request with a callback that sets the promise (will run in executor thread)
                auto send_ok = false;
                try {
                    // Note: capture resp_promise by value
                    yolo_client_->async_send_request(req,
                        [this, resp_promise](rclcpp::Client<DetectAtStamp>::SharedFuture response_future) {
                            try {
                                auto res = response_future.get();
                                resp_promise->set_value(res);
                            } catch (const std::exception &e) {
                                // set exception so worker can handle
                                resp_promise->set_exception(std::make_exception_ptr(e));
                            }
                        }
                    );
                    send_ok = true;
                } catch (const std::exception &e) {
                    RCLCPP_ERROR(this->get_logger(), "Failed to async_send_request from worker: %s", e.what());
                }

                if (!send_ok) {
                    request_inflight_.store(false);
                    continue;
                }

                // Wait for service response (blocks this worker thread only)
                std::shared_ptr<DetectAtStamp::Response> res;
                try {
                    // consider a timeout variant if desired using wait_for + future_error
                    res = resp_future.get();
                } catch (const std::exception &e) {
                    RCLCPP_ERROR(this->get_logger(), "YOLO worker: exception waiting for response: %s", e.what());
                    request_inflight_.store(false);
                    continue;
                }

                // Service Error Output
                if (!res->success) {
                    RCLCPP_ERROR(this->get_logger(), "%s", res->message.c_str());
                    request_inflight_.store(false);
                    continue;
                }

                if (!res->message.empty()) {
                    RCLCPP_INFO(this->get_logger(), "Detections: %zu | %s ",
                        res->detections.detections.size(), res->message.c_str());
                }

                // Process Detections
                try {
                    this->process_new_detections(task, res->detections);
                } catch (const std::exception &e) {
                    RCLCPP_ERROR(this->get_logger(), "Exception in process_new_detections: %s", e.what());
                }

                request_inflight_.store(false);
            } // while
            RCLCPP_INFO(this->get_logger(), "YOLO worker thread exiting");
        });
    }
}


// ###############################################################################
// ----- ROS Callbacks -----
// ###############################################################################
void VlMapperNode::sync_callback(
        const Image::ConstSharedPtr & rgb,
        const Image::ConstSharedPtr & depth,
        const Odometry::ConstSharedPtr & odom)
{
    RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Sync callback triggered! Processing frame...");

    // Snapshot prompts first; if none exist yet, do not arm request_inflight_.
    std::vector<std::string> prompts_snapshot;
    {
        std::lock_guard<std::mutex> lk(prompts_mutex_);
        prompts_snapshot = prompts_;
    }
    if (prompts_snapshot.empty()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "No labels received yet; skipping YOLO request.");
        return;
    }

    if (request_inflight_.exchange(true)) {
        RCLCPP_DEBUG(this->get_logger(), "Skipping YOLO request; previous one still in flight.");
        return;
    }

    YoloTask task;
    task.rgb = rgb;
    task.depth = depth;
    task.odom = odom;
    task.prompts = std::move(prompts_snapshot);

    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        task_queue_.push(std::move(task));
    }
    queue_cv_.notify_one();
}


void VlMapperNode::prompts_callback(const Labels::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(prompts_mutex_);
    prompts_.clear();
    prompts_.assign(msg->labels.begin(), msg->labels.end());
    RCLCPP_INFO(this->get_logger(), "Received %zu labels.", prompts_.size());
}


void VlMapperNode::cinfo_callback(const CameraInfo::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(cinfo_mutex_);
    cinfo_ = msg;
    if (unsubscribe_cinfo_after_first_ && cinfo_sub_) {
        RCLCPP_INFO(this->get_logger(), "CameraInfo received once; unsubscribing to save CPU.");
        cinfo_sub_.reset();
    }
}


// ###############################################################################
// ----- Processing -----
// ###############################################################################
void VlMapperNode::process_new_detections(
    const YoloTask & task,
    const Detection2DArray & detections)
{
    std::vector<PointCloudLabelled> labeled_clouds = compute_detection_point_clouds(task.depth, detections);
    if (labeled_clouds.empty()) return;

    const std::string target_frame = "map";
    const std::string camera_frame = task.depth->header.frame_id;
    const rclcpp::Time meas_stamp(task.depth->header.stamp);

    std::optional<Eigen::Isometry3d> T_map_camera_opt = 
        lookup_transform_eigen(tf_buffer_, target_frame, camera_frame, meas_stamp);

    if (!T_map_camera_opt) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "No TF from '%s' to 'map' available; skipping detection processing.", camera_frame.c_str());
        return;
    }
    const Eigen::Isometry3d& T_map_camera = *T_map_camera_opt;

    std::vector<BboxData> new_measurements;
    std::vector<std::string> new_labels;
    std::vector<float> new_scores;

    for (const auto& pc_labelled : labeled_clouds) {
        // BboxData bbox_eigen_cam = cloud_to_bbox(pc_labelled.cloud);
        // BboxData bbox_eigen_map = transform_bbox(T_map_camera, bbox_eigen_cam);
        BboxData bbox_eigen_map = compute_aabb_in_map_frame(pc_labelled.cloud, T_map_camera);

        new_measurements.push_back(bbox_eigen_map);
        new_labels.push_back(pc_labelled.label);
        new_scores.push_back(pc_labelled.score);
    }

    this->kalman_tracking(new_measurements, new_labels, new_scores);

    // Publish results
    this->publish_tracked_objects();
    this->publish_point_clouds(labeled_clouds, task.depth->header.frame_id);
    this->publish_overlay_depth(task.depth, detections);
    this->publish_overlay_rgb(task.rgb, detections);
}


void VlMapperNode::kalman_tracking(
    const std::vector<BboxData>& new_measurements,
    const std::vector<std::string>& new_labels,
    const std::vector<float>& new_scores)
{
    // Lock tracked objects for the full update cycle
    std::lock_guard<std::mutex> lk(tracked_objects_mutex_);

    // Prediction for all existing tracks
    for (auto& track : tracked_objects_) {
        kalman_predict(track, F_, Q_);
    }

    std::vector<bool> matched_measurements(new_measurements.size(), false);
    std::unordered_map<int, bool> matched_tracks_by_id;
    matched_tracks_by_id.reserve(tracked_objects_.size());
    for (const auto & t : tracked_objects_) matched_tracks_by_id[t.id] = false;

    // Data association
    for (size_t i = 0; i < new_measurements.size(); ++i) {
        int best_track_idx = -1;
        double min_dist_sq = std::numeric_limits<double>::max();

        for (size_t j = 0; j < tracked_objects_.size(); ++j) {
            if (new_labels[i] != tracked_objects_[j].label) continue;
            MeasurementMatrix R_dynamic = get_R_dynamic(new_scores[i], R_);

            // Use Mahalanobis gating instead of is_inside
            if (mahalanobis_gating(new_measurements[i], tracked_objects_[j], R_dynamic, GATING_THRESHOLD)) {
                double dist_sq = (new_measurements[i].center - tracked_objects_[j].x.head<3>()).squaredNorm();
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    best_track_idx = j;
                }
            }
        }

        if (best_track_idx != -1) {
            // Match found!
            TrackedObject& track = tracked_objects_[best_track_idx];

            MeasurementVector z;
            z << new_measurements[i].center.x(),
                 new_measurements[i].center.y(),
                 new_measurements[i].center.z(),
                 new_measurements[i].size.x(),
                 new_measurements[i].size.y(),
                 new_measurements[i].size.z(),
                 get_yaw_from_quaternion(new_measurements[i].orientation);

            // Dynamically adjust R based on detection confidence
            MeasurementMatrix R_dynamic = get_R_dynamic(new_scores[i], R_);
            R_dynamic(6,6) = std::max(R_dynamic(6,6), 0.25);
            kalman_update(track, z, R_dynamic, H_);

            track.time_since_last_update = 0;
            track.hits++;
            matched_measurements[i] = true;
            matched_tracks_by_id[track.id] = true;
        }
    }

    // Lifecycle-Management
    for (auto & track : tracked_objects_) {
        auto it = matched_tracks_by_id.find(track.id);
        if (it == matched_tracks_by_id.end() || !it->second) {
            track.time_since_last_update++;
        }
    }
    // Remove only those tracks that actually exceeded the timeout
    tracked_objects_.erase(
        std::remove_if(tracked_objects_.begin(), tracked_objects_.end(), 
            [](const TrackedObject& track) { return track.time_since_last_update > 30; }),
        tracked_objects_.end()
    );

    for (size_t i = 0; i < new_measurements.size(); ++i) {
        if (!matched_measurements[i]) {
            TrackedObject new_track;
            new_track.id = next_object_id_++;
            new_track.label = new_labels[i];
            
            // Initialize state vector
            new_track.x(0) = new_measurements[i].center.x();
            new_track.x(1) = new_measurements[i].center.y();
            new_track.x(2) = new_measurements[i].center.z();
            new_track.x(3) = new_measurements[i].size.x();
            new_track.x(4) = new_measurements[i].size.y();
            new_track.x(5) = new_measurements[i].size.z();
            new_track.x(6) = get_yaw_from_quaternion(new_measurements[i].orientation);

            // High initial uncertainty
            new_track.P = StateMatrix::Identity() * 10.0;
            tracked_objects_.push_back(new_track);
        }
    }
}


std::vector<PointCloudLabelled> VlMapperNode::compute_detection_point_clouds(
        const Image::ConstSharedPtr & depth,
        const Detection2DArray & detections)
{
    // Early validation
    if (!depth) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
            "No depth image received; cannot compute 3D BBoxes.");
        return {};
    }

    CameraInfo::ConstSharedPtr local_cinfo;
    {
        std::lock_guard<std::mutex> lk(cinfo_mutex_);
        if (!cinfo_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "No CameraInfo received yet; cannot compute 3D BBoxes.");
            return {};
        }
        local_cinfo = cinfo_;
    }

    cv_bridge::CvImageConstPtr depth_ptr;
    try {
        depth_ptr = cv_bridge::toCvShare(depth, sensor_msgs::image_encodings::TYPE_16UC1);
    } catch(const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error converting depth image: %s", e.what());
        return {};
    }
    const cv::Mat depth_mat = depth_ptr->image.clone();

    // Validate and extract camera intrinsics
    const float fx = static_cast<float>(local_cinfo->k[0]);
    const float fy = static_cast<float>(local_cinfo->k[4]);
    const float cx = static_cast<float>(local_cinfo->k[2]);
    const float cy = static_cast<float>(local_cinfo->k[5]);

    if (fx <= 0.0f || fy <= 0.0f || local_cinfo->width == 0 || local_cinfo->height == 0) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
            "Invalid camera intrinsics (fx=%.2f, fy=%.2f, w=%d, h=%d)", 
            fx, fy, local_cinfo->width, local_cinfo->height);
        return {};
    }

    const float scale_x = static_cast<float>(depth_mat.cols) / static_cast<float>(local_cinfo->width);
    const float scale_y = static_cast<float>(depth_mat.rows) / static_cast<float>(local_cinfo->height);

    // Configuration constants
    constexpr uint16_t MIN_DEPTH_MM = 100;      // 10cm minimum
    constexpr uint16_t MAX_DEPTH_MM = 10000;    // 10m maximum
    constexpr float MAX_OBJECT_DISTANCE = 8.0f; // 8m max tracking distance
    constexpr size_t MIN_POINTS_THRESHOLD = 50; // Minimum points for valid cloud

    // Back-project 2D pixel (u, v) with depth to 3D point in camera frame
    auto backproject_pixel = [&](int u, int v, float depth_m) -> pcl::PointXYZ {
        const float logical_x = static_cast<float>(u) / scale_x;
        const float logical_y = static_cast<float>(v) / scale_y;
        const float X = (logical_x - cx) * depth_m / fx;
        const float Y = (logical_y - cy) * depth_m / fy;
        return pcl::PointXYZ(X, Y, depth_m);
    };

    std::vector<PointCloudLabelled> results_3D;
    results_3D.reserve(detections.detections.size());

    // Process each detection
    for (const auto& det : detections.detections) {
        // Skip detections without results
        if (det.results.empty()) {
            continue;
        }

        // Get 2D bounding box in image coordinates
        auto [x_min_raw, y_min_raw, x_max_raw, y_max_raw] = get_bbox_in_img(det.bbox, depth_mat);
        
        // Clamp to image bounds
        const int x_min = std::max(0, x_min_raw);
        const int y_min = std::max(0, y_min_raw);
        const int x_max = std::min(depth_mat.cols, x_max_raw);
        const int y_max = std::min(depth_mat.rows, y_max_raw);
        
        // Skip invalid bounding boxes
        if (x_min >= x_max || y_min >= y_max) {
            continue;
        }

        // Extract all valid depth points within the bounding box
        pcl::PointCloud<pcl::PointXYZ>::Ptr initial_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        initial_cloud->reserve(static_cast<size_t>((x_max - x_min) * (y_max - y_min)));

        for (int v = y_min; v < y_max; ++v) {
            const uint16_t* row_ptr = depth_mat.ptr<uint16_t>(v);
            for (int u = x_min; u < x_max; ++u) {
                const uint16_t depth_val_mm = row_ptr[u];
                
                if (depth_val_mm < MIN_DEPTH_MM || depth_val_mm > MAX_DEPTH_MM) {
                    continue;
                }

                const float depth_m = static_cast<float>(depth_val_mm) / 1000.0f;
                initial_cloud->points.push_back(backproject_pixel(u, v, depth_m));
            }
        }

        if (initial_cloud->points.size() < MIN_POINTS_THRESHOLD) {
            continue;
        }

        // Find median depth
        std::vector<float> depth_values;
        depth_values.reserve(initial_cloud->points.size());
        for (const auto& pt : initial_cloud->points) {
            depth_values.push_back(pt.z);
        }

        const size_t mid_idx = depth_values.size() / 2;
        std::nth_element(depth_values.begin(), depth_values.begin() + mid_idx, depth_values.end());
        const float median_depth = depth_values[mid_idx];

        if (median_depth > MAX_OBJECT_DISTANCE) {
            continue;
        }

        const float adaptive_tolerance = std::clamp(0.15f * median_depth, 0.1f, 0.5f);

        // Filter foreground points
        pcl::PointCloud<pcl::PointXYZ>::Ptr foreground_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        foreground_cloud->reserve(initial_cloud->points.size());

        for (const auto& pt : initial_cloud->points) {
            if (std::abs(pt.z - median_depth) <= adaptive_tolerance) {
                foreground_cloud->points.push_back(pt);
            }
        }

        if (foreground_cloud->points.size() < MIN_POINTS_THRESHOLD) {
            continue;
        }

        // Create labeled point cloud result
        PointCloudLabelled pc_labelled;
        pc_labelled.cloud = foreground_cloud;
        pc_labelled.label = det.results.front().hypothesis.class_id;
        pc_labelled.score = det.results.front().hypothesis.score;
        
        results_3D.push_back(std::move(pc_labelled));
    }

    return results_3D;
}


// ###############################################################################
// ----- Publishing -----
// ###############################################################################
void VlMapperNode::publish_tracked_objects() {
    if (detections_topic_.empty() || !detections_pub_) return;
    
    auto det_array_msg = std::make_unique<Detection3DArray>();
    det_array_msg->header.stamp = this->get_clock()->now();
    det_array_msg->header.frame_id = "map";

    // Copy tracked objects under lock
    std::vector<TrackedObject> tracks_copy;
    {
        std::lock_guard<std::mutex> lk(tracked_objects_mutex_);
        tracks_copy = tracked_objects_;
    }

    for (const auto& track : tracks_copy) {
        vision_msgs::msg::Detection3D det_msg;
        det_msg.header = det_array_msg->header;
        
        // Fill the message
        det_msg.bbox.center.position.x = track.x(0);
        det_msg.bbox.center.position.y = track.x(1);
        det_msg.bbox.center.position.z = track.x(2);
        det_msg.bbox.size.x = track.x(3);
        det_msg.bbox.size.y = track.x(4);
        det_msg.bbox.size.z = track.x(5);
        
        Eigen::Quaterniond orientation(Eigen::AngleAxisd(track.x(6), Eigen::Vector3d::UnitZ()));
        det_msg.bbox.center.orientation = tf2::toMsg(orientation);

        det_msg.results.resize(1);
        det_msg.results[0].hypothesis.class_id = track.label;
        det_msg.results[0].hypothesis.score = 1.0; 

        det_array_msg->detections.push_back(det_msg);
    }

    detections_pub_->publish(std::move(det_array_msg));
}


void VlMapperNode::publish_point_clouds(
        const std::vector<PointCloudLabelled> & clouds, 
        const std::string & frame_id) 
{
    if (viz_pc_topic_.empty() || !viz_pc_pub_ || !pub_debug_topics_) return;
    if (clouds.empty()) return;

    pcl::PointCloud<pcl::PointXYZ>::Ptr combined_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& cloud : clouds) {
        *combined_cloud += *cloud.cloud;
    }

    sensor_msgs::msg::PointCloud2 pc2_msg;
    pcl::toROSMsg(*combined_cloud, pc2_msg);
    pc2_msg.header.stamp = this->get_clock()->now();
    pc2_msg.header.frame_id = frame_id;

    viz_pc_pub_->publish(pc2_msg);
}


void VlMapperNode::publish_overlay_depth(
        const Image::ConstSharedPtr & depth,
        const Detection2DArray & detections)
{
    if (viz_depth_topic_.empty() || !viz_depth_pub_ || !pub_debug_topics_) return;
    if (!depth || depth->height == 0 || depth->width == 0) return;

    try {
        cv_bridge::CvImageConstPtr cv_d = cv_bridge::toCvShare(depth, sensor_msgs::image_encodings::TYPE_16UC1);
        cv::Mat depth_f32;
        cv_d->image.convertTo(depth_f32, CV_32F, 1.0/1000.0);

        double minv = 0.0, maxv = 0.0;
        cv::minMaxLoc(depth_f32, &minv, &maxv, nullptr, nullptr, depth_f32 > 0);
        if (!(maxv > minv)) maxv = minv + 1.0;

        cv::Mat norm01 = (depth_f32 - minv) / (maxv - minv);
        cv::Mat depth_u8;
        norm01.convertTo(depth_u8, CV_8U, 255.0);
        cv::Mat depth_bgr;
        cv::applyColorMap(depth_u8, depth_bgr, cv::COLORMAP_JET);

        // Draw boxes and labels for visualization only
        for (const auto & det : detections.detections) {
            if (det.bbox.size_x <= 0.0f || det.bbox.size_y <= 0.0f) continue;
            auto [x, y, x2, y2] = get_bbox_in_img(det.bbox, depth_bgr);
            cv::rectangle(depth_bgr, cv::Point(x, y), cv::Point(x2, y2), cv::Scalar(0, 255, 255), 2);

            // Add Label
            if (!det.results.empty()) {
                const auto & hyp = det.results.front().hypothesis;
                std::string label = hyp.class_id + " " + cv::format("%.2f", hyp.score);
                cv::putText(depth_bgr, label, cv::Point(x + 5, y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }
        }

        cv_bridge::CvImage out_msg;
        out_msg.header = depth->header;
        out_msg.encoding = "bgr8";
        out_msg.image = depth_bgr;
        viz_depth_pub_->publish(*out_msg.toImageMsg());
    } catch (const std::exception & e) {
        RCLCPP_WARN(this->get_logger(), "Failed to draw/publish Depth overlay: %s", e.what());
    }
}


void VlMapperNode::publish_overlay_rgb(
        const Image::ConstSharedPtr & rgb,
        const Detection2DArray & detections) 
{
    if (viz_rgb_topic_.empty() || !viz_rgb_pub_ || !pub_debug_topics_) return;
    if (!rgb || rgb->height == 0 || rgb->width == 0) return;

    try {
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(rgb, rgb->encoding);
        cv::Mat bgr;
        if (rgb->encoding == "bgr8") {
            bgr = cv_ptr->image.clone();
        } else if (rgb->encoding == "rgb8") {
            cv::cvtColor(cv_ptr->image, bgr, cv::COLOR_RGB2BGR);
        } else {
            cv_bridge::CvImageConstPtr cv_bgr = cv_bridge::toCvShare(rgb, "bgr8");
            bgr = cv_bgr->image.clone();
        }

        for (const auto & det : detections.detections) {
            if (det.bbox.size_x <= 0.0f || det.bbox.size_y <= 0.0f) continue;
            auto [x, y, x2, y2] = get_bbox_in_img(det.bbox, bgr);
            cv::rectangle(bgr, cv::Rect(cv::Point(x, y), cv::Point(x2, y2)), cv::Scalar(0, 255, 0), 2);
            
            // Add label
            if (!det.results.empty()) {
                const auto & hyp = det.results.front().hypothesis;
                std::string label = hyp.class_id + " " + cv::format("%.2f", hyp.score);
                cv::putText(bgr, label, cv::Point(x + 5, y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }
        }

        cv_bridge::CvImage out_msg;
        out_msg.header = rgb->header;
        out_msg.encoding = "bgr8";
        out_msg.image = bgr;
        viz_rgb_pub_->publish(*out_msg.toImageMsg());
    } catch (const std::exception & e) {
        RCLCPP_WARN(this->get_logger(), "Failed to draw/publish RGB overlay: %s", e.what());
    }
}



} // namespace vl_mapper


// ###############################################################################
// ----- Main -----
// ###############################################################################
int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<vl_mapper::VlMapperNode>();
    node->run();

    // Use a MultiThreadedExecutor so callback groups can execute in parallel
    unsigned int threads = std::max<unsigned int>(1u, std::thread::hardware_concurrency());
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), threads);
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}

