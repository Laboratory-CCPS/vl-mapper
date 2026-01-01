#pragma once

#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <limits>
#include <array>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <future>
#include <cstdlib>
#include <numeric>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <rmw/qos_profiles.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
#include <builtin_interfaces/msg/time.hpp>

#include <onnxruntime_cxx_api.h>

#include "vl_mapper_interface/srv/detect_at_stamp.hpp"
#include "vl_mapper_interface/srv/tokenize_labels.hpp"
#include "yolo_onnx/onnx_utils.hpp"

namespace yolo_onnx
{

// Key and hash for timestamp-based cache
using StampKey = std::pair<int32_t, uint32_t>;  // (sec, nanosec)
struct StampKeyHash {
    std::size_t operator()(const StampKey & k) const noexcept {
        uint64_t v = (static_cast<uint64_t>(static_cast<uint32_t>(k.first)) << 32) |
                      static_cast<uint64_t>(k.second);
        return std::hash<uint64_t>{}(v);
    }
};

class YoloOnnxNode : public rclcpp::Node {
private:
    // Parameters
    std::string clip_onnx_path_;
    std::string vision_onnx_path_;
    std::string rgb_topic_;
    std::string yolo_service_name_;
    std::string tokenizer_service_name_;
    bool use_tensorrt_ {false};
    int cache_size_ {0};
    int stamp_tolerance_ms_ {0};
    float conf_threshold_ {0.f};
    float iou_threshold_ {0.f};
    int64_t max_labels_ {0};
    int64_t seq_len_ {0};
    int64_t target_h_ {0};
    int64_t target_w_ {0};
    int64_t embed_dim_ {0};
    int64_t total_clip_out_ {0};
    std::array<int64_t, 3> vision_out_shape_ {0,0,0};
    int64_t total_vision_out_ {0};

    // ROS interfaces
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
    rclcpp::Service<vl_mapper_interface::srv::DetectAtStamp>::SharedPtr srv_;
    rclcpp::CallbackGroup::SharedPtr cb_group_;
    rclcpp::Client<vl_mapper_interface::srv::TokenizeLabels>::SharedPtr tok_client_;

    // Image cache (LRU)
    std::deque<StampKey> lru_order_;
    std::unordered_map<StampKey, sensor_msgs::msg::Image::SharedPtr, 
    struct StampKeyHash> rgb_cache_;  // custom hash below
    std::mutex cache_mtx_;

    // Track labels
    std::vector<std::string> prev_labels_{};
    std::mutex model_mtx_;
    std::vector<float> last_embeddings_{};  // cached [N*embed_dim]
    size_t last_token_count_ {0};
    bool have_cached_emb_ {false};

    // ONNX Runtime
    Ort::Env ort_env_ {ORT_LOGGING_LEVEL_WARNING, "yolo_onnx"};
    Ort::SessionOptions ort_opts_ {};
    std::unique_ptr<Ort::Session> clip_sess_;
    std::unique_ptr<Ort::Session> vision_sess_;
    Ort::AllocatorWithDefaultOptions allocator_{};

    // IO names cache
    std::string clip_in_name_ {"input_ids"};
    std::string clip_out_name_ {"text_embeddings"};
    std::string vision_in_image_ {"image"};
    std::string vision_in_text_ {"text_embeddings"};
    std::string vision_out_name_ {"raw_predictions"};

    // YOLO anchor generation parameters
    std::vector<int> strides_ {8, 16, 32};  // YOLO strides
    std::vector<std::array<int, 2>> anchor_grid_ {
        {80, 80}, {40, 40}, {20, 20}  // Grid sizes for 640x640 input
    };

public:
    explicit YoloOnnxNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~YoloOnnxNode() override;

private:
    // --- ROS setup ---
    void setup_parameters();
    void setup_interfaces();

    // --- ROS callbacks ---
    void rgb_callback(
        const sensor_msgs::msg::Image::SharedPtr msg);
    void handle_detect(
        const std::shared_ptr<vl_mapper_interface::srv::DetectAtStamp::Request> request,
        std::shared_ptr<vl_mapper_interface::srv::DetectAtStamp::Response> response);

    // --- setup and run ONNX ---
    void init_onnx();

    // --- Text encoder ---
    std::optional<std::vector<float>> get_embeddings_for_labels(
        const std::vector<std::string>& labels,
        std::string& err_msg);
    std::vector<float> run_text_encoder(
        const std::vector<int64_t>& token_ids, 
        size_t token_count);

    // --- Vision encoder ---
    std::vector<float> preprocess_image_to_nchw(
        const sensor_msgs::msg::Image& img_msg, 
        const int target_h,
        const int target_w,
        float& scale_x,
        float& scale_y,
        int& pad_x,
        int& pad_y);
    std::vector<std::array<float, 6>> run_vision(
        const std::vector<float>& image_nchw,
        const std::vector<float>& text_embeddings,
        const size_t num_valid_labels);
    std::vector<std::array<float, 6>> postprocess_raw_predictions(
        const std::vector<float>& raw_predictions,
        const size_t num_valid_labels);
    vision_msgs::msg::Detection2DArray postprocess_detections(
        const std::vector<std::array<float, 6>>& dets, 
        const sensor_msgs::msg::Image& src_img, 
        const float scale_x, 
        const float scale_y, 
        const int pad_x, 
        const int pad_y);
    
    // --- Helpers ---
    std::optional<sensor_msgs::msg::Image::SharedPtr> get_image_for_stamp(
        const builtin_interfaces::msg::Time& t);
    std::vector<int> non_max_suppression(
        const std::vector<std::array<float, 6>>& detections);
    float calculate_iou(
        const std::array<float, 6>& det1, 
        const std::array<float, 6>& det2);
    void get_model_dims();
    static StampKey make_key(const builtin_interfaces::msg::Time& t) {
        return {t.sec, t.nanosec};
    }
};

}  // namespace yolo_onnx

