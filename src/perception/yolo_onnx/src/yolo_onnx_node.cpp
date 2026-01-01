#include "yolo_onnx/yolo_onnx_node.hpp"
#include <rmw/qos_profiles.h>

namespace yolo_onnx
{


YoloOnnxNode::YoloOnnxNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("yolo_onnx_node", options) {
    RCLCPP_INFO(this->get_logger(), "Starting YOLO ONNX Node...");
    setup_parameters();
    init_onnx();
    setup_interfaces();
    RCLCPP_INFO(this->get_logger(), "YOLO ONNX service ready at %s", yolo_service_name_.c_str());
}


YoloOnnxNode::~YoloOnnxNode() = default;


void YoloOnnxNode::setup_parameters() {
    clip_onnx_path_ = this->declare_parameter<std::string>(
        "clip_model", "/home/josua/yolo_world_models/onnx_exports/clip_text_encoder.onnx");
    vision_onnx_path_ = this->declare_parameter<std::string>(
        "vision_model", "/home/josua/yolo_world_models/onnx_exports/yoloworld_vision.onnx");
    rgb_topic_ = this->declare_parameter<std::string>(
        "rgb_topic", "/camera/color/image_raw");
    yolo_service_name_ = this->declare_parameter<std::string>(
        "yolo_service", "/detect_labels");
    tokenizer_service_name_ = this->declare_parameter<std::string>(
        "tokenizer_service", "/tokenizer_service/tokenize");
    use_tensorrt_ = this->declare_parameter<bool>("use_tensorrt", false);
    cache_size_ = this->declare_parameter<int>("cache_size", 20);
    stamp_tolerance_ms_ = this->declare_parameter<int>("stamp_tolerance_ms", 0);
    conf_threshold_ = this->declare_parameter<double>("conf_threshold", 0.35);
    iou_threshold_ = this->declare_parameter<double>("iou_threshold", 0.5);
}


void YoloOnnxNode::setup_interfaces() {
    cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.callback_group = cb_group_;
    rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        rgb_topic_, rclcpp::SensorDataQoS(),
        std::bind(&YoloOnnxNode::rgb_callback, this, std::placeholders::_1), 
        sub_opts);

    // Put the service into the Reentrant callback group so it can invoke
    // the tokenizer client from within the service callback without deadlocking.
    srv_ = this->create_service<vl_mapper_interface::srv::DetectAtStamp>(
        yolo_service_name_,
        std::bind(&YoloOnnxNode::handle_detect, this, std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default,
        cb_group_);

    // Create tokenizer client in the same Reentrant callback group
    tok_client_ = this->create_client<vl_mapper_interface::srv::TokenizeLabels>(
        tokenizer_service_name_, rmw_qos_profile_services_default, cb_group_);
    if (!tok_client_->wait_for_service(std::chrono::seconds(2))) {
        RCLCPP_WARN(this->get_logger(), "Tokenizer service %s not available yet; will retry on demand", tokenizer_service_name_.c_str());
    }
}


// --- ROS Callbacks ---
void YoloOnnxNode::rgb_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    StampKey key {msg->header.stamp.sec, msg->header.stamp.nanosec};
    std::lock_guard<std::mutex> lk(cache_mtx_);
    // insert or refresh in LRU
    auto it = rgb_cache_.find(key);
    if (it == rgb_cache_.end()) {
        rgb_cache_[key] = msg;
        lru_order_.push_back(key);
        while (rgb_cache_.size() > static_cast<size_t>(std::max(1, cache_size_))) {
            auto old = lru_order_.front();
            lru_order_.pop_front();
            rgb_cache_.erase(old);
        }
    } else {
        it->second = msg;
        // move key to back
        auto pos = std::find(lru_order_.begin(), lru_order_.end(), key);
        if (pos != lru_order_.end()) {
            lru_order_.erase(pos);
        }
        lru_order_.push_back(key);
    }
}


void YoloOnnxNode::handle_detect(
        const std::shared_ptr<vl_mapper_interface::srv::DetectAtStamp::Request> request,
        std::shared_ptr<vl_mapper_interface::srv::DetectAtStamp::Response> response
) {
    auto img_opt = this->get_image_for_stamp(request->stamp);
    if (!img_opt.has_value()) {
        size_t cache_size = 0;
        {
            std::lock_guard<std::mutex> lk(cache_mtx_);
            cache_size = rgb_cache_.size();
        }
        RCLCPP_DEBUG(this->get_logger(), "Detect request rejected: no cached RGB frame for stamp %d.%09u (cache_size=%zu)",
            request->stamp.sec, request->stamp.nanosec, cache_size);
        response->success = false;
        response->message = "No RGB frame for requested stamp";
        response->detections = vision_msgs::msg::Detection2DArray();
        return;
    }
    const auto& img = *img_opt.value();

    // Prepare labels
    std::vector<std::string> labels(request->labels.labels.begin(), request->labels.labels.end());
    if (labels.empty()) {
        response->success = true;
        response->message = "ok (no labels)";
        response->detections = vision_msgs::msg::Detection2DArray();
        return;
    }

    const size_t num_valid_labels = std::min(labels.size(), static_cast<size_t>(std::max<int64_t>(1, max_labels_)));
    if (labels.size() > static_cast<size_t>(max_labels_)) {
        RCLCPP_WARN(this->get_logger(), "Truncating labels from %zu to max_labels=%ld", labels.size(), (long)max_labels_);
        labels.resize(static_cast<size_t>(max_labels_));
    } else if (labels.size() < static_cast<size_t>(max_labels_)) {
        RCLCPP_DEBUG(this->get_logger(), "Padding labels from %zu to max_labels=%ld", labels.size(), (long)max_labels_);
        labels.resize(static_cast<size_t>(max_labels_), std::string(""));
    }

    const size_t label_count = num_valid_labels;

    // Get text embeddings via helper
    std::string emb_err;
    auto emb_opt = this->get_embeddings_for_labels(labels, emb_err);
    if (!emb_opt.has_value()) {
        response->success = false;
        response->message = emb_err;
        response->detections = vision_msgs::msg::Detection2DArray();
        return;
    }
    const auto& embeddings = emb_opt.value();

    // Preprocess image and run vision
    float scale_x = 1.f, scale_y = 1.f;
    int pad_x = 0, pad_y = 0;
    auto nchw = this->preprocess_image_to_nchw(img, target_h_, target_w_, scale_x, scale_y, pad_x, pad_y);
    auto dets = this->run_vision(nchw, embeddings, label_count);

    // Postprocess
    response->detections = this->postprocess_detections(dets, img, scale_x, scale_y, pad_x, pad_y);
    response->success = true;
    response->message = "ok";
}


// --- ONNX Runtime setup and inference ---
void YoloOnnxNode::init_onnx() {
    ort_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    if (use_tensorrt_) {
        const auto& api = Ort::GetApi();
        OrtTensorRTProviderOptionsV2* tensorrt_options = nullptr;

        try {
            Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));

            std::vector<const char*> option_keys = {
                "device_id",
                "trt_fp16_enable",
                "trt_dump_subgraphs",
                // recommended options
                "trt_engine_cache_enable",
                "trt_engine_cache_path",
                "trt_timing_cache_enable",
                "trt_timing_cache_path",
            };
            std::vector<const char*> option_values = {
                "0",
                "1",
                "1",
                // recommended options
                "1",
                "/home/josua/yolo_trt_cache",
                "1",
                "/home/josua/yolo_trt_cache",
            };

            Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(
                tensorrt_options, option_keys.data(), option_values.data(), option_keys.size()));
            ort_opts_.AppendExecutionProvider_TensorRT_V2(*tensorrt_options);

        } catch (const Ort::Exception& e) {
            RCLCPP_WARN(this->get_logger(), "Failed to add TensorRT Execution Provider. Falling back to CPU. Error: %s", e.what());
        }

        if (tensorrt_options) {
            api.ReleaseTensorRTProviderOptions(tensorrt_options);
        }
    }

    try {
        RCLCPP_INFO(this->get_logger(), "Loading CLIP model from: %s", clip_onnx_path_.c_str());
        clip_sess_ = std::make_unique<Ort::Session>(ort_env_, clip_onnx_path_.c_str(), ort_opts_);

        RCLCPP_INFO(this->get_logger(), "Loading Vision model from: %s", vision_onnx_path_.c_str());
        vision_sess_ = std::make_unique<Ort::Session>(ort_env_, vision_onnx_path_.c_str(), ort_opts_);

    } catch (const Ort::Exception& e) {
        RCLCPP_FATAL(this->get_logger(), "Failed to create ONNX session: %s", e.what());
        throw; 
    }
    
    inspect_model_dimensions(clip_sess_.get());
    inspect_model_dimensions(vision_sess_.get());
    this->get_model_dims();
}


// --- Text encoder ---
std::optional<std::vector<float>>
YoloOnnxNode::get_embeddings_for_labels(
        const std::vector<std::string>& labels,
        std::string& err_msg
) {
    // Validate tokenizer client
    if (!tok_client_) {
        err_msg = "Tokenizer client not initialized";
        return std::nullopt;
    }

    // Check cached labels are the same
    {
        std::lock_guard<std::mutex> lk(model_mtx_);
        if (have_cached_emb_ && labels == prev_labels_) {
            return last_embeddings_;
        }
    }

    // The exported text encoder expects a fixed batch size.
    if (labels.size() != static_cast<size_t>(std::max<int64_t>(1, max_labels_))) {
        err_msg = "labels batch size mismatch: got=" + std::to_string(labels.size()) +
                  " expected=" + std::to_string(static_cast<size_t>(std::max<int64_t>(1, max_labels_)));
        return std::nullopt;
    }

    // Ensure service available
    if (!tok_client_->wait_for_service(std::chrono::milliseconds(500))) {
        err_msg = "Tokenizer service unavailable";
        return std::nullopt;
    }

    // Tokenize via service
    auto req = std::make_shared<vl_mapper_interface::srv::TokenizeLabels::Request>();
    req->labels.labels = labels;
    RCLCPP_DEBUG(this->get_logger(), "Sending tokenizer request for %zu labels", labels.size());
    auto fut = tok_client_->async_send_request(req);
    auto status = fut.wait_for(std::chrono::seconds(5));
    if (status != std::future_status::ready) {
        err_msg = "Tokenizer request timed out";
        return std::nullopt;
    }
    auto resp = fut.get();
    if (!resp->success) {
        err_msg = resp->message.empty() ? "Tokenizer request failed" : resp->message;
        return std::nullopt;
    }
    if (resp->token_count <= 0 || static_cast<size_t>(resp->token_ids.size()) != static_cast<size_t>(resp->token_count) * static_cast<size_t>(seq_len_)) {
        err_msg = "Tokenizer returned invalid shape";
        return std::nullopt;
    }

    if (static_cast<int64_t>(resp->token_count) != max_labels_) {
        err_msg = "Tokenizer returned token_count=" + std::to_string(resp->token_count) +
                  " but model requires max_labels=" + std::to_string(max_labels_);
        return std::nullopt;
    }

    // Run text encoder
    try {
        auto embeddings = this->run_text_encoder(resp->token_ids, static_cast<size_t>(resp->token_count));
        {
            std::lock_guard<std::mutex> lk(model_mtx_);
            last_embeddings_ = embeddings;
            last_token_count_ = static_cast<size_t>(resp->token_count);
            prev_labels_ = labels;
            have_cached_emb_ = true;
        }
        return embeddings;
    } catch (const std::exception& e) {
        err_msg = std::string("Text encoder failed: ") + e.what();
        return std::nullopt;
    }
}


std::vector<float>
YoloOnnxNode::run_text_encoder(const std::vector<int64_t>& token_ids, size_t token_count) {
    if (static_cast<int64_t>(token_count) != max_labels_) {
        throw std::runtime_error(
            "text encoder requires fixed batch size max_labels=" + std::to_string(max_labels_) +
            ", got token_count=" + std::to_string(token_count));
    }
    // token_ids shape: [token_count, seq_len_]
    std::array<int64_t, 2> in_shape {static_cast<int64_t>(token_count), static_cast<int64_t>(seq_len_)};
    size_t n_elems = static_cast<size_t>(in_shape[0] * in_shape[1]);
    if (token_ids.size() != n_elems) {
        throw std::runtime_error("token count mismatch for text encoder input");
    }

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<int64_t>(mem_info, const_cast<int64_t*>(token_ids.data()), n_elems, in_shape.data(), in_shape.size());
    const char* in_names[] = {clip_in_name_.c_str()};
    const char* out_names[] = {clip_out_name_.c_str()};
    auto outputs = clip_sess_->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
   
    std::vector<float> embeddings(total_clip_out_);
    auto& out_tensor = outputs.front();
    std::memcpy(embeddings.data(), out_tensor.GetTensorMutableData<float>(), total_clip_out_ * sizeof(float));
    return embeddings;
}


// --- Vision encoder ---
std::vector<float>
YoloOnnxNode::preprocess_image_to_nchw(
        const sensor_msgs::msg::Image& img_msg,
        const int target_h,
        const int target_w,
        float& scale_x,
        float& scale_y,
        int& pad_x,
        int& pad_y
) {
    if (img_msg.encoding != "rgb8" && img_msg.encoding != "bgr8") {
        throw std::runtime_error("Expected rgb8 or bgr8 image encoding");
    }
    const int H = static_cast<int>(img_msg.height);
    const int W = static_cast<int>(img_msg.width);
    cv::Mat src(H, W, CV_8UC3, const_cast<uint8_t*>(img_msg.data.data()));
    cv::Mat rgb;
    if (img_msg.encoding == "rgb8") {
        rgb = src; // already RGB
    } else {
        cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
    }
    float r = std::min(static_cast<float>(target_w) / static_cast<float>(W), static_cast<float>(target_h) / static_cast<float>(H));
    int new_w = static_cast<int>(std::round(W * r));
    int new_h = static_cast<int>(std::round(H * r));
    pad_x = (target_w - new_w) / 2;
    pad_y = (target_h - new_h) / 2;
    scale_x = static_cast<float>(W) / static_cast<float>(new_w);
    scale_y = static_cast<float>(H) / static_cast<float>(new_h);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat canvas(target_h, target_w, CV_8UC3, cv::Scalar(114,114,114));
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, new_w, new_h)));

    // to float NCHW [1,3,H,W]
    std::vector<float> nchw(static_cast<size_t>(1 * 3 * target_h * target_w));
    size_t plane = static_cast<size_t>(target_h * target_w);
    for (int y = 0; y < target_h; ++y) {
        const uint8_t* row = canvas.ptr<uint8_t>(y);
        for (int x = 0; x < target_w; ++x) {
            size_t idx = static_cast<size_t>(y * target_w + x);
            // canvas is RGB
            nchw[0 * plane + idx] = static_cast<float>(row[x*3 + 0]) / 255.0f;
            nchw[1 * plane + idx] = static_cast<float>(row[x*3 + 1]) / 255.0f;
            nchw[2 * plane + idx] = static_cast<float>(row[x*3 + 2]) / 255.0f;
        }
    }
    return nchw;
}

std::vector<std::array<float, 6>>
YoloOnnxNode::run_vision(
        const std::vector<float>& image_nchw,
        const std::vector<float>& text_embeddings,
        const size_t num_valid_labels
) {
    // image: [1,3,H,W], text: [token_count, embed_dim_]
    std::array<int64_t, 4> img_shape {1, 3, target_h_, target_w_};
    std::array<int64_t, 2> txt_shape {static_cast<int64_t>(max_labels_), static_cast<int64_t>(embed_dim_)};

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value img = Ort::Value::CreateTensor<float>(mem_info, const_cast<float*>(image_nchw.data()), image_nchw.size(), img_shape.data(), img_shape.size());
    Ort::Value txt = Ort::Value::CreateTensor<float>(mem_info, const_cast<float*>(text_embeddings.data()), text_embeddings.size(), txt_shape.data(), txt_shape.size());

    const char* in_names[] = {vision_in_image_.c_str(), vision_in_text_.c_str()};
    const char* out_names[] = {vision_out_name_.c_str()};
    const Ort::Value inputs_arr[2] = {std::move(img), std::move(txt)};
    auto outputs = vision_sess_->Run(Ort::RunOptions{nullptr}, in_names, inputs_arr, 2, out_names, 1);

    // Extract raw predictions
    auto& out = outputs.front();
    std::vector<float> raw_predictions(total_vision_out_);
    const float* ptr = out.GetTensorMutableData<float>();
    std::memcpy(raw_predictions.data(), ptr, total_vision_out_ * sizeof(float));
    
    // Post-process raw predictions to get final detections
    auto detections = this->postprocess_raw_predictions(raw_predictions, num_valid_labels);
    
    RCLCPP_DEBUG(this->get_logger(), "Post-processed detections: %zu", detections.size());
    
    return detections;
}


std::vector<std::array<float, 6>>
YoloOnnxNode::postprocess_raw_predictions(
        const std::vector<float>& raw_predictions,
        const size_t num_valid_labels
) {
    // pred_shape: [batch_size, n_classes+4, n_anchors]
    const int64_t batch_size = vision_out_shape_[0];
    const int64_t n_features = vision_out_shape_[1]; // n_classes + 4
    const int64_t n_anchors = vision_out_shape_[2];
    
    RCLCPP_DEBUG(this->get_logger(), "Post-processing: batch=%ld, features=%ld, anchors=%ld, token_count=%zu",
                batch_size, n_features, n_anchors, num_valid_labels);
    
    // Transpose from [1, n_classes+4, n_anchors] to [1, n_anchors, n_classes+4]
    std::vector<float> transposed(raw_predictions.size());
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t a = 0; a < n_anchors; ++a) {
            for (int64_t f = 0; f < n_features; ++f) {
                // Original: [b, f, a]
                // Target: [b, a, f]
                size_t src_idx = b * n_features * n_anchors + f * n_anchors + a;
                size_t dst_idx = b * n_anchors * n_features + a * n_features + f;
                transposed[dst_idx] = raw_predictions[src_idx];
            }
        }
    }
    
    std::vector<std::array<float, 6>> detections;
    
    // Process first batch only
    for (int64_t a = 0; a < n_anchors; ++a) {
        size_t base_idx = a * n_features;
        
        // Extract box coordinates [x, y, w, h]
        float x = transposed[base_idx + 0];
        float y = transposed[base_idx + 1];
        float w = transposed[base_idx + 2];
        float h = transposed[base_idx + 3];
        
        // Find best class and confidence
        float best_score = -1.0f;
        int best_class = -1;
        for (size_t c = 0; c < num_valid_labels && c < static_cast<size_t>(max_labels_); ++c) {
            float score = transposed[base_idx + 4 + c];
            if (score > best_score) {
                best_score = score;
                best_class = static_cast<int>(c);
            }
        }
        
        // Apply confidence threshold
        if (best_score > conf_threshold_) {
            // Normalize coordinates to [0, 1] range
            // YOLO predictions are relative to the 640x640 input
            float norm_x = x / static_cast<float>(target_w_);
            float norm_y = y / static_cast<float>(target_h_);
            float norm_w = w / static_cast<float>(target_w_);
            float norm_h = h / static_cast<float>(target_h_);
            
            detections.push_back({norm_x, norm_y, norm_w, norm_h, best_score, static_cast<float>(best_class)});
        }
    }
    
    RCLCPP_DEBUG(this->get_logger(), "Before NMS: %zu detections", detections.size());
    
    // Apply NMS
    auto keep_indices = non_max_suppression(detections);
    
    std::vector<std::array<float, 6>> final_detections;
    for (int idx : keep_indices) {
        final_detections.push_back(detections[idx]);
    }
    
    RCLCPP_DEBUG(this->get_logger(), "After NMS: %zu detections", final_detections.size());
    
    return final_detections;
}


vision_msgs::msg::Detection2DArray
YoloOnnxNode::postprocess_detections(
        const std::vector<std::array<float, 6>>& dets,
        const sensor_msgs::msg::Image& src_img,
        const float scale_x,
        const float scale_y,
        const int pad_x,
        const int pad_y
) {
    vision_msgs::msg::Detection2DArray array;
    array.header = src_img.header;
    const float img_w = static_cast<float>(target_w_);
    const float img_h = static_cast<float>(target_h_);
    for (const auto & d : dets) {
        float cx = d[0];  // normalized center x [0,1] from post-processed YOLO
        float cy = d[1];  // normalized center y [0,1] from post-processed YOLO
        float w = d[2];   // normalized width [0,1] from post-processed YOLO
        float h = d[3];   // normalized height [0,1] from post-processed YOLO
        float conf = d[4];
        float cls = d[5];
        if (conf < conf_threshold_) continue;

        RCLCPP_DEBUG(this->get_logger(), 
            "Detection: cx=%.3f cy=%.3f w=%.3f h=%.3f conf=%.3f cls=%.0f", 
            cx, cy, w, h, conf, cls);
            
        // Convert normalized letterbox coords to absolute letterbox coordinates
        float abs_cx = cx * img_w;  // absolute center x in letterboxed image (640x640)
        float abs_cy = cy * img_h;  // absolute center y in letterboxed image
        float abs_w = w * img_w;    // absolute width in letterboxed image
        float abs_h = h * img_h;    // absolute height in letterboxed image
        
        // Remove padding and scale back to original image coordinates
        float orig_cx = (abs_cx - pad_x) * scale_x; // center x in original image
        float orig_cy = (abs_cy - pad_y) * scale_y; // center y in original image
        float orig_w = abs_w * scale_x;             // width in original image
        float orig_h = abs_h * scale_y;             // height in original image

        
        vision_msgs::msg::Detection2D det;
        det.header = src_img.header;
        det.bbox.center.position.x = orig_cx / static_cast<float>(src_img.width);
        det.bbox.center.position.y = orig_cy / static_cast<float>(src_img.height);
        det.bbox.size_x = orig_w / static_cast<float>(src_img.width);
        det.bbox.size_y = orig_h / static_cast<float>(src_img.height);

        vision_msgs::msg::ObjectHypothesisWithPose hyp;
        hyp.hypothesis.class_id = std::to_string(static_cast<int>(cls));
        hyp.hypothesis.score = conf;
        det.results.push_back(hyp);
        array.detections.push_back(det);
    }
    return array;
}


// --- Helpers ---
std::optional<sensor_msgs::msg::Image::SharedPtr>
YoloOnnxNode::get_image_for_stamp(const builtin_interfaces::msg::Time& t) {
    StampKey target {t.sec, t.nanosec};
    std::lock_guard<std::mutex> lk(cache_mtx_);
    auto it = rgb_cache_.find(target);
    if (it != rgb_cache_.end()) {
        return it->second;
    }
    if (stamp_tolerance_ms_ <= 0 || rgb_cache_.empty()) {
        return std::nullopt;
    }
    // find nearest within tolerance
    const int64_t target_ns = static_cast<int64_t>(t.sec) * 1'000'000'000LL + static_cast<int64_t>(t.nanosec);
    const int64_t tol_ns = static_cast<int64_t>(stamp_tolerance_ms_) * 1'000'000LL;
    StampKey best_key {0, 0};
    int64_t best_delta = std::numeric_limits<int64_t>::max();
    for (const auto & k : lru_order_) {
        int64_t cur_ns = static_cast<int64_t>(k.first) * 1'000'000'000LL + static_cast<int64_t>(k.second);
        int64_t delta = std::llabs(cur_ns - target_ns);
        if (delta < best_delta) {
            best_delta = delta;
            best_key = k;
        }
    }
    if (best_delta <= tol_ns) {
        auto it2 = rgb_cache_.find(best_key);
        if (it2 != rgb_cache_.end()) {
            return it2->second;
        }
    }
    return std::nullopt;
}


std::vector<int>
YoloOnnxNode::non_max_suppression(const std::vector<std::array<float, 6>>& detections) {
    std::vector<int> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // Sort by confidence (descending)
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return detections[a][4] > detections[b][4]; // Compare confidence
    });
    
    std::vector<bool> suppressed(detections.size(), false);
    std::vector<int> keep;
    
    for (int i : indices) {
        if (suppressed[i]) continue;
        
        keep.push_back(i);
        
        // Suppress overlapping detections
        for (int j : indices) {
            if (i == j || suppressed[j]) continue;
            
            float iou = calculate_iou(detections[i], detections[j]);
            if (iou > iou_threshold_) {
                suppressed[j] = true;
            }
        }
    }
    
    return keep;
}


float 
YoloOnnxNode::calculate_iou(const std::array<float, 6>& det1, const std::array<float, 6>& det2) {
    // Convert center-width-height to corner coordinates
    float x1_min = det1[0] - det1[2] / 2.0f;  // x - w/2
    float y1_min = det1[1] - det1[3] / 2.0f;  // y - h/2
    float x1_max = det1[0] + det1[2] / 2.0f;  // x + w/2
    float y1_max = det1[1] + det1[3] / 2.0f;  // y + h/2
    
    float x2_min = det2[0] - det2[2] / 2.0f;
    float y2_min = det2[1] - det2[3] / 2.0f;
    float x2_max = det2[0] + det2[2] / 2.0f;
    float y2_max = det2[1] + det2[3] / 2.0f;
    
    // Calculate intersection
    float inter_x_min = std::max(x1_min, x2_min);
    float inter_y_min = std::max(y1_min, y2_min);
    float inter_x_max = std::min(x1_max, x2_max);
    float inter_y_max = std::min(y1_max, y2_max);
    
    float inter_width = std::max(0.0f, inter_x_max - inter_x_min);
    float inter_height = std::max(0.0f, inter_y_max - inter_y_min);
    float intersection = inter_width * inter_height;
    
    // Calculate union
    float area1 = det1[2] * det1[3];  // width * height
    float area2 = det2[2] * det2[3];
    float union_area = area1 + area2 - intersection;
    
    // Avoid division by zero
    if (union_area <= 1e-6f) return 0.0f;
    
    return intersection / union_area;
}


void YoloOnnxNode::get_model_dims() {
    auto clip_sess = clip_sess_.get();
    auto vision_sess = vision_sess_.get();
    try {
        Ort::AllocatorWithDefaultOptions allocator;

        // --- Clip Dimensions ---
        Ort::TypeInfo clip_in_type_info = clip_sess->GetInputTypeInfo(0);
        auto clip_in_info = clip_in_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> clip_in_shape = clip_in_info.GetShape();
        if (clip_in_shape.size() == 2) {
            max_labels_ = clip_in_shape[0];
            seq_len_ = clip_in_shape[1];
        } else {
            RCLCPP_ERROR(this->get_logger(), "Unexpected CLIP input shape");
        }

        Ort::TypeInfo clip_out_type_info = clip_sess->GetOutputTypeInfo(0);
        auto clip_out_info = clip_out_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> clip_out_shape = clip_out_info.GetShape();
        if (clip_out_shape.size() == 2 && clip_out_shape[0] == clip_in_shape[0]) {
            embed_dim_ = clip_out_shape[1];
            total_clip_out_ = clip_out_shape[0] * clip_out_shape[1];
        } else {
            RCLCPP_ERROR(this->get_logger(), "Unexpected CLIP output shape");
        }


        // --- Vision Dimensions ---
        Ort::TypeInfo vision_in_type_info = vision_sess->GetInputTypeInfo(0);
        auto vision_in_info = vision_in_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> vision_in_shape = vision_in_info.GetShape();
        if (vision_in_shape.size() == 4) {
            target_h_ = vision_in_shape[2];
            target_w_ = vision_in_shape[3];
        } else {
            RCLCPP_ERROR(this->get_logger(), "Unexpected Vision input shape");
        }

        Ort::TypeInfo vision_out_type_info = vision_sess->GetOutputTypeInfo(0);
        auto vision_out_info = vision_out_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> vision_out_shape = vision_out_info.GetShape();
        if (vision_out_shape.size() == 3 && (clip_in_shape[0] + 4) == vision_out_shape[1]) {
            vision_out_shape_ = {
                vision_out_shape[0], 
                vision_out_shape[1], 
                vision_out_shape[2]
            };
            total_vision_out_ = vision_out_shape[0] * vision_out_shape[1] * vision_out_shape[2];
        } else {
            RCLCPP_ERROR(this->get_logger(), "Unexpected Vision output shape");
        }

    } catch (const Ort::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Getting the model dimensions failed: %s", e.what());
    }
}

}  // namespace yolo_onnx


int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<yolo_onnx::YoloOnnxNode>();
    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();
    rclcpp::shutdown();
    return 0;
}

