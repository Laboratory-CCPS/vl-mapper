#ifndef YOLO_ONNX_UTILS_HPP
#define YOLO_ONNX_UTILS_HPP

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>


namespace yolo_onnx
{

inline void log_shape_info(const std::string& node_name, size_t index, Ort::TypeInfo& type_info) {
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = tensor_info.GetShape();

    std::cout << "Node Name: " << node_name << " (Index: " << index << ")" << std::endl;
    std::cout << "  - Rank (number of dimensions): " << shape.size() << std::endl;
    
    std::string shape_str = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_str += std::to_string(shape[i]);
        if (i < shape.size() - 1) {
            shape_str += ", ";
        }
    }
    shape_str += "]";
    
    std::cout << "  - Shape: " << shape_str << std::endl;

    // Interpretation
    if (shape.empty()) {
        std::cout << "  - Note: This might be a scalar value." << std::endl;
    }
    for (int64_t dim : shape) {
        if (dim <= 0) {
            std::cout << "  - Note: Found a dynamic dimension (" << dim << "). Its size must be specified at runtime." << std::endl;
            break;
        }
    }
}


inline void inspect_model_dimensions(Ort::Session* session) {
    if (!session) {
        std::cerr << "Session is null!" << std::endl;
        return;
    }

    try {
        Ort::AllocatorWithDefaultOptions allocator;

        // --- Inspect Inputs ---
        size_t input_count = session->GetInputCount();
        std::cout << "\n--- Model Inputs (" << input_count << ") ---" << std::endl;
        for (size_t i = 0; i < input_count; ++i) {
            Ort::AllocatedStringPtr name_alloc = session->GetInputNameAllocated(i, allocator);
            std::string input_name = name_alloc.get();
            
            Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
            log_shape_info(input_name, i, type_info);
        }

        // --- Inspect Outputs ---
        size_t output_count = session->GetOutputCount();
        std::cout << "\n--- Model Outputs (" << output_count << ") ---" << std::endl;
        for (size_t i = 0; i < output_count; ++i) {
            Ort::AllocatedStringPtr name_alloc = session->GetOutputNameAllocated(i, allocator);
            std::string output_name = name_alloc.get();

            Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
            log_shape_info(output_name, i, type_info);
        }
        std::cout << "\n" << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Exception while inspecting model: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception while inspecting model: " << e.what() << std::endl;
    }
}

} // namespace yolo_onnx

#endif // YOLO_ONNX_UTILS_HPP