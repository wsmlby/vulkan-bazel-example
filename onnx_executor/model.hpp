#pragma once

#include "tensor.hpp"
#include <map>
#include <memory>
#include <variant>

namespace onnxrt {

// Represents a value (tensor) in the computation graph
struct Value {
    std::string name;
    Shape shape;
    DataType dtype = DataType::FLOAT32;
};

// Represents an ONNX operator node
struct Node {
    std::string name;
    std::string opType;                              // "Add", "Conv", "MatMul", etc.
    std::vector<std::string> inputs;                 // Input value names
    std::vector<std::string> outputs;                // Output value names

    // Operator attributes (key -> value)
    using AttrValue = std::variant<
        int64_t,
        float,
        std::string,
        std::vector<int64_t>,
        std::vector<float>,
        std::vector<std::string>
    >;
    std::map<std::string, AttrValue> attributes;

    // Helper to get attributes with type checking
    template<typename T>
    T getAttr(const std::string& key, const T& defaultVal = T{}) const {
        auto it = attributes.find(key);
        if (it == attributes.end()) return defaultVal;
        if (auto* val = std::get_if<T>(&it->second)) return *val;
        return defaultVal;
    }

    bool hasAttr(const std::string& key) const {
        return attributes.find(key) != attributes.end();
    }
};

/**
 * Model - In-memory representation of an ONNX model
 *
 * Stores the computation graph, initializers (weights), and metadata.
 */
class Model {
public:
    Model() = default;

    // Load model from .onnx file
    static Model load(const std::string& onnxPath, vkcompute::Context& ctx);

    // Graph inputs (excluding initializers - these are the real inputs)
    const std::vector<Value>& inputs() const { return inputs_; }

    // Graph outputs
    const std::vector<Value>& outputs() const { return outputs_; }

    // Computation nodes in topological order
    const std::vector<Node>& nodes() const { return nodes_; }

    // Initializers (constant weights/biases)
    std::map<std::string, Tensor>& initializers() { return initializers_; }
    const std::map<std::string, Tensor>& initializers() const { return initializers_; }

    // Check if a value is an initializer
    bool isInitializer(const std::string& name) const;

    // Get value info by name (returns nullptr if not found)
    const Value* getValueInfo(const std::string& name) const;

    // Get tensor shape by name (from value info or initializer)
    Shape getShape(const std::string& name) const;

private:
    std::vector<Value> inputs_;
    std::vector<Value> outputs_;
    std::vector<Node> nodes_;
    std::map<std::string, Tensor> initializers_;
    std::map<std::string, Value> valueInfos_;  // All value metadata
};

} // namespace onnxrt
