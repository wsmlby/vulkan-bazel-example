#pragma once

#include "operator_base.hpp"
#include <functional>
#include <map>
#include <memory>

namespace onnxrt {

/**
 * OperatorRegistry - Factory for creating operators by name
 */
class OperatorRegistry {
public:
    using CreateFn = std::function<std::unique_ptr<Operator>()>;

    // Get singleton instance
    static OperatorRegistry& instance();

    // Register an operator type
    void registerOp(const std::string& opType, CreateFn createFn);

    // Create an operator instance
    std::unique_ptr<Operator> createOp(const std::string& opType);

    // Check if operator is supported
    bool hasOp(const std::string& opType) const;

    // List all registered operators
    std::vector<std::string> listOps() const;

private:
    OperatorRegistry() = default;
    std::map<std::string, CreateFn> registry_;
};

// Helper macro for registering operators
#define REGISTER_OPERATOR(OpTypeName, OpClass)                                \
    static bool _registered_##OpClass = []() {                                \
        OperatorRegistry::instance().registerOp(OpTypeName, []() {            \
            return std::make_unique<OpClass>();                               \
        });                                                                   \
        return true;                                                          \
    }()

} // namespace onnxrt
