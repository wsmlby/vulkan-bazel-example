#include "operator_registry.hpp"
#include <stdexcept>
#include <iostream>

namespace onnxrt {

OperatorRegistry& OperatorRegistry::instance() {
    static OperatorRegistry registry;
    return registry;
}

void OperatorRegistry::registerOp(const std::string& opType, CreateFn createFn) {
    registry_[opType] = std::move(createFn);
}

std::unique_ptr<Operator> OperatorRegistry::createOp(const std::string& opType) {
    auto it = registry_.find(opType);
    if (it == registry_.end()) {
        return nullptr;
    }
    return it->second();
}

bool OperatorRegistry::hasOp(const std::string& opType) const {
    return registry_.find(opType) != registry_.end();
}

std::vector<std::string> OperatorRegistry::listOps() const {
    std::vector<std::string> ops;
    for (const auto& [name, _] : registry_) {
        ops.push_back(name);
    }
    return ops;
}

} // namespace onnxrt
