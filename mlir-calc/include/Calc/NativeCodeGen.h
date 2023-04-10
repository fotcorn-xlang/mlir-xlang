#ifndef CALC_NATIVECODEGEN_H
#define CALC_NATIVECODEGEN_H

#include <memory>
#include <string>

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

namespace mlir {
class ModuleOp;
}

namespace calc {
llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
generateNativeBinary(const mlir::ModuleOp &module);
}
#endif // CALC_NATIVECODEGEN_H
