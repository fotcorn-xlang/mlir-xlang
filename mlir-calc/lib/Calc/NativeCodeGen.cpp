#include "Calc/NativeCodeGen.h"

#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/MemoryBuffer.h"

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
calc::generateNativeBinary(const mlir::ModuleOp &module) {
  return llvm::createStringError(
      llvm::errc::invalid_argument,
      "Failed to compile LLVM IR module to native object code.");
}
