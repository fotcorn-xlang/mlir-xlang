#include <iostream>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

int main() {
  mlir::MLIRContext context;
  
  mlir::ModuleOp mod = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  // create function with no arguments, i32 return type
  mlir::FunctionType funcType = mlir::FunctionType::get(
      &context, {}, {}/*mlir::IntegerType::get(&context, 32)*/);
  mlir::FuncOp func =
      mlir::FuncOp::create(mlir::UnknownLoc::get(&context), "main", funcType, {});

  mod.push_back(func);
  mlir::Block *entryBlock = func.addEntryBlock();
  
  mlir::OpBuilder builder(&context);
  builder.setInsertionPointToStart(entryBlock);

  // mlir::Value op1 =
  // builder.create<mlir::ConstantIntOp>(builder.getUnknownLoc(), 13, 32);
  // mlir::Value op2 =
  // builder.create<mlir::ConstantIntOp>(builder.getUnknownLoc(), 29, 32);

  

  // return 0
  mlir::Value retVal = builder.create<mlir::arith::ConstantIntOp>(
      builder.getUnknownLoc(), 0, 32);
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), retVal);

  // Verify the function and module.
  if (mlir::failed(verify(func)) || mlir::failed(mlir::verify(mod))) {
    llvm::errs() << "Error: module verification failed\n";
    return 1;
  }

  // Print the module.
  mod.dump();

  return 0;
}
