//===- CalcTypes.cpp - Calc dialect types -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Calc/CalcTypes.h"

#include "Calc/CalcDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::calc;

#define GET_TYPEDEF_CLASSES
#include "Calc/CalcOpsTypes.cpp.inc"

void CalcDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Calc/CalcOpsTypes.cpp.inc"
      >();
}
