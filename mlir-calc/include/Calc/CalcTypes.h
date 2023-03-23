//===- CalcTypes.h - Calc dialect types -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CALC_CALCTYPES_H
#define CALC_CALCTYPES_H

#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Calc/CalcOpsTypes.h.inc"

#endif // CALC_CALCTYPES_H
