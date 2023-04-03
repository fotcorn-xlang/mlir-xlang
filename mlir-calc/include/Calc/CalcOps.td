//===- CalcOps.td - Calc dialect ops -----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CALC_OPS
#define CALC_OPS

include "Calc/CalcDialect.td"
//include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ConstantOp : Calc_Op<"constant",
    [ConstantLike, Pure]> {
  let summary = "constant";
  let arguments = (ins I64Attr:$value);
  let results = (outs I64);
  let builders = [
    OpBuilder<(ins "uint64_t":$value)>
  ];
}

def AddOp : Calc_Op<"add"> {
  let summary = "add";
  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64);
}

def SubOp : Calc_Op<"sub"> {
  let summary = "sub";
  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64);
}

def MulOp : Calc_Op<"mul"> {
  let summary = "mul";
  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64);
}

def DivOp : Calc_Op<"div"> {
  let summary = "div";
  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64);
}

def GetVariableOp : Calc_Op<"getVar"> {
  let summary = "getVar";
  let arguments = (ins StrAttr:$varName);
  let results = (outs I64);
}

def SetVariableOp : Calc_Op<"setVar"> {
  let summary = "setVar";
  let arguments = (ins StrAttr:$varName, I64:$value);
}

def EvaluateOp : Calc_Op<"evaluate"> {
  let summary = "evaluate";
  let arguments = (I64:$value);
}

#endif // CALC_OPS
