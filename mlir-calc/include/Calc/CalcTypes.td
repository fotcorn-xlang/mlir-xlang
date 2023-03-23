//===- CalcTypes.td - Calc dialect types -------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CALC_TYPES
#define CALC_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Calc/CalcDialect.td"

//===----------------------------------------------------------------------===//
// Calc type definitions
//===----------------------------------------------------------------------===//

class Calc_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Calc_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Calc_CustomType : Calc_Type<"Custom", "custom"> {
    let summary = "Calc custom type";
    let description = "Custom type in calc dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // CALC_TYPES
