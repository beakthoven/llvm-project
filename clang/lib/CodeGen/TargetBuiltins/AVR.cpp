//===------ AVR.cpp - Emit LLVM Code for AVR builtins ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Builtin calls as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CGBuiltin.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicsAVR.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm;

/// Emit an inline-asm-based fractional multiply (fmul/fmuls/fmulsu).
/// All three variants share the same shape: two i8 inputs → one i16 output,
/// with the result collected from R1:R0 via movw, then R1 cleared.
static Value *EmitAVRFMulInlineAsm(CodeGenFunction &CGF, const CallExpr *E,
                                   const char *AsmInsn) {
  Value *Arg0 = CGF.EmitScalarExpr(E->getArg(0));
  Value *Arg1 = CGF.EmitScalarExpr(E->getArg(1));
  llvm::LLVMContext &Ctx = CGF.getLLVMContext();
  llvm::Type *ResTy = llvm::Type::getInt16Ty(Ctx);
  llvm::Type *ArgTy = llvm::Type::getInt8Ty(Ctx);
  llvm::FunctionType *FTy =
      llvm::FunctionType::get(ResTy, {ArgTy, ArgTy}, false);

  // Fallback to libcall if the target does not support the hardware multiplier.
  // The libgcc implementation of __fmul/__fmuls/__fmulsu does NOT use the
  // standard C ABI. It expects arguments in r24 and r25, and returns the
  // result in r22:r23. We must emit inline asm to bind these registers.
  if (!CGF.getTarget().hasFeature("mul") || !CGF.getTarget().hasFeature("movw")) {
    std::string FuncName = std::string("__") + AsmInsn;
    // ${0:A} and ${0:B} select the lo/hi bytes of the i16 output register
    // pair; handled by AVRAsmPrinter::PrintAsmOperand (A=byte 0, B=byte 1).
    std::string Asm = "mov r25, $2\n\trcall " + FuncName +
                      "\n\tmov ${0:A}, r22\n\tmov ${0:B}, r23";
    // Clobbers: r0 (__tmp_reg__), r22:r23 (return value read then clobbered),
    // r24 (input but destroyed by callee), r25 (set by mov above).
    llvm::InlineAsm *IA = llvm::InlineAsm::get(
        FTy, Asm, "=&r,{r24},r,~{r0},~{r22},~{r23},~{r24},~{r25}", true);
    return CGF.Builder.CreateCall(IA, {Arg0, Arg1});
  }

  // Build the asm string: "<insn> $1, $2\n\tmovw $0, r0\n\tclr r1"
  std::string Asm = std::string(AsmInsn) + " $1, $2\n\tmovw $0, r0\n\tclr r1";
  llvm::InlineAsm *IA =
      llvm::InlineAsm::get(FTy, Asm, "=r,a,a,~{r0},~{r1}", true);
  return CGF.Builder.CreateCall(IA, {Arg0, Arg1});
}

/// Emit __builtin_avr_delay_cycles(N).
///
/// Generates an optimal sequence of inline assembly delay loops and NOPs
/// to consume exactly N clock cycles.
///
/// The decomposed N into a sum of contributions from nested loops
/// of decreasing register width, then fills the remainder with rjmp/.+0
/// (2 cycles) and nop (1 cycle).
///
/// Loop types:
///   4-byte loop: ldi×4 + (subi + sbci×3 + brne) = 9 setup + 6/iter
///   3-byte loop: ldi×3 + (subi + sbci×2 + brne) = 7 setup + 5/iter
///   2-byte loop: ldi×2 + (sbiw + brne)           = 5 setup + 4/iter
///   1-byte loop: ldi   + (dec  + brne)            = 3/iter (no setup overhead)
static Value *EmitAVRDelayLoops(CodeGenFunction &CGF, uint32_t Cycles) {
  if (Cycles == 0)
    return nullptr;

  std::string Asm;
  std::string Clobbers;
  unsigned ClobberIdx = 0;
  unsigned LabelIdx = 1;

  auto AddClobber = [&](unsigned Reg) {
    if (!Clobbers.empty())
      Clobbers += ",";
    Clobbers += "~{r" + std::to_string(Reg) + "}";
  };

  // 4-byte loop: 9 + 6*(loop_count-1) cycles
  // ldi×4 + (subi + sbci×3 + brne) per iteration
  if (Cycles >= 83886082u) {
    uint32_t LoopCount = ((Cycles - 9) / 6) + 1;
    uint32_t Used = ((LoopCount - 1) * 6) + 9;
    unsigned Base = 16 + ClobberIdx;
    std::string L = std::to_string(LabelIdx++);
    Asm += "ldi r" + std::to_string(Base) + ", lo8(" +
           std::to_string(LoopCount) + ")\n\t";
    Asm += "ldi r" + std::to_string(Base + 1) + ", hi8(" +
           std::to_string(LoopCount) + ")\n\t";
    Asm += "ldi r" + std::to_string(Base + 2) + ", hlo8(" +
           std::to_string(LoopCount) + ")\n\t";
    Asm += "ldi r" + std::to_string(Base + 3) + ", hhi8(" +
           std::to_string(LoopCount) + ")\n\t";
    Asm += L + ": subi r" + std::to_string(Base) + ", 1\n\t";
    Asm += "sbci r" + std::to_string(Base + 1) + ", 0\n\t";
    Asm += "sbci r" + std::to_string(Base + 2) + ", 0\n\t";
    Asm += "sbci r" + std::to_string(Base + 3) + ", 0\n\t";
    Asm += "brne " + L + "b\n\t";
    AddClobber(Base);
    AddClobber(Base + 1);
    AddClobber(Base + 2);
    AddClobber(Base + 3);
    ClobberIdx += 4;
    Cycles -= Used;
  }

  // 3-byte loop: 7 + 5*(loop_count-1) cycles
  // ldi×3 + (subi + sbci×2 + brne) per iteration
  if (Cycles >= 262145u) {
    uint32_t LoopCount = ((Cycles - 7) / 5) + 1;
    if (LoopCount > 0xFFFFFFu)
      LoopCount = 0xFFFFFFu;
    uint32_t Used = ((LoopCount - 1) * 5) + 7;
    unsigned Base = 16 + ClobberIdx;
    std::string L = std::to_string(LabelIdx++);
    Asm += "ldi r" + std::to_string(Base) + ", lo8(" +
           std::to_string(LoopCount) + ")\n\t";
    Asm += "ldi r" + std::to_string(Base + 1) + ", hi8(" +
           std::to_string(LoopCount) + ")\n\t";
    Asm += "ldi r" + std::to_string(Base + 2) + ", hlo8(" +
           std::to_string(LoopCount) + ")\n\t";
    Asm += L + ": subi r" + std::to_string(Base) + ", 1\n\t";
    Asm += "sbci r" + std::to_string(Base + 1) + ", 0\n\t";
    Asm += "sbci r" + std::to_string(Base + 2) + ", 0\n\t";
    Asm += "brne " + L + "b\n\t";
    AddClobber(Base);
    AddClobber(Base + 1);
    AddClobber(Base + 2);
    ClobberIdx += 3;
    Cycles -= Used;
  }

  // 2-byte loop: 5 + 4*(loop_count-1) cycles
  // ldi×2 + (sbiw + brne) per iteration
  // sbiw requires an even register in {r24, r26, r28, r30}.
  if (Cycles >= 768u) {
    uint32_t LoopCount = ((Cycles - 5) / 4) + 1;
    if (LoopCount > 0xFFFFu)
      LoopCount = 0xFFFFu;
    uint32_t Used = ((LoopCount - 1) * 4) + 5;
    std::string L = std::to_string(LabelIdx++);
    // Use r24:r25 for sbiw (hardcoded per AVR ISA constraint).
    Asm += "ldi r24, lo8(" + std::to_string(LoopCount) + ")\n\t";
    Asm += "ldi r25, hi8(" + std::to_string(LoopCount) + ")\n\t";
    Asm += L + ": sbiw r24, 1\n\t";
    Asm += "brne " + L + "b\n\t";
    AddClobber(24);
    AddClobber(25);
    Cycles -= Used;
  }

  // 1-byte loop: 3*loop_count cycles
  // ldi + (dec + brne) per iteration
  if (Cycles >= 6u) {
    uint32_t LoopCount = Cycles / 3;
    if (LoopCount > 255u)
      LoopCount = 255u;
    uint32_t Used = LoopCount * 3;
    unsigned Reg = 16 + ClobberIdx;
    if (Reg > 31)
      Reg = 31; // safety
    std::string L = std::to_string(LabelIdx++);
    Asm += "ldi r" + std::to_string(Reg) + ", " + std::to_string(LoopCount) +
           "\n\t";
    Asm += L + ": dec r" + std::to_string(Reg) + "\n\t";
    Asm += "brne " + L + "b\n\t";
    AddClobber(Reg);
    ClobberIdx++;
    Cycles -= Used;
  }

  // Fill remaining with rjmp .+0 (2 cycles each)
  while (Cycles >= 2) {
    Asm += "rjmp .+0\n\t";
    Cycles -= 2;
  }

  // Final single cycle
  if (Cycles == 1) {
    Asm += "nop\n\t";
  }

  if (Asm.empty())
    return nullptr;

  // Remove trailing \n\t
  if (Asm.size() >= 3 && Asm.substr(Asm.size() - 3) == "\n\t")
    Asm.resize(Asm.size() - 3);

  llvm::LLVMContext &Ctx = CGF.getLLVMContext();
  llvm::FunctionType *FTy =
      llvm::FunctionType::get(llvm::Type::getVoidTy(Ctx), false);
  llvm::InlineAsm *IA = llvm::InlineAsm::get(FTy, Asm, Clobbers, true);
  return CGF.Builder.CreateCall(IA);
}

Value *CodeGenFunction::EmitAVRBuiltinExpr(unsigned BuiltinID,
                                           const CallExpr *E) {
  switch (BuiltinID) {
  default:
    return nullptr;
  case AVR::BI__builtin_avr_nop:
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::avr_nop));
  case AVR::BI__builtin_avr_sei:
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::avr_sei));
  case AVR::BI__builtin_avr_cli:
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::avr_cli));
  case AVR::BI__builtin_avr_sleep:
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::avr_sleep));
  case AVR::BI__builtin_avr_wdr:
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::avr_wdr));
  case AVR::BI__builtin_avr_swap: {
    Value *Arg0 = EmitScalarExpr(E->getArg(0));
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::avr_swap), Arg0);
  }
  case AVR::BI__builtin_avr_fmul:
    return EmitAVRFMulInlineAsm(*this, E, "fmul");
  case AVR::BI__builtin_avr_fmuls:
    return EmitAVRFMulInlineAsm(*this, E, "fmuls");
  case AVR::BI__builtin_avr_fmulsu:
    return EmitAVRFMulInlineAsm(*this, E, "fmulsu");

  case AVR::BI__builtin_avr_delay_cycles: {
    // Argument is validated as a compile time constant in Sema.
    Expr::EvalResult Result;
    E->getArg(0)->EvaluateAsInt(Result, getContext());
    uint32_t Cycles = static_cast<uint32_t>(Result.Val.getInt().getZExtValue());
    if (Cycles == 0) {
      // Nothing to emit.
      llvm::FunctionType *FTy = llvm::FunctionType::get(
          llvm::Type::getVoidTy(getLLVMContext()), false);
      return Builder.CreateCall(llvm::InlineAsm::get(FTy, "", "", true));
    }
    return EmitAVRDelayLoops(*this, Cycles);
  }

  case AVR::BI__builtin_avr_nops: {
    // Argument is validated as a compile-time constant in Sema.
    Expr::EvalResult Result;
    E->getArg(0)->EvaluateAsInt(Result, getContext());
    uint32_t N = static_cast<uint32_t>(Result.Val.getInt().getZExtValue());
    if (N == 0) {
      // Nothing to emit.
      llvm::FunctionType *FTy = llvm::FunctionType::get(
          llvm::Type::getVoidTy(getLLVMContext()), false);
      return Builder.CreateCall(llvm::InlineAsm::get(FTy, "", "", true));
    }
    llvm::Function *NopFn = CGM.getIntrinsic(Intrinsic::avr_nop);
    Value *Last = nullptr;
    for (uint32_t I = 0; I < N; ++I)
      Last = Builder.CreateCall(NopFn);
    return Last;
  }
  case AVR::BI__builtin_avr_insert_bits: {
    // Map is a compile-time constant (validated in Sema).
    Expr::EvalResult MapResult;
    E->getArg(0)->EvaluateAsInt(MapResult, getContext());
    uint32_t Map =
        static_cast<uint32_t>(MapResult.Val.getInt().getZExtValue());
    Value *Bits = EmitScalarExpr(E->getArg(1));
    Value *Val = EmitScalarExpr(E->getArg(2));

    llvm::Type *I8Ty = Builder.getInt8Ty();
    Value *Result = llvm::ConstantInt::get(I8Ty, 0);

    for (unsigned I = 0; I < 8; ++I) {
      unsigned Nibble = (Map >> (I * 4)) & 0xF;
      Value *Bit;
      if (Nibble < 8) {
        // Extract bit 'Nibble' from 'Bits' and place it at position 'I'.
        Bit = Builder.CreateAnd(
            Builder.CreateLShr(Bits, llvm::ConstantInt::get(I8Ty, Nibble)),
            llvm::ConstantInt::get(I8Ty, 1));
      } else if (Nibble == 0xF) {
        // Keep bit 'I' from 'Val'.
        Bit = Builder.CreateAnd(
            Builder.CreateLShr(Val, llvm::ConstantInt::get(I8Ty, I)),
            llvm::ConstantInt::get(I8Ty, 1));
      } else {
        // Nibble 8-14: undefined per GCC docs, treat as 0.
        continue;
      }
      Value *Shifted =
          Builder.CreateShl(Bit, llvm::ConstantInt::get(I8Ty, I));
      Result = Builder.CreateOr(Result, Shifted);
    }
    return Result;
  }

  case AVR::BI__builtin_avr_flash_segment: {
    // Extract the address space from the pointer argument's type.
    // __flash = addrspace(1) -> segment 0
    // __flash1 = addrspace(2) -> segment 1
    // ...
    // __flash5 = addrspace(6) -> segment 5
    // Non-flash (addrspace 0 or >6) -> -1
    QualType ArgTy = E->getArg(0)->getType();
    int8_t Segment = -1;
    if (ArgTy->isPointerType()) {
      LangAS AS = ArgTy->getPointeeType().getAddressSpace();
      if (isTargetAddressSpace(AS)) {
        unsigned TargetAS = toTargetAddressSpace(AS);
        if (TargetAS >= 1 && TargetAS <= 6)
          Segment = static_cast<int8_t>(TargetAS - 1);
      }
    }
    return llvm::ConstantInt::get(Builder.getInt8Ty(), Segment,
                                  /*IsSigned=*/true);
  }
  }
}
