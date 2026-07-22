//! # `save_context` / `restore_context`: the unwinder's teleport primitives
//!
//! A `Context` is a full snapshot of AArch64 CPU register state (GPRs,
//! SP, callee-saved FP regs). `save_context` and `restore_context` are
//! the two low-level primitives that let stack unwinding *relocate*
//! execution from a panic site directly into a landing pad, skipping
//! every intermediate frame's normal `ret` chain.
//!
//! ## Example: unwinding through a panic
//!
//! ```text
//! main() -> foo() -> bar()   // bar() panics
//!            ^landing pad
//! ```
//!
//! 1. `bar()` panics. `save_context(unwind_start, ptr)` is called,
//!    capturing the *current* register state into a `Context`
//!    (`Context.gp[30]` == return address back into `foo`).
//!
//! 2. `unwind_start` (the callback `f`) walks the stack using DWARF
//!    CFI rules from `.eh_frame`, starting from that `Context`.
//!    Each step asks: "given this frame's CFI rules and CFA, what
//!    were the registers in the *caller's* frame?" This produces a
//!    new `Context` per frame: `bar` -> `foo` -> `main`, without
//!    ever executing a real `ret`.
//!
//! 3. The walk finds a frame with a `catch_unwind`-style landing pad
//!    in `foo`. The unwinder builds a `Context` describing register
//!    state *at that landing pad* — critically, `gp[30]` now holds
//!    the landing pad's address, not `bar`'s old return address.
//!
//! 4. `restore_context(&landing_pad_context)` force-loads every
//!    register from that `Context` and `ret`s via the freshly loaded
//!    `x30` — jumping straight into `foo`'s landing pad. `bar` and
//!    its frame are never returned through; they're simply discarded.
//!
use core::fmt;
use core::ops;
use gimli::{AArch64, Register};

use super::maybe_cfi;

// Match DWARF_FRAME_REGISTERS in libgcc
// The maximum number of rules preallocated by libunwind is 97 for AArch64.
pub const MAX_REG_RULES: usize = 97;

#[repr(C)]
#[derive(Clone, Default)]
pub struct Context {
    pub gp: [usize; 31], // 31 * 8 = 248 bytes
    pub sp: usize,       // 1 * 8 = 1 bytes
    pub fp: [usize; 32], // 32 * 8 = 256 bytes
}

impl fmt::Debug for Context {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut fmt = fmt.debug_struct("Context");
        for i in 0..=30 {
            fmt.field(
                AArch64::register_name(Register(i as _)).unwrap(),
                &self.gp[i],
            );
        }
        fmt.field("sp", &self.sp);
        for i in 0..=31 {
            fmt.field(
                AArch64::register_name(Register((i + 64) as _)).unwrap(),
                &self.fp[i],
            );
        }
        fmt.finish()
    }
}

impl ops::Index<Register> for Context {
    type Output = usize;

    fn index(&self, reg: Register) -> &usize {
        match reg {
            Register(0..=30) => &self.gp[reg.0 as usize],
            AArch64::SP => &self.sp,
            Register(64..=95) => &self.fp[(reg.0 - 64) as usize],
            _ => unimplemented!(),
        }
    }
}

impl ops::IndexMut<gimli::Register> for Context {
    fn index_mut(&mut self, reg: Register) -> &mut usize {
        match reg {
            Register(0..=30) => &mut self.gp[reg.0 as usize],
            AArch64::SP => &mut self.sp,
            Register(64..=95) => &mut self.fp[(reg.0 - 64) as usize],
            _ => unimplemented!(),
        }
    }
}

macro_rules! save {
    (gp$(, $fp:ident)?) => {
        // No need to save caller-saved registers here.
        // `naked_asm`  writes the raw, complete instruction sequence that becomes the entire body of a #[naked] function,
        // with no compiler-added code before or after it.
        core::arch::naked_asm!(
            // Indicates that the CFI directives that follow describe how to unwind this function,
            // also marks CFA = SP + 0, which will be used relatively for CFA offsets,
            // aka CFA is a fixed address and it never moves during the function.
            maybe_cfi!(".cfi_startproc"),
            // Decrement SP, make 16 bytes of space for storing x29(FP), x30(LR).
            "stp x29, x30, [sp, -16]!",
            // Offset of Current SP from CFA is now current_sp+16
            // x29 is stored at CFA-16.
            // x30 (link register) is saved at CFA - 8.
            maybe_cfi!("
            .cfi_def_cfa_offset 16
            .cfi_offset x29, -16
            .cfi_offset x30, -8
            "),
            // Decrement SP, make room for 512 bytes of space for storing the Context Struct fields.
            "sub sp, sp, 512",
            // Offset of Current SP from CFA is now current_sp+512+16.
            maybe_cfi!(".cfi_def_cfa_offset 528"),
            // Store Argument x0 to scratch register x8.
            // Store SP to x0.
            "
            mov x8, x0
            mov x0, sp
            ",
            // x0 above now points to Context Struct.
            // Save FP/ lower SIMD registers, if fp was caputred in macro invocation.
            save!(maybesavefp($($fp)?)),
            // Save x19 to x30(only the callee-saved registers) at offset 152 to 240 of the Context Struct,
            // also stores the reconstructed orignal SP at the Context.sp field location.
            // Branch to the callback fn address stored in x8 with link register set to next instruction address,
            // and since x0 contains pointer to Context Struct, and x1 is untouched,
            // the callback fn arguments are satisfied.
            // Reduce Stack size by 512 bytes after return.
            "
            str x19, [sp, 0x98]
            stp x20, x21, [sp, 0xA0]
            stp x22, x23, [sp, 0xB0]
            stp x24, x25, [sp, 0xC0]
            stp x26, x27, [sp, 0xD0]
            stp x28, x29, [sp, 0xE0]
            add x2, sp, 528
            stp x30, x2, [sp, 0xF0]

            blr x8

            add sp, sp, 512
            ",
            // Offset of Current SP from CFA is now current_sp+16.
            maybe_cfi!(".cfi_def_cfa_offset 16"),
            // Load the FP and LR from stack to its registers.
            "ldp x29, x30, [sp], 16",
            // Offset of Current SP from CFA is now current_sp+16.
            // Indicate that x29 and x30 can be read directly from the registers,
            // no need to follow the earlier offset from CFA.
            maybe_cfi!("
            .cfi_def_cfa_offset 0
            .cfi_restore x29
            .cfi_restore x30
            "),
            // Return back to the caller, aka whoever called save_context.
            "ret",
            // Unwind info description ends here.
            maybe_cfi!(".cfi_endproc"),
        );
    };
    // Save dSized callee-saved registers from offset 320 to 368 of the Context Struct field.
    (maybesavefp(fp)) => {
        "
        stp d8, d9, [sp, 0x140]
        stp d10, d11, [sp, 0x150]
        stp d12, d13, [sp, 0x160]
        stp d14, d15, [sp, 0x170]
        "
    };
    (maybesavefp()) => { "" };
}

// `naked`: Compiler should emit zero generated code,
// aka no prologue, epilogue, register spills, or stack frame for this function.
// The function body must supply those actual instructions itself as frame layout matters for unwinding.
#[unsafe(naked)]
// extern "C": use the standard C calling convention; on panic, unwinding across this function boundary is UB.
// extern "C-unwind": same C calling convention, but permits a Rust/foreign unwind (panic) to propagate through this function boundary safely.
pub extern "C-unwind" fn save_context(f: extern "C" fn(&mut Context, *mut ()), ptr: *mut ()) {
    // save float/simd registers only if the architecture supports it.
    #[cfg(target_feature = "neon")]
    save!(gp, fp);
    #[cfg(not(target_feature = "neon"))]
    save!(gp);
}

macro_rules! restore {
    ($ctx:expr, gp$(, $fp:ident)?) => {
        core::arch::asm!(
            restore!(mayberestore($($fp)?)),
            // Load register values x0 to x30 from Context gp fields and sp from Context sp field.
            // Return at address stored in x30.
            "
            ldp x2, x3, [x0, 0x10]
            ldp x4, x5, [x0, 0x20]
            ldp x6, x7, [x0, 0x30]
            ldp x8, x9, [x0, 0x40]
            ldp x10, x11, [x0, 0x50]
            ldp x12, x13, [x0, 0x60]
            ldp x14, x15, [x0, 0x70]
            ldp x16, x17, [x0, 0x80]
            ldp x18, x19, [x0, 0x90]
            ldp x20, x21, [x0, 0xA0]
            ldp x22, x23, [x0, 0xB0]
            ldp x24, x25, [x0, 0xC0]
            ldp x26, x27, [x0, 0xD0]
            ldp x28, x29, [x0, 0xE0]
            ldp x30, x1, [x0, 0xF0]
            mov sp, x1

            ldp x0, x1, [x0, 0x00]
            ret
            ",
            // Place the value of Rust variable `ctx` into register x0 before this block runs,
            // aka x0 stores address, which points to the Context Struct.
            in("x0") $ctx,
            // Indicate compiler that this block diverges(-> !),
            // no code generation such as epilogue should be done as this block calls ret itself,
            // thus no need for restore_context fn to set epilogue as it will be deadcode.
            options(noreturn)
        );
    };
    (mayberestore(fp)) => {
        // Load d0 to d31 register from Context fp fields.
        "
        ldp d0, d1, [x0, 0x100]
        ldp d2, d3, [x0, 0x110]
        ldp d4, d5, [x0, 0x120]
        ldp d6, d7, [x0, 0x130]
        ldp d8, d9, [x0, 0x140]
        ldp d10, d11, [x0, 0x150]
        ldp d12, d13, [x0, 0x160]
        ldp d14, d15, [x0, 0x170]
        ldp d16, d17, [x0, 0x180]
        ldp d18, d19, [x0, 0x190]
        ldp d20, d21, [x0, 0x1A0]
        ldp d22, d23, [x0, 0x1B0]
        ldp d24, d25, [x0, 0x1C0]
        ldp d26, d27, [x0, 0x1D0]
        ldp d28, d29, [x0, 0x1E0]
        ldp d30, d31, [x0, 0x1F0]
        "
    };
    (mayberestore()) => { "" };
}

pub unsafe fn restore_context(ctx: &Context) -> ! {
    unsafe {
        // restore float/simd registers only if the architecture supports it.
        #[cfg(target_feature = "neon")]
        restore!(ctx, gp, fp);
        #[cfg(not(target_feature = "neon"))]
        restore!(ctx, gp);
    }
}
