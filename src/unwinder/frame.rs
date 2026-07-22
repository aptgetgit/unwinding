use gimli::{
    BaseAddresses, CfaRule, Register, RegisterRule, UnwindContext, UnwindExpression, UnwindTableRow,
};
#[cfg(feature = "dwarf-expr")]
use gimli::{Evaluation, EvaluationResult, Location, Value};

use super::arch::*;
use super::find_fde::{self, FDEFinder, FDESearchResult};
use crate::abi::PersonalityRoutine;
use crate::arch::*;
use crate::util::*;

struct StoreOnStack;

// gimli's MSRV doesn't allow const generics, so we need to pick a supported array size.
const fn next_value(x: usize) -> usize {
    let supported = [0, 1, 2, 3, 4, 8, 16, 32, 64, 128];
    let mut i = 0;
    while i < supported.len() {
        if supported[i] >= x {
            return supported[i];
        }
        i += 1;
    }
    192
}

impl<O: gimli::ReaderOffset> gimli::UnwindContextStorage<O> for StoreOnStack {
    // Array of Register, Register-Rule Tuple,
    // where Register Rule: An entry in the abstract CFI table that describes how to find the value of a register.
    type Rules = [(Register, RegisterRule<O>); next_value(MAX_REG_RULES)];
    // Virtual Stack that stores at most 2 Unwind Table Rows.
    // where Unwind Table Row: A row in the virtual unwind table that describes for a given range of PC addresses, how to find the values of
    // the registers in the *previous/callers* frame. It does so by using CFA Rule and Register Rules. aka Per PC range → (CFA rule, register rules for each saved register).
    // where CFA Rule: The canonical frame address (CFA) recovery rules.
    type Stack = [UnwindTableRow<O, Self>; 2];
}

#[cfg(feature = "dwarf-expr")]
impl<R: gimli::Reader> gimli::EvaluationStorage<R> for StoreOnStack {
    type Stack = [Value; 64];
    type ExpressionStack = [(R, R); 0];
    type Result = [gimli::Piece<R>; 1];
}

#[derive(Debug)]
pub struct Frame {
    fde_result: FDESearchResult,
    row: UnwindTableRow<usize, StoreOnStack>,
}

impl Frame {
    // Given the current frame's context, finds the FDE covering its PC (via
    // ctx[Arch::RA]) and computes the UnwindTableRow for that PC, i.e. the
    // *formula* (CFA rule + register rules) for recovering the caller's
    // register values. Evaluating that formula against a context is `unwind`'s job.
    pub fn from_context(ctx: &Context, signal: bool) -> Result<Option<Self>, gimli::Error> {
        let mut ra = ctx[Arch::RA];

        // Reached end of stack
        if ra == 0 {
            return Ok(None);
        }

        // RA points to the address of the *next* instruction after the call,
        // so subtract 1 to guarantee we land inside the byte range of the call
        // instruction itself (not at the start of whatever follows it). This
        // matters because FDEs cover PC ranges: if `ra` sits exactly on a range
        // boundary, using it unadjusted could select the wrong FDE/row.
        //
        // Signal == true: `ra` is the exact PC where execution was paused by a
        // signal trampoline (DW_CFA_signal_frame), the trampoline records the
        // precise faulting/paused PC itself, not a return address, so there is
        // no "next instruction" to back up from. Use `ra` as-is for the CFA lookup.
        //
        // Signal == false indicates `ra` is a normal call's return address
        // (points past the call), so subtract 1 to fall back inside the call
        // instruction that caused this frame to exist.
        if !signal {
            ra -= 1;
        }

        // Get the frame description entry for the given `ra`, base addresses and the parsed FDE's CFI program.
        let fde_result = match find_fde::get_finder().find_fde(ra as _) {
            Some(v) => v,
            None => return Ok(None),
        };
        // Compute exactly one UnwindTableRow for the given ra:
        // the CFA rule + saved-register rules needed to reconstruct one previous frame's register state.
        // aka data needed to compute previous frame's data.
        let mut unwinder = UnwindContext::<_, StoreOnStack>::new_in();
        let row = fde_result
            .fde
            .unwind_info_for_address(
                &fde_result.eh_frame,
                &fde_result.bases,
                &mut unwinder,
                ra as _,
            )?
            .clone();

        Ok(Some(Self { fde_result, row }))
    }

    #[cfg(feature = "dwarf-expr")]
    fn evaluate_expression(
        &self,
        ctx: &Context,
        expr: UnwindExpression<usize>,
    ) -> Result<usize, gimli::Error> {
        let expr = expr.get(&self.fde_result.eh_frame).unwrap();
        let mut eval =
            Evaluation::<_, StoreOnStack>::new_in(expr.0, self.fde_result.fde.cie().encoding());
        let mut result = eval.evaluate()?;
        loop {
            match result {
                EvaluationResult::Complete => break,
                EvaluationResult::RequiresMemory { address, .. } => {
                    let value = unsafe { (address as usize as *const usize).read_unaligned() };
                    result = eval.resume_with_memory(Value::Generic(value as _))?;
                }
                EvaluationResult::RequiresRegister { register, .. } => {
                    let value = ctx[register];
                    result = eval.resume_with_register(Value::Generic(value as _))?;
                }
                EvaluationResult::RequiresRelocatedAddress(address) => {
                    let value = unsafe { (address as usize as *const usize).read_unaligned() };
                    result = eval.resume_with_memory(Value::Generic(value as _))?;
                }
                _ => unreachable!(),
            }
        }

        Ok(
            match eval
                .as_result()
                .last()
                .ok_or(gimli::Error::PopWithEmptyStack)?
                .location
            {
                Location::Address { address } => address as usize,
                _ => unreachable!(),
            },
        )
    }

    // DWARF CFI supports two kinds of rules: simple ones (`register + constant
    // offset` for CFA, and "saved at register+offset" for callee-saved regs),
    // and full `DW_CFA_expression` / `DW_CFA_val_expression` rules, which embed
    // a small DWARF bytecode program to be evaluated to compute the value.
    //
    // Rustc/LLVM's codegen for Rust's own frames only ever emits the simple
    // form — Rust doesn't use variable-length stack allocations (`alloca`),
    // dynamic realignment, or segmented/split stacks that would require a
    // computed rule, so a full DWARF expression evaluator is dead weight
    // for pure-Rust unwinding. Other languages/toolchains (C with dynamic
    // `alloca`, some C++ codegen under frame-pointer-omission + dynamic
    // stack realignment, or historically Go's segmented stacks) *do* need
    // it, since their CFA/register locations aren't expressible as a fixed
    // offset.
    //
    // NOTE: the linker concatenates .eh_frame entries from every linked
    // object as-is, it does not rewrite them. So statically linking or FFI
    // with C/C++ (or any) .o files whose CFI uses DW_CFA_expression/
    // DW_CFA_val_expression (e.g. from alloca, dynamic stack realignment,
    // or frame-pointer-omission) means unwinding will fail with
    // `UnsupportedEvaluation` the moment it hits one of *those* frames,
    // even though pure-Rust frames unwind fine. Enable the "dwarf-expr"
    // feature if such FFI/linked code is in the unwind path.
    #[cfg(not(feature = "dwarf-expr"))]
    fn evaluate_expression(
        &self,
        _ctx: &Context,
        _expr: UnwindExpression<usize>,
    ) -> Result<usize, gimli::Error> {
        Err(gimli::Error::UnsupportedEvaluation)
    }

    // Frees leftover outgoing-arg stack space (`saved_args_size`, from the CFI
    // row) by moving SP back up in Context struct which is in memory.
    // The Installer code which is `restore_context` runs next,
    // where the cpu registers are replaced by the Context struct values.
    //
    // Needed because we're about to jump straight
    // into this frame's landing pad instead of letting the call it made
    // return normally(epilogue doesn't run for the panicking fn),
    // so the usual step of popping its outgoing-arg bytes never happens,
    // and this does it instead.
    pub fn adjust_stack_for_args(&self, ctx: &mut Context) {
        let size = self.row.saved_args_size();
        ctx[Arch::SP] = ctx[Arch::SP].wrapping_add(size as usize);
    }

    // Evaluates this frame's row (CFA rule + register rules) against `ctx`,
    // resolving the formula into actual values, to produce the caller's
    // (parent's) real register state as a new Context.
    pub fn unwind(&self, ctx: &Context) -> Result<Context, gimli::Error> {
        let row = &self.row;
        let mut new_ctx = ctx.clone();

        // CFA is computed here from the *callee's* row/rules and ctx, the
        // resulting value is the caller's SP — see definition below.
        let cfa = match *row.cfa() {
            CfaRule::RegisterAndOffset { register, offset } => {
                ctx[register].wrapping_add(offset as usize)
            }
            CfaRule::Expression(expr) => self.evaluate_expression(ctx, expr)?,
        };

        // CFA is defined, by the DWARF CFI spec itself, as:
        // "the value of the stack pointer at the call site in the previous frame",
        // i.e., CFA = the caller's SP, at the exact moment the caller executed the
        // call/bl instruction into the callee. Thus the CFA here is the Stack
        // Pointer for the Caller.
        new_ctx[Arch::SP] = cfa as _;
        new_ctx[Arch::RA] = 0; // stays 0 (= "end of stack") unless overwritten below by an explicit RA rule.

        for (reg, rule) in row.registers() {
            let value = match *rule {
                // For most registers, `Undefined` indicates the value does not need to
                // be preserved so the value content does not matter. However when RA is
                // `Undefined` it indicates that the unwinding is complete.
                RegisterRule::Undefined => 0,
                RegisterRule::SameValue => ctx[*reg],
                RegisterRule::Offset(offset) => unsafe {
                    *((cfa.wrapping_add(offset as usize)) as *const usize)
                },
                RegisterRule::ValOffset(offset) => cfa.wrapping_add(offset as usize),
                RegisterRule::Register(r) => ctx[r],
                RegisterRule::Expression(expr) => {
                    let addr = self.evaluate_expression(ctx, expr)?;
                    unsafe { *(addr as *const usize) }
                }
                RegisterRule::ValExpression(expr) => self.evaluate_expression(ctx, expr)?,
                RegisterRule::Architectural => unreachable!(),
                RegisterRule::Constant(value) => value as usize,
            };
            new_ctx[*reg] = value;
        }

        Ok(new_ctx)
    }

    // Return the base addresses for the FDE.
    pub fn bases(&self) -> &BaseAddresses {
        &self.fde_result.bases
    }

    // --- Exception-handling metadata for this frame's CIE ---
    //
    // `personality` and `lsda` work together during an unwinding panic:
    // `personality` is *who* decides what to do at this frame, and `lsda` is
    // *the data* they use to decide it.

    // Resolves this frame's personality routine, in Rust, `rust_eh_personality`.
    // The unwind runtime calls this function at each frame during a panic to
    // decide whether the frame has Drop cleanup code to run, or whether to keep
    // propagating further up the stack.
    //
    // The lookup is per-CIE (not global) because `.eh_frame` is a generic
    // format: a single binary can contain multiple CIEs, and different CIEs
    // can reference different personality routines — e.g. in a mixed Rust+C++
    // binary, C++ CIEs point to `__gxx_personality_v0` while Rust CIEs point to
    // `rust_eh_personality`. In a pure-Rust binary there can still be many
    // CIEs, but every one with a personality at all resolves to this same
    // `rust_eh_personality` address.
    pub fn personality(&self) -> Option<PersonalityRoutine> {
        self.fde_result
            .fde
            .personality()
            .map(|x| unsafe { deref_pointer(x) })
            .map(|x| unsafe { core::mem::transmute(x) })
    }

    // Resolves this frame's LSDA (`.gcc_except_table` entry), the table the
    // personality routine reads to answer, at a given PC, whether there's a
    // Drop-cleanup landing pad here or the unwind should keep going up. Unlike
    // `personality`, this is genuinely per-frame: different functions have
    // different landing-pad layouts. Returns 0 if the frame has no LSDA at all
    // (e.g. a leaf function with nothing to clean up).
    pub fn lsda(&self) -> usize {
        self.fde_result
            .fde
            .lsda()
            .map(|x| unsafe { deref_pointer(x) })
            .unwrap_or(0)
    }

    // Start address of the PC range this FDE covers (i.e. the first instruction
    // of the function/range this frame belongs to).
    pub fn initial_address(&self) -> usize {
        self.fde_result.fde.initial_address() as _
    }

    // Marks a synthetic signal-handler-entry frame, i.e. a frame created not
    // by a normal `call` but by the kernel forcibly redirecting execution to a
    // signal trampoline stub (e.g. Linux's `__restore_rt`) after delivering a
    // signal like SIGSEGV. This is unrelated to page-table-switch trampolines
    // (like xv6's uservec/userret): no address-space change happens here, it's
    // purely "PC was hijacked mid-execution, with synthetic state pushed onto
    // the same user stack." Rust's own unwinder rarely hits this itself (Rust
    // doesn't unwind across signal boundaries by default), but it matters when
    // unwinding through mixed Rust/C code that does use signal-based control flow.
    pub fn is_signal_trampoline(&self) -> bool {
        self.fde_result.fde.is_signal_trampoline()
    }
}
