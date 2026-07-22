mod arch;
mod find_fde;
mod frame;

use core::ffi::c_void;
use core::ptr;
use gimli::Register;

use crate::abi::*;
use crate::arch::*;
use crate::util::*;
use arch::*;
use find_fde::FDEFinder;
use frame::Frame;

#[cfg(feature = "fde-custom")]
pub use find_fde::custom_eh_frame_finder;

/// Lets us pass a generic `FnOnce(&mut Context) -> T` closure into
/// `save_context`, even though `save_context` is `#[naked]` and can't
/// be generic.
///
/// A naked function's asm body is written out once, by hand, at a fixed
/// stack layout (fixed offsets, fixed size like `sub sp, sp, 512`).
/// Normal generic functions get a fresh copy of their body compiled per
/// concrete type used (per call site), but naked asm can't do that,
/// there's only one hand-written body, shared by every caller. So
/// `save_context` can't know in advance how much stack space some
/// arbitrary closure's captured variables would need, or how to call it.
/// It can only work with one fixed function pointer signature
/// (`extern "C" fn(&mut Context, *mut ())`) and one raw data pointer.
fn with_context<T, F: FnOnce(&mut Context) -> T>(f: F) -> T {
    use core::mem::ManuallyDrop;

    /// One slot that holds either `F` (before the call) or `T` (after).
    /// Never holds both at once, so we don't need space for both, and
    /// we don't need to make up a fake starting value for `T` (which we
    /// can't do anyway, since `T` is generic).
    ///
    /// `ManuallyDrop` on both fields stops Rust from auto-dropping them.
    /// A union doesn't know which field is "the real one" right now, so
    /// we have to manage that ourselves.
    union Data<T, F> {
        f: ManuallyDrop<F>,
        t: ManuallyDrop<T>,
    }

    /// Fixed-signature function that `save_context`'s asm can actually call.
    /// A new copy of this gets generated for each `(T, F)` combo (normal
    /// Rust generics), but every copy has the same signature, so the asm
    /// doesn't need to care which one it's calling.
    ///
    /// Called from asm as `delegate(ctx, ptr)`:
    /// - `ctx`: `&mut Context`, built by `save_context` from the registers
    ///   it just saved.
    /// - `ptr`: pointer to the `Data<T, F>` sitting in `with_context`'s
    ///   stack frame.
    extern "C" fn delegate<T, F: FnOnce(&mut Context) -> T>(ctx: &mut Context, ptr: *mut ()) {
        // SAFETY: this only runs once, and `ptr` always points to a real
        // `Data<T, F>` whose `f` field is the one currently set.
        // This fn is plain `extern "C"` (not `-unwind`): if the closure panics,
        // Rust cannot legally unwind across this boundary (that's UB), so it
        // aborts the process instead. That's what we want here, it stops an
        // unwind from ever seeing `data` in its half-taken state (f moved out,
        // t not yet written), and stops an unwind from trying to pass through
        // `save_context`'s naked asm, which has no unwind info to guide it.
        unsafe {
            let data = &mut *ptr.cast::<Data<T, F>>();
            // Pull the closure out of the union and call it with ctx.
            let t = ManuallyDrop::take(&mut data.f)(ctx);
            // Put the result back in the same slot, now as `t`.
            data.t = ManuallyDrop::new(t);
        }
    }

    // Start with the slot holding the closure.
    let mut data = Data {
        f: ManuallyDrop::new(f),
    };

    // Give the naked function:
    //   arg0 = delegate::<T, F> (a plain function pointer)
    //   arg1 = pointer to data
    // Its asm saves registers into a Context, then jumps to `delegate`
    // with (&Context, &mut data).
    save_context(delegate::<T, F>, ptr::addr_of_mut!(data).cast());

    // By now `delegate` has already filled `data` with the result.
    // SAFETY: `data.t` was just set by `delegate` before `save_context` returned.
    unsafe { ManuallyDrop::into_inner(data.t) }
}

#[repr(C)] // stable layout required: crosses the language-agnostic unwind ABI boundary
pub struct UnwindException {
    pub exception_class: u64, // tags which language runtime owns this exception
    pub exception_cleanup: Option<UnwindExceptionCleanupFn>, // called once handling is done, to free the exception
    private_1: Option<UnwindStopFn>, // Some(stop_fn) if forced unwind, None if normal two-phase unwind
    private_2: usize,                // stop_arg (forced unwind) or handler_cfa (normal unwind)
    private_unused: [usize; Arch::UNWIND_PRIVATE_DATA_SIZE - 2], // padding to match ABI's reserved private size
}

pub struct UnwindContext<'a> {
    frame: Option<&'a Frame>, // current frame's unwind info (FDE, LSDA, personality); None if no frame (end of stack)
    ctx: &'a mut Context, // mutable borrow of saved registers; personality routines read/write via this
    signal: bool, // true if this frame is a signal trampoline (changes CFA/PC offset handling)
}

// read general register index from the current frame's saved context.
#[unsafe(no_mangle)]
pub extern "C" fn _Unwind_GetGR(unwind_ctx: &UnwindContext<'_>, index: c_int) -> usize {
    unwind_ctx.ctx[Register(index as u16)]
}

// get the Canonical Frame Address (current frame's stack pointer).
#[unsafe(no_mangle)]
pub extern "C" fn _Unwind_GetCFA(unwind_ctx: &UnwindContext<'_>) -> usize {
    unwind_ctx.ctx[Arch::SP]
}

// write value into general register index in the saved context (used to set up a landing pad).
#[unsafe(no_mangle)]
pub extern "C" fn _Unwind_SetGR(unwind_ctx: &mut UnwindContext<'_>, index: c_int, value: usize) {
    unwind_ctx.ctx[Register(index as u16)] = value;
}

// get the current instruction pointer (return address register).
#[unsafe(no_mangle)]
pub extern "C" fn _Unwind_GetIP(unwind_ctx: &UnwindContext<'_>) -> usize {
    unwind_ctx.ctx[Arch::RA]
}

// get the instruction pointer, and also tell the caller (via ip_before_insn)
// whether this is a signal frame (IP is exact) or a normal
// frame (IP is a return address, meaning the actual call site is one instruction before it).
#[unsafe(no_mangle)]
pub extern "C" fn _Unwind_GetIPInfo(
    unwind_ctx: &UnwindContext<'_>,
    ip_before_insn: &mut c_int,
) -> usize {
    *ip_before_insn = unwind_ctx.signal as _;
    unwind_ctx.ctx[Arch::RA]
}

//  set the instruction pointer (return address register) to value (jump target for the landing pad).
#[unsafe(no_mangle)]
pub extern "C" fn _Unwind_SetIP(unwind_ctx: &mut UnwindContext<'_>, value: usize) {
    unwind_ctx.ctx[Arch::RA] = value;
}

// get the pointer to this frame's LSDA (the table its personality routine uses to decide catch/cleanup),
// or null if there's no frame.
#[unsafe(no_mangle)]
pub extern "C" fn _Unwind_GetLanguageSpecificData(unwind_ctx: &UnwindContext<'_>) -> *mut c_void {
    unwind_ctx
        .frame
        .map(|f| f.lsda() as *mut c_void)
        .unwrap_or(ptr::null_mut())
}

// get the start address of the current function/frame, or 0 if there's no frame.
#[unsafe(no_mangle)]
pub extern "C" fn _Unwind_GetRegionStart(unwind_ctx: &UnwindContext<'_>) -> usize {
    unwind_ctx.frame.map(|f| f.initial_address()).unwrap_or(0)
}

// get the base address used for text-relative encoded pointers in .eh_frame, or 0 if unavailable.
#[unsafe(no_mangle)]
pub extern "C" fn _Unwind_GetTextRelBase(unwind_ctx: &UnwindContext<'_>) -> usize {
    unwind_ctx
        .frame
        .map(|f| f.bases().eh_frame.text.unwrap() as _)
        .unwrap_or(0)
}

// get the base address used for data-relative encoded pointers in .eh_frame, or 0 if unavailable.
#[unsafe(no_mangle)]
pub extern "C" fn _Unwind_GetDataRelBase(unwind_ctx: &UnwindContext<'_>) -> usize {
    unwind_ctx
        .frame
        .map(|f| f.bases().eh_frame.data.unwrap() as _)
        .unwrap_or(0)
}

// given any PC, find and return the start address of the function containing it, by looking up its FDE.
#[unsafe(no_mangle)]
pub extern "C" fn _Unwind_FindEnclosingFunction(pc: *mut c_void) -> *mut c_void {
    find_fde::get_finder()
        .find_fde(pc as usize - 1)
        .map(|r| r.fde.initial_address() as usize as _)
        .unwrap_or(ptr::null_mut())
}

macro_rules! try1 {
    ($e: expr) => {{
        match $e {
            Ok(v) => v,
            Err(_) => return UnwindReasonCode::FATAL_PHASE1_ERROR,
        }
    }};
}

macro_rules! try2 {
    ($e: expr) => {{
        match $e {
            Ok(v) => v,
            Err(_) => return UnwindReasonCode::FATAL_PHASE2_ERROR,
        }
    }};
}

// Phase 1 walks a clone of the context looking only for a catch handler, ignoring destructors (Drop frames return CONTINUE_UNWIND);
// on finding one it records handler_cfa and stops, the stack is still fully intact,
// so if no handler exists you can terminate cleanly(abort/terminate process) and OS level post-moterm can be done.
//
// Phase 2 re-walks from the top on the real context, calling each personality
// with CLEANUP_PHASE (plus HANDLER_FRAME on the frame whose CFA matches
// handler_cfa). Frames with no work return CONTINUE_UNWIND and are skipped;
// the first with a landing pad returns INSTALL_CONTEXT and we jump into it via
// restore_context.
//
// Both kinds of pad are entered the same way (INSTALL_CONTEXT → restore_context); only the exit differs aka
// a Drop pad runs its destructor and calls _Unwind_Resume to re-enter the unwinder and continue to the next pad,
// while the catch pad (the HANDLER_FRAME, whose CFA matches) runs its handler, does not resume, and falls through to the code after catch_unwind,
// which is what ends the whole cycle.

#[inline(never)]
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn _Unwind_RaiseException(
    exception: *mut UnwindException,
) -> UnwindReasonCode {
    with_context(|saved_ctx| {
        // Phase 1: Search for handler
        let mut ctx = saved_ctx.clone();
        // _Unwind_RaiseException is invoked via a normal function call (from Rust panic machinery),
        // not via signal delivery, so the frame it starts in is, by definition, a regular call frame,
        // and signal = false is simply the accurate initial value.
        let mut signal = false;
        loop {
            // Find frame(FDE and Unwind Table Row) for the give context.
            if let Some(frame) = try1!(Frame::from_context(&ctx, signal)) {
                // Get the personaloty routine.
                if let Some(personality) = frame.personality() {
                    // Invoke the personality routine with current exception data.
                    let result = unsafe {
                        personality(
                            1,                          // Version 1
                            UnwindAction::SEARCH_PHASE, // Search Phase
                            (*exception).exception_class,
                            exception,
                            &mut UnwindContext {
                                frame: Some(&frame), // Frame found for the context.
                                ctx: &mut ctx,       // Context
                                signal,              // Frame is signal or not
                            },
                        )
                    };

                    match result {
                        // on result being CONTINUE_UNWIND, do nothing and let the loop continue.
                        UnwindReasonCode::CONTINUE_UNWIND => (),
                        // on handler found, break the loop.
                        UnwindReasonCode::HANDLER_FOUND => {
                            break;
                        }
                        // anything else, return as erorr in phase 1.
                        _ => return UnwindReasonCode::FATAL_PHASE1_ERROR,
                    }
                }

                // Compute the caller's register state, move up one frame.
                ctx = try1!(frame.unwind(&ctx));
                signal = frame.is_signal_trampoline();
            } else {
                // If no frame was found, return as end of stack.
                return UnwindReasonCode::END_OF_STACK;
            }
        }

        // Disambiguate normal frame and signal frame.
        let handler_cfa = ctx[Arch::SP] - signal as usize;
        unsafe {
            (*exception).private_1 = None;
            (*exception).private_2 = handler_cfa;
        }

        // Run Cleanup and isntall context.
        let code = raise_exception_phase2(exception, saved_ctx, handler_cfa);
        match code {
            UnwindReasonCode::INSTALL_CONTEXT => unsafe { restore_context(saved_ctx) },
            _ => code,
        }
    })
}

fn raise_exception_phase2(
    exception: *mut UnwindException,
    ctx: &mut Context,
    handler_cfa: usize,
) -> UnwindReasonCode {
    // _Unwind_RaiseException is invoked via a normal function call (from Rust panic machinery),
    // not via signal delivery, so the frame it starts in is, by definition, a regular call frame,
    // and signal = false is simply the accurate initial value.
    let mut signal = false;
    loop {
        // Same run as Phase 1, but it runs clearnup action in every frame found aka destructors, until the handler CFA is matched.
        if let Some(frame) = try2!(Frame::from_context(ctx, signal)) {
            // Get the Frame.
            let frame_cfa = ctx[Arch::SP] - signal as usize;
            // Get the Personaloty Routine.
            if let Some(personality) = frame.personality() {
                // Invoke personality routine with data, for cleanup.
                let code = unsafe {
                    personality(
                        1,
                        // BitOr opertaion, if the CFA matches that of the handler frame we observed in Phase 1, and the routine must setup the Landing Pad.
                        UnwindAction::CLEANUP_PHASE
                            | if frame_cfa == handler_cfa {
                                UnwindAction::HANDLER_FRAME
                            } else {
                                UnwindAction::empty()
                            },
                        (*exception).exception_class,
                        exception,
                        &mut UnwindContext {
                            frame: Some(&frame),
                            ctx,
                            signal,
                        },
                    )
                };

                match code {
                    // Continue Loop if nothing found.
                    UnwindReasonCode::CONTINUE_UNWIND => (),
                    UnwindReasonCode::INSTALL_CONTEXT => {
                        // We're jumping straight into this frame's landing pad instead of
                        // letting the call we unwound through actually return, so its
                        // outgoing-arg stack space never gets popped normally, free it
                        // now, then this frame's landing pad gets installed next
                        // where landing pad is for example: catch_unwind code,
                        // or a Drop Impl that needs to run before resuming unwind.
                        frame.adjust_stack_for_args(ctx);
                        return UnwindReasonCode::INSTALL_CONTEXT;
                    }
                    _ => return UnwindReasonCode::FATAL_PHASE2_ERROR,
                }
            }

            // Compute the caller's register state, move up one frame.
            *ctx = try2!(frame.unwind(ctx));
            signal = frame.is_signal_trampoline();
        } else {
            return UnwindReasonCode::FATAL_PHASE2_ERROR;
        }
    }
}

/// No handler search: nothing catches during a forced unwind, so there is no
/// handler to find and no stack to preserve, just one cleanup pass. `stop`
/// and `stop_arg` are saved in the exception's private data so a cleanup pad
/// can resume via `_Unwind_Resume`. `force_unwind_phase2` then walks each
/// frame, calling `stop` (which decides when to halt, usually by longjmp) and
/// running its destructors; on `INSTALL_CONTEXT` we jump into the landing pad,
/// otherwise the reason code is returned.
#[inline(never)]
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn _Unwind_ForcedUnwind(
    exception: *mut UnwindException,
    stop: UnwindStopFn,
    stop_arg: *mut c_void,
) -> UnwindReasonCode {
    with_context(|ctx| {
        unsafe {
            // Install the Stop Fn
            (*exception).private_1 = Some(stop);
            // Install the Stop Argument
            (*exception).private_2 = stop_arg as _;
        }

        let code = force_unwind_phase2(exception, ctx, stop, stop_arg);
        match code {
            UnwindReasonCode::INSTALL_CONTEXT => unsafe { restore_context(ctx) },
            _ => code,
        }
    })
}

fn force_unwind_phase2(
    exception: *mut UnwindException,
    ctx: &mut Context,
    stop: UnwindStopFn,
    stop_arg: *mut c_void,
) -> UnwindReasonCode {
    // _Unwind_ForcedUnwind is invoked via a normal function call, not via
    // signal delivery, so the frame it starts in is, by definition, a
    // regular call frame, and signal = false is simply the accurate
    // initial value.
    let mut signal = false;
    loop {
        let frame = try2!(Frame::from_context(ctx, signal));

        let code = unsafe {
            // Excute the stop callback to check if this is the frame where stop_arg matches,
            // If there is a match, the execution never returns here, aka the `match code` block below this will not run.
            // If there is no match, in happy case NO_Reason will be returned or END OF STACK which will trigger and error return.
            stop(
                1,
                UnwindAction::FORCE_UNWIND
                    | UnwindAction::CLEANUP_PHASE
                    | if frame.is_none() {
                        UnwindAction::END_OF_STACK
                    } else {
                        UnwindAction::empty()
                    },
                (*exception).exception_class,
                exception,
                &mut UnwindContext {
                    frame: frame.as_ref(),
                    ctx,
                    signal,
                },
                stop_arg,
            )
        };
        match code {
            // Do Nothing, continue ahead.
            UnwindReasonCode::NO_REASON => (),
            _ => return UnwindReasonCode::FATAL_PHASE2_ERROR,
        }

        if let Some(frame) = frame {
            if let Some(personality) = frame.personality() {
                let code = unsafe {
                    // Run the personality routine on the current frame, aka destructors.
                    personality(
                        1,
                        UnwindAction::FORCE_UNWIND | UnwindAction::CLEANUP_PHASE,
                        (*exception).exception_class,
                        exception,
                        &mut UnwindContext {
                            frame: Some(&frame),
                            ctx,
                            signal,
                        },
                    )
                };

                match code {
                    // Continue Loop if nothing found.
                    UnwindReasonCode::CONTINUE_UNWIND => (),
                    UnwindReasonCode::INSTALL_CONTEXT => {
                        // We're jumping straight into this frame's landing pad instead of
                        // letting the call we unwound through actually return, so its
                        // outgoing-arg stack space never gets popped normally, free it
                        // now, then this frame's landing pad gets installed next
                        // where landing pad is for example: catch_unwind code,
                        // or a Drop Impl that needs to run before resuming unwind.
                        frame.adjust_stack_for_args(ctx);
                        return UnwindReasonCode::INSTALL_CONTEXT;
                    }
                    _ => return UnwindReasonCode::FATAL_PHASE2_ERROR,
                }
            }

            // Compute the caller's register state, move up one frame.
            *ctx = try2!(frame.unwind(ctx));
            signal = frame.is_signal_trampoline();
        } else {
            return UnwindReasonCode::END_OF_STACK;
        }
    }
}

#[inline(never)]
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn _Unwind_Resume(exception: *mut UnwindException) -> ! {
    with_context(|ctx| {
        // Check the presence of a stop fn, if present it means its a forced unwind, otherwise its a normal unwind.
        let code = match unsafe { (*exception).private_1 } {
            // If a normal unwind, continue the phase 2, with the handler cfa reinstated.
            None => {
                let handler_cfa = unsafe { (*exception).private_2 };
                raise_exception_phase2(exception, ctx, handler_cfa)
            }
            // If a forced unwind, continue the phase 2 forced unwind, with the stop arg and stop fn reinstated.
            Some(stop) => {
                let stop_arg = unsafe { (*exception).private_2 as _ };
                force_unwind_phase2(exception, ctx, stop, stop_arg)
            }
        };
        // `_Unwind_Resume` returns `!`, so it can only jump to the next landing
        // pad, never report back.
        // Two guarantees make `INSTALL_CONTEXT` certain:
        // phase 1 already found a handler up the stack in phase 1 (recorded in the private
        // data), and unwinding only moves toward it, so phase 2 must reach a
        // landing pad and return `INSTALL_CONTEXT` before it could run off the
        // end.
        // For the forced-unwind branch, the guarantee is instead the stop
        // function's contract: it is expected to take control (longjmp) at or
        // before the end of the stack, so phase 2 likewise reaches a pad.
        assert!(code == UnwindReasonCode::INSTALL_CONTEXT);

        // Exchance the values in CPU registers with the Context values, and jump there aka execute the landing pad!
        unsafe { restore_context(ctx) }
    })
}

// When a Catch Pad, re-propagates.
#[inline(never)]
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn _Unwind_Resume_or_Rethrow(
    exception: *mut UnwindException,
) -> UnwindReasonCode {
    // Check if this is forced unwind or nomral unwind
    let stop = match unsafe { (*exception).private_1 } {
        // On Normal unwind, we need to restart from phase 1 to get the next handler, cause the previous handler was already caught but catch decided to re propagate it!
        None => return unsafe { _Unwind_RaiseException(exception) },
        Some(v) => v,
    };
    // Since catch pad doesn't matter in forced unwind, we just continue as normal, and this doesn't return.
    with_context(|ctx| {
        let stop_arg = unsafe { (*exception).private_2 as _ };
        let code = force_unwind_phase2(exception, ctx, stop, stop_arg);
        assert!(code == UnwindReasonCode::INSTALL_CONTEXT);

        unsafe { restore_context(ctx) }
    })
}

// Run the exception cleanup code when foreign exception class is caught!,
// If the cleanup is on rust exception, we just unwrap from C type to Rust owned type and unwrap it oursleves!
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _Unwind_DeleteException(exception: *mut UnwindException) {
    if let Some(cleanup) = unsafe { (*exception).exception_cleanup } {
        unsafe { cleanup(UnwindReasonCode::FOREIGN_EXCEPTION_CAUGHT, exception) };
    }
}

// Read only stack walk, no exception, no handler, no cleanup.
#[inline(never)]
#[unsafe(no_mangle)]
pub extern "C-unwind" fn _Unwind_Backtrace(
    trace: UnwindTraceFn,
    trace_argument: *mut c_void,
) -> UnwindReasonCode {
    with_context(|ctx| {
        let mut ctx = ctx.clone();
        let mut signal = false;
        // walk keeps skipping until it reaches the frame whose function is _Unwind_Backtrace as seen in the `eq` op done on intial address.
        // cause these first few frames are the unwinders frames, so they are noise, we can skip them when `hide-trace` is enabled.
        let mut skipping = cfg!(feature = "hide-trace");

        loop {
            let frame = try1!(Frame::from_context(&ctx, signal));
            if !skipping {
                // Trace the current frame!
                let code = trace(
                    &UnwindContext {
                        frame: frame.as_ref(),
                        ctx: &mut ctx,
                        signal,
                    },
                    trace_argument,
                );
                // Callback will return NO_REASOn, as it has no other purpose.
                match code {
                    UnwindReasonCode::NO_REASON => (),
                    // Return if any other code is there aka on error!
                    _ => return UnwindReasonCode::FATAL_PHASE1_ERROR,
                }
            }
            if let Some(frame) = frame {
                if skipping {
                    // If we have reached teh _Unwind Backtrace itself, we can stop skipping now, next frame will be that of the caller of Unwind Backtrace,
                    // which is not noise but the acutall user code.
                    if frame.initial_address() == _Unwind_Backtrace as *const () as usize {
                        skipping = false;
                    }
                }

                // Compute the caller's register state, move up one frame.
                ctx = try1!(frame.unwind(&ctx));
                signal = frame.is_signal_trampoline();
            } else {
                // If no frame we have reached the end of stack.
                return UnwindReasonCode::END_OF_STACK;
            }
        }
    })
}
