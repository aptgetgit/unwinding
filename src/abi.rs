use core::ffi::c_void;
use core::ops;

use crate::util::*;

#[cfg(not(feature = "unwinder"))]
use crate::arch::Arch;
#[cfg(feature = "unwinder")]
pub use crate::unwinder::*;

/// Return/status code for the two-phase unwind protocol (Itanium C++ ABI).
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct UnwindReasonCode(pub c_int);

// Two "not yet" codes exist because normal unwind and forced unwind hand the
// decision to different callbacks:
//
//   - Personality routine (phase 1 search) checks the frame's LSD/type
//     table for a matching catch clause -> CONTINUE_UNWIND if no match.
//   - Stop function (forced unwind only) checks whatever the caller cares
//     about, e.g. glibc's unwind_stop checks "have I reached the
//     thread's entry trampoline?" -> NO_REASON if not there yet.
impl UnwindReasonCode {
    /// Used in Forced unwind only: stop function's "not the destination frame yet" signal.
    /// Not used by personality routines in phase 1/2 (they use CONTINUE_UNWIND).
    pub const NO_REASON: Self = Self(0);

    /// Passed to exception_cleanup fn `UnwindExceptionCleanupFn`:
    /// a different language runtime caught this exception
    /// (e.g. a Java exception reaching a C++ catch(...)).
    pub const FOREIGN_EXCEPTION_CAUGHT: Self = Self(1);

    /// Personality routine hit an unrecoverable error during phase 2
    /// (cleanup), e.g. detected stack corruption.
    pub const FATAL_PHASE2_ERROR: Self = Self(2);

    /// Personality routine hit an unrecoverable error during phase 1
    /// (search), other than the specific defined error codes.
    pub const FATAL_PHASE1_ERROR: Self = Self(3);

    /// Forced unwind: stop function aka `UnwindStopFn` reached its target frame and is finished.
    pub const NORMAL_STOP: Self = Self(4);

    /// Phase 1: walked off the top of the stack without finding a handler.
    /// Also used in forced unwind for the equivalent end-of-stack case.
    pub const END_OF_STACK: Self = Self(5);

    /// Phase 1: personality routine found a frame that catches this exception.
    pub const HANDLER_FOUND: Self = Self(6);

    /// Phase 2: personality routine set up landing-pad registers via this
    /// context; may be the actual handler frame OR an intermediate cleanup-
    /// only landing pad (which calls _Unwind_Resume when it finishes).
    pub const INSTALL_CONTEXT: Self = Self(7);

    /// This frame has no landing pad to run; keep walking outward.
    pub const CONTINUE_UNWIND: Self = Self(8);
}

/// Bitset passed *into* a personality routine (or stop function) describing
/// why/how it's being invoked. First four flags are base Itanium ABI;
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct UnwindAction(pub c_int);

impl UnwindAction {
    /// Phase 1: check for a handler only; cannot be combined with CLEANUP_PHASE.
    pub const SEARCH_PHASE: Self = Self(1);
    /// Phase 2: perform cleanup and/or set up a landing pad transfer.
    pub const CLEANUP_PHASE: Self = Self(2);
    /// Phase 2 only: this frame was flagged as the handler in phase 1,
    /// the personality routine must not change its mind here and must set
    /// up the landing pad (via _Unwind_SetGR/_Unwind_SetIP) and return
    /// INSTALL_CONTEXT.
    pub const HANDLER_FRAME: Self = Self(4);
    /// Set during forced unwind (longjmp / thread cancellation): a catch(...)
    /// may still execute as pass-through code, but no language's catch
    /// semantics may terminate the unwind here, it must call _Unwind_Resume
    /// when done, since only the stop function decides when to stop.
    pub const FORCE_UNWIND: Self = Self(8);
    /// GCC extension, not in the base ABI: forced unwind reached the last
    /// frame (stack pointer is NULL), letting the stop function detect
    /// end-of-stack explicitly.
    pub const END_OF_STACK: Self = Self(16);
}

impl ops::BitOr for UnwindAction {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl UnwindAction {
    #[inline]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[inline]
    pub const fn contains(&self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

/// Frees/releases an exception object once the runtime is done with it
/// (e.g. an uncaught panic payload, or after a handler finishes).
/// Stored in `UnwindException::exception_cleanup`.
pub type UnwindExceptionCleanupFn = unsafe extern "C" fn(UnwindReasonCode, *mut UnwindException);

/// Forced-unwind destination check, supplied by the caller of
/// `_Unwind_ForcedUnwind` (e.g. glibc's `unwind_stop` for `pthread_cancel`).
///
/// Called at every frame in place of a personality routine, since forced
/// unwind has no exception type to match against, this callback alone
/// decides when the walk stops. Returns `NO_REASON` to keep walking;
/// `NORMAL_STOP` once it recognizes the target frame.
pub type UnwindStopFn = unsafe extern "C" fn(
    c_int,        // version
    UnwindAction, // actions (phase/force flags)
    u64,          // exception_class
    *mut UnwindException,
    &mut UnwindContext<'_>, // mutable: may need to set registers at the target
    *mut c_void, // stop_parameter: caller-supplied data used to recognize/reach the target frame where the forced unwind should stop, can be anything
) -> UnwindReasonCode;

#[cfg(not(feature = "unwinder"))]
#[repr(C)]
pub struct UnwindException {
    pub exception_class: u64,
    pub exception_cleanup: Option<UnwindExceptionCleanupFn>,
    private: [usize; Arch::UNWIND_PRIVATE_DATA_SIZE],
}

/// Per-frame callback for `_Unwind_Backtrace` (pure introspection, e.g. `backtrace()`).
/// Return `NO_REASON` to keep walking; anything else aborts the backtrace immediately.
pub type UnwindTraceFn =
    extern "C" fn(ctx: &UnwindContext<'_>, arg: *mut c_void) -> UnwindReasonCode;

#[cfg(not(feature = "unwinder"))]
#[repr(C)]
pub struct UnwindContext<'a> {
    opaque: core::cell::UnsafeCell<()>,
    phantom: core::marker::PhantomData<(&'a (), *mut (), core::marker::PhantomPinned)>,
}

/// Language-specific handler-matching logic, called during both phase 1 search
/// and phase 2 cleanup. Consults the frame's LSD to decide: catch, no catch, or
/// (cleanup phase, handler frame) install the landing pad via `context`.
pub type PersonalityRoutine = unsafe extern "C" fn(
    c_int,        // version
    UnwindAction, // actions (phase/handler-frame flags)
    u64,          // exception_class
    *mut UnwindException,
    &mut UnwindContext<'_>, // mutable: INSTALL_CONTEXT writes landing-pad registers here
) -> UnwindReasonCode;

#[cfg(not(feature = "unwinder"))]
macro_rules! binding {
    () => {};
    (unsafe extern $abi: literal fn $name: ident ($($arg: ident : $arg_ty: ty),*$(,)?) $(-> $ret: ty)?; $($rest: tt)*) => {
        unsafe extern $abi {
            pub unsafe fn $name($($arg: $arg_ty),*) $(-> $ret)?;
        }
        binding!($($rest)*);
    };

    (extern $abi: literal fn $name: ident ($($arg: ident : $arg_ty: ty),*$(,)?) $(-> $ret: ty)?; $($rest: tt)*) => {
        unsafe extern $abi {
            pub safe fn $name($($arg: $arg_ty),*) $(-> $ret)?;
        }
        binding!($($rest)*);
    };
}

// Feature "unwinder": no bindings are generated, functions come from
// `crate::unwinder` (imported above via `pub use crate::unwinder::*`).
// Each `const _: <expected sig> = $name;` is a compile-time-only check
// that the provided implementation's signature (args/return/ABI/unsafety)
// matches exactly what the unwind protocol requires. No code is emitted.
#[cfg(feature = "unwinder")]
macro_rules! binding {
    () => {};
    (unsafe extern $abi: literal fn $name: ident ($($arg: ident : $arg_ty: ty),*$(,)?) $(-> $ret: ty)?; $($rest: tt)*) => {
        const _: unsafe extern $abi fn($($arg_ty),*) $(-> $ret)? = $name;
    };

    (extern $abi: literal fn $name: ident ($($arg: ident : $arg_ty: ty),*$(,)?) $(-> $ret: ty)?; $($rest: tt)*) => {
        const _: extern $abi fn($($arg_ty),*) $(-> $ret)? = $name;
    };
}

binding! {
    extern "C" fn _Unwind_GetGR(unwind_ctx: &UnwindContext<'_>, index: c_int) -> usize;
    extern "C" fn _Unwind_GetCFA(unwind_ctx: &UnwindContext<'_>) -> usize;
    extern "C" fn _Unwind_SetGR(
        unwind_ctx: &mut UnwindContext<'_>,
        index: c_int,
        value: usize,
    );
    extern "C" fn _Unwind_GetIP(unwind_ctx: &UnwindContext<'_>) -> usize;
    extern "C" fn _Unwind_GetIPInfo(
        unwind_ctx: &UnwindContext<'_>,
        ip_before_insn: &mut c_int,
    ) -> usize;
    extern "C" fn _Unwind_SetIP(
        unwind_ctx: &mut UnwindContext<'_>,
        value: usize,
    );
    extern "C" fn _Unwind_GetLanguageSpecificData(unwind_ctx: &UnwindContext<'_>) -> *mut c_void;
    extern "C" fn _Unwind_GetRegionStart(unwind_ctx: &UnwindContext<'_>) -> usize;
    extern "C" fn _Unwind_GetTextRelBase(unwind_ctx: &UnwindContext<'_>) -> usize;
    extern "C" fn _Unwind_GetDataRelBase(unwind_ctx: &UnwindContext<'_>) -> usize;
    extern "C" fn _Unwind_FindEnclosingFunction(pc: *mut c_void) -> *mut c_void;
    unsafe extern "C-unwind" fn _Unwind_RaiseException(
        exception: *mut UnwindException,
    ) -> UnwindReasonCode;
    unsafe extern "C-unwind" fn _Unwind_ForcedUnwind(
        exception: *mut UnwindException,
        stop: UnwindStopFn,
        stop_arg: *mut c_void,
    ) -> UnwindReasonCode;
    unsafe extern "C-unwind" fn _Unwind_Resume(exception: *mut UnwindException) -> !;
    unsafe extern "C-unwind" fn _Unwind_Resume_or_Rethrow(
        exception: *mut UnwindException,
    ) -> UnwindReasonCode;
    unsafe extern "C" fn _Unwind_DeleteException(exception: *mut UnwindException);
    extern "C-unwind" fn _Unwind_Backtrace(
        trace: UnwindTraceFn,
        trace_argument: *mut c_void,
    ) -> UnwindReasonCode;
}
