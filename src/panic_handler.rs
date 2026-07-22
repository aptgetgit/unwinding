use crate::abi::*;
use crate::print::*;
use alloc::boxed::Box;
use core::any::Any;
use core::cell::Cell;
use core::ffi::c_void;
use core::panic::{Location, PanicInfo};
use core::sync::atomic::{AtomicI32, Ordering};

// Thread local panic count, since unwinding happens on the stack of a particular thread,
// Panic count prenvents two unwinds running on the same stack, cause it will clobber the stack!
#[thread_local]
static PANIC_COUNT: Cell<usize> = Cell::new(0);

// Passes -lc to the linker, aka link against libc,
// since this crate is no_std, but we are using libc fns below!
#[link(name = "c")]
unsafe extern "C" {}

pub(crate) fn drop_panic() {
    eprintln!("Rust panics must be rethrown");
}

pub(crate) fn foreign_exception() {
    eprintln!("Rust cannot catch foreign exceptions");
}

pub(crate) fn panic_caught() {
    PANIC_COUNT.set(0);
}

fn check_env() -> bool {
    static ENV: AtomicI32 = AtomicI32::new(-1);

    let env = ENV.load(Ordering::Relaxed);
    if env != -1 {
        return env != 0;
    }

    let val = unsafe {
        // Get RUST BACKTRACE env value
        let ptr = libc::getenv(b"RUST_BACKTRACE\0".as_ptr() as _);
        if ptr.is_null() {
            b""
        } else {
            let len = libc::strlen(ptr);
            core::slice::from_raw_parts(ptr as *const u8, len)
        }
    };
    let (note, env) = match val {
        b"" => (true, false),
        b"1" | b"full" => (false, true),
        _ => (false, false),
    };

    // Issue a note for the first panic.
    if ENV.swap(env as _, Ordering::Relaxed) == -1 && note {
        eprintln!("note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace");
    }
    env
}

fn stack_trace() {
    struct CallbackData {
        counter: usize,
    }
    // `UnwindTraceFn`, callback fn, increment the counter, and print the return address.
    extern "C" fn callback(unwind_ctx: &UnwindContext<'_>, arg: *mut c_void) -> UnwindReasonCode {
        let data = unsafe { &mut *(arg as *mut CallbackData) };
        data.counter += 1;
        eprintln!(
            "{:4}:{:#19x} - <unknown>",
            data.counter,
            _Unwind_GetIP(unwind_ctx)
        );
        UnwindReasonCode::NO_REASON
    }
    // Init the Counter.
    let mut data = CallbackData { counter: 0 };
    // Pass the callback.
    _Unwind_Backtrace(callback, &mut data as *mut _ as _);
}

fn do_panic(msg: Box<dyn Any + Send>) -> ! {
    // Abort if panicked during panic!
    if PANIC_COUNT.get() >= 1 {
        stack_trace();
        eprintln!("thread panicked while processing panic. aborting.");
        crate::util::abort();
    }
    PANIC_COUNT.set(1);
    // If Bakctrace is enabled, do a stack trace first!
    if check_env() {
        stack_trace();
    }
    // Wrap the Rust panic type to C exception type and Raise Exception.
    let code = crate::panic::begin_panic(Box::new(msg));
    // Raise Exception will never return in success case, something is wrong, thus abort.
    eprintln!("failed to initiate panic, error {}", code.0);
    crate::util::abort();
}

#[panic_handler]
fn panic(info: &PanicInfo<'_>) -> ! {
    eprintln!("{}", info);

    struct NoPayload;
    do_panic(Box::new(NoPayload))
}

// #[track_caller] makes the compiler pass a hidden &Location argument holding the call site,
// so Location::caller() inside the function reports where it was called from rather than its own line
// and it propagates up through other #[track_caller] callers to the first untracked one.
#[track_caller]
pub fn panic_any<M: 'static + Any + Send>(msg: M) -> ! {
    eprintln!("panicked at {}", Location::caller());
    do_panic(Box::new(msg))
}
