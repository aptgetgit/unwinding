use gimli::{EndianSlice, NativeEndian, Pointer};

pub type StaticSlice = EndianSlice<'static, NativeEndian>;

// Create the largest possible slice for this address.
pub unsafe fn get_unlimited_slice<'a>(start: *const u8) -> &'a [u8] {
    let start = start as usize;
    // Rust slice's total byte size must not exceed isize::MAX,
    // Therefore it is the largest possible slice that can be constructed in Rust.
    let end = start.saturating_add(isize::MAX as usize);
    let len = end - start;
    unsafe { core::slice::from_raw_parts(start as *const u8, len) }
}

// Get the address from pointer.
pub unsafe fn deref_pointer(ptr: Pointer) -> usize {
    match ptr {
        // x is the address we need.
        Pointer::Direct(x) => x as usize,
        // x is the address of the location where the actuall address we need resides.
        Pointer::Indirect(x) => unsafe { *(x as *const usize) },
    }
}

#[cfg(feature = "libc")]
pub use libc::c_int;

#[cfg(not(feature = "libc"))]
#[allow(non_camel_case_types)]
pub type c_int = i32;

#[cfg(all(
    any(feature = "panic", feature = "panic-handler-dummy"),
    feature = "libc"
))]
pub fn abort() -> ! {
    unsafe { libc::abort() };
}

#[cfg(all(
    any(feature = "panic", feature = "panic-handler-dummy"),
    not(feature = "libc")
))]
// Trap by calling an illegal instruction, no cleanup or unwinding done.
pub fn abort() -> ! {
    core::intrinsics::abort();
}
