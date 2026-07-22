use core::mem::ManuallyDrop;

use crate::abi::*;

pub unsafe trait Exception {
    const CLASS: [u8; 8];

    fn wrap(this: Self) -> *mut UnwindException;
    unsafe fn unwrap(ex: *mut UnwindException) -> Self;
}

pub fn begin_panic<E: Exception>(exception: E) -> UnwindReasonCode {
    // Clean Up fn- that unwraps the Unwind Exception type to Owned Rust Exception type, and frees it!
    unsafe extern "C" fn exception_cleanup<E: Exception>(
        _unwind_code: UnwindReasonCode,
        exception: *mut UnwindException,
    ) {
        unsafe { E::unwrap(exception) };
    }

    // Wrap the Exception type to C Unwind Exception type,
    // Add the exception class and exception clearnup callback.
    // Raies Exception, which will never return on happy path!
    let ex = E::wrap(exception);
    unsafe {
        (*ex).exception_class = u64::from_ne_bytes(E::CLASS);
        (*ex).exception_cleanup = Some(exception_cleanup::<E>);
        _Unwind_RaiseException(ex)
    }
}

// Catch Unwind runs with a closure F.
pub fn catch_unwind<E: Exception, R, F: FnOnce() -> R>(f: F) -> Result<R, Option<E>> {
    #[repr(C)]
    union Data<F, R, E> {
        f: ManuallyDrop<F>,
        r: ManuallyDrop<R>,
        p: ManuallyDrop<Option<E>>,
    }

    let mut data = Data {
        f: ManuallyDrop::new(f),
    };

    let data_ptr = &mut data as *mut _ as *mut u8;
    unsafe {
        // `catch_unwind`: try_fn runs with data_ptr as argument, if try_fn panics or forign unwind happen then do_catch runs!
        return if core::intrinsics::catch_unwind(do_call::<F, R>, data_ptr, do_catch::<E>) {
            Err(ManuallyDrop::into_inner(data.p))
        } else {
            Ok(ManuallyDrop::into_inner(data.r))
        };
    }

    // Extract the closure provided by user and execute it!
    // Save the result in the data.r field.
    #[inline]
    fn do_call<F: FnOnce() -> R, R>(data: *mut u8) {
        unsafe {
            let data = &mut *(data as *mut Data<F, R, ()>);
            let f = ManuallyDrop::take(&mut data.f);
            data.r = ManuallyDrop::new(f());
        }
    }

    // Runs when `try_fn` aka `do_call` panics or foreign unwind happens.
    #[cold]
    fn do_catch<E: Exception>(data: *mut u8, exception: *mut u8) {
        unsafe {
            // The Union data struct is logically unintialized cause `f` was taken out, and `r` was never achived, cause the `f` panicked.
            let data = &mut *(data as *mut ManuallyDrop<Option<E>>);
            let exception = exception as *mut UnwindException;
            // Check if this is rust exception or foerign exception,
            // If foregin exception, execute Delete exception, whiich basicaly runs cleanup callback if exists.
            if (*exception).exception_class != u64::from_ne_bytes(E::CLASS) {
                _Unwind_DeleteException(exception);
                // Or foreign exception, Data is made None and then None is returned.
                *data = ManuallyDrop::new(None);
                return;
            }
            // Otherwise just unwraps and data is overwritten, byt manyally drop prevents it from dropping!, and data is returned.
            *data = ManuallyDrop::new(Some(E::unwrap(exception)));
        }
    }
}
