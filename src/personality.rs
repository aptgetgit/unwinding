// References:
// https://github.com/rust-lang/rust/blob/c4be230b4a30eb74e3a3908455731ebc2f731d3d/library/panic_unwind/src/gcc.rs
// https://github.com/rust-lang/rust/blob/c4be230b4a30eb74e3a3908455731ebc2f731d3d/library/panic_unwind/src/dwarf/eh.rs
// https://docs.rs/gimli/0.25.0/src/gimli/read/cfi.rs.html

use core::mem;
use gimli::{EndianSlice, Error, Pointer, Reader};
use gimli::{NativeEndian, constants};

use crate::abi::*;
use crate::arch::*;
use crate::util::*;

// Exception Handling Action: Verdict about one frame at one PC,
// aka given the current PC in this frame, what does the LSDA suggest to do here?
#[derive(Debug)]
enum EHAction {
    // The call Site entry exists but its landing pad field is zero, aka compiler tracked but nothing needs to be done!
    None,

    // `ttype_index` says what the landing pad does:
    //   0   cleanup - pad always runs, then resumes unwinding
    //  >0   catch   - names one type; pad runs if the exception matches it
    //  <0   filter  - names a list of allowed types; pad runs if the exception is not in it.
    //
    // Drop glue code, Pad runs the destructors and then calls _Unwind_Resume.
    // ttype = 0, the index in ttype is 1-based, so slot 0 doesn't exists, thus 0 is free to repurpose, compilers use it to indicate cleanup(drop glue).
    Cleanup(usize),
    // ttype > 0, aka positive, since rust has only one exception type there is noting to match agains,
    // just run Catch Unwind, the pad is **Terminus** aka it does not resume normally.
    Catch(usize),
    // Emitted at no-unwind boundary, points the pad at `core::panicking::panic_cannot_unwind()`, which aborts.
    // where `extern "c"` is a no-unwind boundary and `extern "C-unwind"` is a unwind boundary which resolves to None (cs_lpad == 0) or Cleanup (ttype_index == 0),
    // this no pad for "C-unwind".
    //  ttype < 0, aka negative, always empty list, always fires, ex: any panic reaching "extern C" must abort!
    Filter(usize),
    // No entry for PC found in call site table, aka LSDA does not descibe this PC at all, making it a error condition, maps to FATAL Error.
    Terminate,
}

// Validate DWARF Exception Handling Pointer Encoding.
// Pointer Encoding: how to read the pointer that follows this byte.
fn parse_pointer_encoding(input: &mut StaticSlice) -> gimli::Result<constants::DwEhPe> {
    // Read the first byte
    let eh_pe = input.read_u8()?;
    let eh_pe = constants::DwEhPe(eh_pe);

    // Check if encoding is valid?
    if eh_pe.is_valid_encoding() {
        Ok(eh_pe)
    } else {
        Err(gimli::Error::UnknownPointerEncoding(eh_pe))
    }
}

// Read the pointer that follows, decoding it as the encoding byte specifies.
fn parse_encoded_pointer(
    encoding: constants::DwEhPe,
    unwind_ctx: &UnwindContext<'_>,
    input: &mut StaticSlice,
) -> gimli::Result<Pointer> {
    // No bytes to read aka no pointer stored here, marker for optional LSDA field.
    if encoding == constants::DW_EH_PE_omit {
        return Err(Error::CannotParseOmitPointerEncoding);
    }

    // What the value is relative to: compute that base address.
    let base = match encoding.application() {
        constants::DW_EH_PE_absptr => 0,
        constants::DW_EH_PE_pcrel => input.slice().as_ptr() as u64,
        constants::DW_EH_PE_textrel => _Unwind_GetTextRelBase(unwind_ctx) as u64,
        constants::DW_EH_PE_datarel => _Unwind_GetDataRelBase(unwind_ctx) as u64,
        constants::DW_EH_PE_funcrel => _Unwind_GetRegionStart(unwind_ctx) as u64,
        constants::DW_EH_PE_aligned => {
            // DW_EH_PE_aligned means the same as DW_EH_PE_absptr, but that the address is naturally
            // aligned.  In reality it's not being emitted (and libunwind doesn't support it) but
            // it's not tricky to implement so do it anyway.
            let ptr = input.slice().as_ptr() as usize;
            input.skip(ptr.next_multiple_of(size_of::<usize>()) - ptr)?;
            0
        }
        _ => unreachable!(),
    };

    // How the value itself is stored: read it in that form to get the offset.
    let offset = match encoding.format() {
        constants::DW_EH_PE_absptr => input.read_address(mem::size_of::<usize>() as _),
        constants::DW_EH_PE_uleb128 => input.read_uleb128(),
        constants::DW_EH_PE_udata2 => input.read_u16().map(u64::from),
        constants::DW_EH_PE_udata4 => input.read_u32().map(u64::from),
        constants::DW_EH_PE_udata8 => input.read_u64(),
        constants::DW_EH_PE_sleb128 => input.read_sleb128().map(|a| a as u64),
        constants::DW_EH_PE_sdata2 => input.read_i16().map(|a| a as u64),
        constants::DW_EH_PE_sdata4 => input.read_i32().map(|a| a as u64),
        constants::DW_EH_PE_sdata8 => input.read_i64().map(|a| a as u64),
        _ => unreachable!(),
    }?;

    // Get the actuall address
    let address = base.wrapping_add(offset);
    Ok(if encoding.is_indirect() {
        Pointer::Indirect(address)
    } else {
        Pointer::Direct(address)
    })
}

// LSDA format:
// LSDA
// ├── Header
// │    ├── lpad_base: base that landing pad offsets are added to; usually omitted, then func_start.
// │    ├── ttype_base: end of the Type Table; indices count backward from here.
// │    ├── call_site_encoding: how the Call Site Table rows are encoded.
// │    └── call_site_table_length: length of the Call Site Table, and where the Action Table starts.
// ├── Call Site Table  (rows, sorted by address)
// │    ├── cs_start: start of the PC range, relative to func_start.
// │    ├── cs_len: length of the PC range.
// │    ├── cs_lpad: landing pad, relative to lpad_base; 0 means nothing to do.
// │    └── cs_action: 1-based index into the Action Table; 0 means cleanup.
// ├── Action Table  (rows): labels what the pad does; its address alone cannot say
// │    │                     whether it catches or only cleans up, and phase 1 must know before running anything.
// │    ├── ttype_index: 0 = cleanup, >0 = catch, <0 = filter.
// │    └── next_offset: next action for this call site; 0 ends the chain.
// └── Type Table: Never read, a null slot always matches(>0 side), and an empty filter list can
//      │          contain nothing(<0 side), so both answers are fixed before the lookup happens,
//      │          the sign of ttype_index is all that is left to learn. This is why
//      │          ttype_base is parsed only to skip past it.
//      │
//      └── one type per slot, indexed backward from ttype_base; null in Rust.
//          (null = catch-all, since every Rust panic is the same exception type).
//
// What Exception Handling Action should be done for this context!
fn find_eh_action(
    reader: &mut StaticSlice,
    unwind_ctx: &UnwindContext<'_>,
) -> gimli::Result<EHAction> {
    // Get the start address for this function/frame.
    let func_start = _Unwind_GetRegionStart(unwind_ctx);

    // Is this a signal frame or not? adjust Instruction Pointer accordingly!
    let mut ip_before_instr = 0;
    let ip = _Unwind_GetIPInfo(unwind_ctx, &mut ip_before_instr);
    let ip = if ip_before_instr != 0 { ip } else { ip - 1 };

    // Get Actuall address for the landing pad base.
    let start_encoding = parse_pointer_encoding(reader)?;
    let lpad_base = if !start_encoding.is_absent() {
        unsafe { deref_pointer(parse_encoded_pointer(start_encoding, unwind_ctx, reader)?) }
    } else {
        func_start
    };

    // Get the ttype, just read it to advance the reader, as we don't care about it in rust.
    let ttype_encoding = parse_pointer_encoding(reader)?;
    if !ttype_encoding.is_absent() {
        reader.read_uleb128()?;
    }

    // Split and get the call site table and action table.
    let call_site_encoding = parse_pointer_encoding(reader)?;
    let call_site_table_length = reader.read_uleb128()?;
    let (mut call_site_table, mut action_table) = reader.split_at(call_site_table_length as _);

    while !call_site_table.is_empty() {
        // Get Gall site start range for the frame.
        let cs_start = unsafe {
            deref_pointer(parse_encoded_pointer(
                call_site_encoding,
                unwind_ctx,
                &mut call_site_table,
            )?)
        };
        // call site length for that frame.
        let cs_len = unsafe {
            deref_pointer(parse_encoded_pointer(
                call_site_encoding,
                unwind_ctx,
                &mut call_site_table,
            )?)
        };
        // Get the landing pad for the frame.
        let cs_lpad = unsafe {
            deref_pointer(parse_encoded_pointer(
                call_site_encoding,
                unwind_ctx,
                &mut call_site_table,
            )?)
        };
        // Get the call site action value for the frame.
        let cs_action = call_site_table.read_uleb128()?;

        // If current PC/IP is smaller than call site table start range, that means its not in this row, move to next row!.
        if ip < func_start + cs_start {
            break;
        }
        // If its inside the range:
        if ip < func_start + cs_start + cs_len {
            if cs_lpad == 0 {
                return Ok(EHAction::None);
            } else {
                // Get landing pad address.
                let lpad = lpad_base + cs_lpad;
                // If action is cleanup, return the lpad.
                if cs_action == 0 {
                    return Ok(EHAction::Cleanup(lpad));
                }
                // If cs action is anything else other than zero, get the ttype to determine what to do!
                action_table.skip((cs_action - 1) as _)?;
                let ttype_index = action_table.read_sleb128()?;
                return Ok(if ttype_index == 0 {
                    EHAction::Cleanup(lpad)
                } else if ttype_index > 0 {
                    EHAction::Catch(lpad)
                } else {
                    EHAction::Filter(lpad)
                });
            }
        }
    }
    // Nothing in call site table, terminate.
    Ok(EHAction::Terminate)
}

// rust_eh_personality flow:
// ┌───────────┬────────────────────────┬─────────────────────────────────────────────────────────┐
// │           │ Phase 1 (SEARCH_PHASE) │                    Phase 2 (cleanup)                    │
// ├───────────┼────────────────────────┼─────────────────────────────────────────────────────────┤
// │ None      │ CONTINUE_UNWIND        │ CONTINUE_UNWIND                                         │
// ├───────────┼────────────────────────┼─────────────────────────────────────────────────────────┤
// │ Cleanup   │ CONTINUE_UNWIND        │ install pad                                             │
// ├───────────┼────────────────────────┼─────────────────────────────────────────────────────────┤
// │ Catch     │ HANDLER_FOUND          │ install pad                                             │
// ├───────────┼────────────────────────┼─────────────────────────────────────────────────────────┤
// │ Filter    │ HANDLER_FOUND          │ install pad — unless FORCE_UNWIND, then CONTINUE_UNWIND │
// ├───────────┼────────────────────────┼─────────────────────────────────────────────────────────┤
// │ Terminate │ FATAL_PHASE1_ERROR     │ FATAL_PHASE2_ERROR                                      │
// └───────────┴────────────────────────┴─────────────────────────────────────────────────────────┘
#[lang = "eh_personality"]
unsafe fn rust_eh_personality(
    version: c_int,
    actions: UnwindAction,
    _exception_class: u64,
    exception: *mut UnwindException,
    unwind_ctx: &mut UnwindContext<'_>,
) -> UnwindReasonCode {
    if version != 1 {
        return UnwindReasonCode::FATAL_PHASE1_ERROR;
    }

    // Get pointer to lanugage specific data area in augmentation data.
    let lsda = _Unwind_GetLanguageSpecificData(unwind_ctx);
    // If lsda does't exist for this fn, return continue unwind.
    if lsda.is_null() {
        return UnwindReasonCode::CONTINUE_UNWIND;
    }

    let mut lsda = EndianSlice::new(unsafe { get_unlimited_slice(lsda as _) }, NativeEndian);
    // Get the action that needs to be performed.
    let eh_action = match find_eh_action(&mut lsda, unwind_ctx) {
        Ok(v) => v,
        Err(_) => return UnwindReasonCode::FATAL_PHASE1_ERROR,
    };

    // If unwind action is in search phase!
    if actions.contains(UnwindAction::SEARCH_PHASE) {
        match eh_action {
            // Ignore Drop pads, as this is search phase only!
            // Catch and abort pads should be installed as handler.
            EHAction::None | EHAction::Cleanup(_) => UnwindReasonCode::CONTINUE_UNWIND,
            EHAction::Catch(_) | EHAction::Filter(_) => UnwindReasonCode::HANDLER_FOUND,
            EHAction::Terminate => UnwindReasonCode::FATAL_PHASE1_ERROR,
        }
    } else {
        match eh_action {
            EHAction::None => UnwindReasonCode::CONTINUE_UNWIND,
            // Forced unwinding hits a terminate action, then skip it and continue unwind, cause its forced.
            EHAction::Filter(_) if actions.contains(UnwindAction::FORCE_UNWIND) => {
                UnwindReasonCode::CONTINUE_UNWIND
            }
            // Otherwise run the lpad.
            EHAction::Cleanup(lpad) | EHAction::Catch(lpad) | EHAction::Filter(lpad) => {
                // Point register 0 to exception.
                _Unwind_SetGR(
                    unwind_ctx,
                    Arch::UNWIND_DATA_REG.0.0 as _,
                    exception as usize,
                );
                // Landing pads are entered with two values, and they answer different questions:
                //
                //   _Unwind_SetIP(lpad)   which pad do we jump to?
                //   UNWIND_DATA_REG.0     the exception pointer, so the pad can take the payload
                //                         or hand it back to _Unwind_Resume
                //   UNWIND_DATA_REG.1     the selector: which handler *inside* that pad
                //
                // The selector exists for languages where one pad holds several handlers. A C++
                // `try` with two `catch` clauses compiles to a single pad address, and the
                // personality writes 1 or 2 so the pad can branch to the right clause.
                //
                // Rust pads have a single path through them: drop this set of values and resume,
                // or take the payload and recover into this one `catch_unwind`. Arriving at the
                // pad already determines everything that happens, so there is nothing to select.
                // We write 0 to satisfy the ABI; the pad never reads it.
                //
                // The IP is still chosen per call site (`lpad_base + cs_lpad`) — Rust has many
                // pads, just never more than one handler inside any of them.
                _Unwind_SetGR(unwind_ctx, Arch::UNWIND_DATA_REG.1.0 as _, 0);
                _Unwind_SetIP(unwind_ctx, lpad);
                UnwindReasonCode::INSTALL_CONTEXT
            }
            EHAction::Terminate => UnwindReasonCode::FATAL_PHASE2_ERROR,
        }
    }
}
