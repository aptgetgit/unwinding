use super::FDESearchResult;
use crate::util::*;

use gimli::{BaseAddresses, EhFrame, NativeEndian, UnwindSection};

pub struct StaticFinder(());

// zero-cost singleton for Static Finder.
pub fn get_finder() -> &'static StaticFinder {
    &StaticFinder(())
}

unsafe extern "C" {
    // Symbol emitted on default by ld, marks the very start of the loaded image.
    static __executable_start: u8;
    // Symbol emitted on default by ld, marks the end of `.text` section.
    static __etext: u8;
    // Custom symbol must be emitted while linking, as specified in `README.md`, marks the start of `.eh_frame` section.
    static __eh_frame: u8;
}

impl super::FDEFinder for StaticFinder {
    fn find_fde(&self, pc: usize) -> Option<FDESearchResult> {
        unsafe {
            let text_start = &__executable_start as *const u8 as usize;
            let text_end = &__etext as *const u8 as usize;
            if !(text_start..text_end).contains(&pc) {
                return None;
            }

            let eh_frame = &__eh_frame as *const u8 as usize;
            // Section base addresses gimli adds to DW_EH_PE_pcrel/textrel/datarel-encoded
            // pointers in .eh_frame to resolve them into real absolute addresses.
            let bases = BaseAddresses::default()
                .set_eh_frame(eh_frame as _) // base for DW_EH_PE_datarel
                .set_text(text_start as _); // base for DW_EH_PE_textrel
            let eh_frame = EhFrame::new(get_unlimited_slice(eh_frame as _), NativeEndian);

            // FDE (Frame Description Entry): a per-function .eh_frame record. Maps a PC
            // range [start, start+len) to a CFI(Call Frame Info, a instruction format used inside both CIEs and FDEs) bytecode program
            // describing how CFA and register-recovery rules evolve as PC advances through that function.
            // FDE References a CIE for shared defaults it builds on top of.
            // where CIE: Common information Entry, shared template referenced by many FDEs.
            if let Ok(fde) = eh_frame.fde_for_address(&bases, pc as _, EhFrame::cie_from_offset) {
                return Some(FDESearchResult {
                    fde,
                    bases,
                    eh_frame,
                });
            }

            None
        }
    }
}
