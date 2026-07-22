// user-supplied FDE lookup via a custom callback/trait, for environments none of the built-in strategies fit.
#[cfg(feature = "fde-custom")]
mod custom;
// fixed, build/link-time-known FDE table, single static image, no dynamic loading.
#[cfg(feature = "fde-static")]
mod fixed;
// binary-search FDE lookup via the GNU .eh_frame_hdr index section. Used for fast lookups.
#[cfg(feature = "fde-gnu-eh-frame-hdr")]
// locates .eh_frame/.eh_frame_hdr by walking ELF program headers (PT_GNU_EH_FRAME) across loaded modules.
mod gnu_eh_frame_hdr;
#[cfg(feature = "fde-phdr")]
mod phdr;
// FDEs explicitly registered/deregistered at runtime via an API (e.g. for JIT-generated code).
#[cfg(feature = "fde-registry")]
mod registry;

use crate::util::*;
use gimli::{BaseAddresses, EhFrame, FrameDescriptionEntry};

#[cfg(feature = "fde-custom")]
pub mod custom_eh_frame_finder {
    pub use super::custom::{
        EhFrameFinder, FrameInfo, FrameInfoKind, SetCustomEhFrameFinderError,
        set_custom_eh_frame_finder,
    };
}

#[derive(Debug)]
pub struct FDESearchResult {
    // FDE for a given PC.
    pub fde: FrameDescriptionEntry<StaticSlice>,
    // Section base addresses for resolving DW_EH_PE-relative pointer encodings in `.eh_frame`.
    pub bases: BaseAddresses,
    // Parsed .eh_frame handle, needed to evaluate this FDE's CFI program.
    pub eh_frame: EhFrame<StaticSlice>,
}

pub trait FDEFinder {
    fn find_fde(&self, pc: usize) -> Option<FDESearchResult>;
}

pub struct GlobalFinder(());

impl FDEFinder for GlobalFinder {
    // Find FDE for a given PC or return None.
    fn find_fde(&self, pc: usize) -> Option<FDESearchResult> {
        #[cfg(feature = "fde-custom")]
        if let Some(v) = custom::get_finder().find_fde(pc) {
            return Some(v);
        }
        #[cfg(feature = "fde-registry")]
        if let Some(v) = registry::get_finder().find_fde(pc) {
            return Some(v);
        }
        #[cfg(feature = "fde-gnu-eh-frame-hdr")]
        if let Some(v) = gnu_eh_frame_hdr::get_finder().find_fde(pc) {
            return Some(v);
        }
        #[cfg(feature = "fde-phdr")]
        if let Some(v) = phdr::get_finder().find_fde(pc) {
            return Some(v);
        }
        #[cfg(feature = "fde-static")]
        if let Some(v) = fixed::get_finder().find_fde(pc) {
            return Some(v);
        }
        None
    }
}

// zero-cost singleton for Finder.
pub fn get_finder() -> &'static GlobalFinder {
    &GlobalFinder(())
}
