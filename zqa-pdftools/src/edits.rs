use std::{cmp::Ordering, collections::BTreeSet};

use crate::fonts::FontSizeMarker;

#[derive(Clone, Debug)]
pub(crate) enum EditType {
    Insert(String),
    Replace(String),
}

#[derive(Clone, Debug)]
pub(crate) struct Edit {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) r#type: EditType,
}

impl PartialEq for Edit {
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.end == other.end
    }
}

impl Eq for Edit {}

impl PartialOrd for Edit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Edit {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.start < other.start {
            return Ordering::Greater;
        }

        if self.start > other.start {
            return Ordering::Less;
        }

        other.end.cmp(&self.end)
    }
}

#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn apply_edits(edits: &[Edit], input: &mut String, update_idx: &mut [FontSizeMarker]) {
    let sorted_edits = edits.iter().collect::<BTreeSet<_>>();

    if update_idx.is_empty() {
        for edit in &sorted_edits {
            match &edit.r#type {
                EditType::Insert(ins) => input.insert_str(edit.start, ins),
                EditType::Replace(new) => input.replace_range(edit.start..edit.end, new),
            }
        }
        return;
    }

    let mut updates = vec![0_i64; update_idx.len()];

    for edit in &sorted_edits {
        match &edit.r#type {
            EditType::Insert(ins) => {
                input.insert_str(edit.start, ins);
            }
            EditType::Replace(new) => {
                input.replace_range(edit.start..edit.end, new);
            }
        }

        let change_start_idx = match update_idx.binary_search_by_key(&edit.start, |f| f.byte_index)
        {
            Ok(i) | Err(i) => i.min(updates.len() - 2),
        };

        for val in &mut updates[change_start_idx + 1..] {
            match &edit.r#type {
                EditType::Insert(ins) => {
                    *val += ins.len() as i64;
                }
                EditType::Replace(new) => {
                    *val += new.len() as i64 - (edit.end as i64 - edit.start as i64);
                }
            }
        }
    }

    for (orig, update) in update_idx.iter_mut().zip(updates) {
        orig.byte_index = (orig.byte_index as i64 + update) as usize;
    }
}
