pub mod input;
pub mod library;

use crate::input::{UserInput, UserInputTask};
use rag::task::Chain;

pub fn get_zotero_qa_chain() -> Chain<UserInput> {
    Chain {
        tasks: vec![Box::new(UserInputTask::default())],
        name: "ZoteroQAChain".to_string(),
        ..Default::default()
    }
}

pub fn main() {
    let mut chain = get_zotero_qa_chain();
    chain.run();
}
