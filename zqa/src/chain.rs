use ftail::Ftail;

pub mod tasks;
pub mod utils;

use rag::task::Chain;
use tasks::input::{UserInput, UserInputTask};

pub fn get_zotero_qa_chain() -> Chain<UserInput> {
    Chain {
        tasks: vec![Box::new(UserInputTask::default())],
        name: "ZoteroQAChain".to_string(),
        ..Default::default()
    }
}

pub fn main() {
    Ftail::new().console(log::LevelFilter::Info).init().unwrap();
    let mut chain = get_zotero_qa_chain();
    chain.run();
}
