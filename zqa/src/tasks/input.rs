use std::io::{stdin, stdout, Write};

use rag::task::{ReturnValue, Task};

#[derive(Clone)]
#[allow(dead_code)]
pub struct UserInput {
    query: String,
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct UserInputTask {
    name: String,
    _chain_index: Option<usize>,
    result: Option<UserInput>,
}

impl Default for UserInputTask {
    fn default() -> Self {
        UserInputTask {
            name: "UserInputTask".to_string(),
            _chain_index: None,
            result: None,
        }
    }
}

impl Task for UserInputTask {
    fn get_name(&self) -> &str {
        "UserInputTask"
    }

    fn set_chain_index(&mut self, idx: usize) {
        self._chain_index = Some(idx);
    }

    fn run(&mut self) -> Box<dyn ReturnValue> {
        let mut input = String::new();
        print!("Query: ");
        let _ = stdout().flush();
        stdin().read_line(&mut input).expect("Input error");

        if let Some('\n') = input.chars().next_back() {
            input.pop();
        }
        if let Some('\r') = input.chars().next_back() {
            input.pop();
        }

        Box::new(UserInput { query: input })
    }
}
