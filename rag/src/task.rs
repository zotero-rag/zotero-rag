use downcast::{downcast, Any};
use dyn_clone::{clone_trait_object, DynClone};
use std::{collections::HashMap, fmt, marker::PhantomData};

pub trait Task: TaskClone {
    fn run(&mut self) -> Box<dyn ReturnValue>;
    fn set_chain_index(&mut self, idx: usize);
    fn get_name(&self) -> &str;
}

// This allows us to use clone with Box<dyn Task>
pub trait TaskClone {
    fn clone_box(&self) -> Box<dyn Task>;
}

impl<T> TaskClone for T
where
    T: 'static + Task + Clone,
{
    fn clone_box(&self) -> Box<dyn Task> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Task> {
    fn clone(&self) -> Box<dyn Task> {
        self.clone_box()
    }
}

// from https://users.rust-lang.org/t/how-to-clone-a-struct-with-option-box-dyn-any/96008/2
pub trait ReturnValue: DynClone + Any {}
clone_trait_object!(ReturnValue);
downcast!(dyn ReturnValue);

impl<T: Clone + Any> ReturnValue for T {}

impl fmt::Debug for dyn ReturnValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReturnValue").finish_non_exhaustive()
    }
}

// If you only need read-only access to the value anyways, then the alternative of simply using Option<Rc<dyn Any>> could be better, i.e. simpler to code and also more efficient, as cloning an Rc will be cheaper.

pub struct Chain<T> {
    pub name: String,
    pub tasks: Vec<Box<dyn Task>>,
    pub task_outputs: HashMap<String, Box<dyn ReturnValue>>,

    pub return_type: PhantomData<T>,
}

impl<T> Default for Chain<T> {
    fn default() -> Self {
        Chain {
            name: "DefaultChain".to_string(),
            tasks: Vec::new(),
            task_outputs: HashMap::new(),
            return_type: PhantomData,
        }
    }
}

impl<T> Chain<T> {
    pub fn new(name: &str, tasks: Vec<Box<dyn Task>>) -> Self {
        let mut chain = Chain::<T> {
            name: name.to_string(),
            tasks,
            task_outputs: HashMap::new(),
            return_type: PhantomData,
        };

        for (i, task) in chain.tasks.iter_mut().enumerate() {
            task.set_chain_index(i);
        }

        chain
    }

    pub fn run(&mut self) -> Box<dyn ReturnValue> {
        let mut result: Box<dyn ReturnValue> = Box::new(());

        for task in self.tasks.iter_mut() {
            let task_name = task.get_name().to_string();

            result = task.run();
            self.task_outputs
                .insert(task_name, Box::new(result.clone()));
        }

        result
    }

    pub fn add_task<U>(&mut self, other: Box<dyn Task>) -> Chain<U> {
        let mut new_task = other;
        new_task.set_chain_index(self.tasks.len());
        self.tasks.push(new_task);

        Chain::<U> {
            name: self.name.clone(),
            tasks: self.tasks.clone(),
            task_outputs: self.task_outputs.clone(),
            return_type: PhantomData,
        }
    }
}
