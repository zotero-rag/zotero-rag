//! Owned input requests that retain their reply channel until submission or dismissal.

use tokio::sync::oneshot;
use zqa::io::EngineEvent;

/// A pending prompt, deliberately neither cloneable nor debug-printable.
pub(crate) struct PromptRequest {
    message: String,
    input: Input,
}

enum Input {
    Choices {
        options: Vec<String>,
        selected: usize,
        reply: Reply,
    },
    Text {
        secret: bool,
        reply: oneshot::Sender<String>,
    },
}

enum Reply {
    Confirm(oneshot::Sender<bool>),
    Choose(oneshot::Sender<usize>),
}

impl PromptRequest {
    /// Take ownership of an input event without answering it.
    ///
    /// # Arguments
    ///
    /// * `event` - An engine input request, including its reply sender.
    ///
    /// # Returns
    ///
    /// A pending prompt with a valid default selection. Out-of-range defaults become zero.
    ///
    /// # Errors
    ///
    /// * Rejects non-input events and empty choices, dropping their payloads and senders.
    pub(crate) fn from_event(event: EngineEvent) -> Result<Self, &'static str> {
        let (message, input) = match event {
            EngineEvent::Confirm {
                message,
                default,
                reply,
            } => (
                message,
                Input::Choices {
                    options: vec!["Yes".into(), "No".into()],
                    selected: usize::from(!default),
                    reply: Reply::Confirm(reply),
                },
            ),
            EngineEvent::Choose {
                message,
                options,
                default,
                reply,
            } => {
                if options.is_empty() {
                    return Err("A choice requires at least one option");
                }

                let selected = if default < options.len() { default } else { 0 };
                (
                    message,
                    Input::Choices {
                        options,
                        selected,
                        reply: Reply::Choose(reply),
                    },
                )
            }
            EngineEvent::Line { message, reply } => (
                message.unwrap_or_else(|| "Your response".into()),
                Input::Text {
                    secret: false,
                    reply,
                },
            ),
            EngineEvent::Secret { message, reply } => (
                message,
                Input::Text {
                    secret: true,
                    reply,
                },
            ),
            _ => return Err("Event does not request input"),
        };

        Ok(Self { message, input })
    }

    /// Return the prompt label, using "Your response" for an absent line label.
    pub(crate) fn message(&self) -> &str {
        &self.message
    }

    /// Return the options and zero-based selection, or `None` for text input.
    pub(crate) fn choices(&self) -> Option<(&[String], usize)> {
        match &self.input {
            Input::Choices {
                options, selected, ..
            } => Some((options, *selected)),
            Input::Text { .. } => None,
        }
    }

    /// Update a choice without submitting it; invalid indices and text prompts are ignored.
    ///
    /// # Arguments
    ///
    /// * `index` - Zero-based index of the option to select.
    pub(crate) fn select(&mut self, index: usize) {
        if let Input::Choices {
            options, selected, ..
        } = &mut self.input
            && index < options.len()
        {
            *selected = index;
        }
    }

    /// Return whether the UI must conceal entered text.
    pub(crate) fn is_secret(&self) -> bool {
        matches!(self.input, Input::Text { secret: true, .. })
    }

    /// Return whether submission uses entered text rather than a selection.
    pub(crate) fn is_text(&self) -> bool {
        matches!(self.input, Input::Text { .. })
    }

    /// Return whether the engine has closed or dropped the reply receiver.
    pub(crate) fn is_closed(&self) -> bool {
        match &self.input {
            Input::Choices {
                reply: Reply::Confirm(reply),
                ..
            } => reply.is_closed(),
            Input::Choices {
                reply: Reply::Choose(reply),
                ..
            } => reply.is_closed(),
            Input::Text { reply, .. } => reply.is_closed(),
        }
    }

    /// Consume the prompt and submit its answer without logging it.
    ///
    /// A cancelled engine may have dropped the receiver; send failures are ignored.
    ///
    /// # Arguments
    ///
    /// * `text` - Unmodified text for line or secret input; ignored for choice prompts.
    pub(crate) fn respond(self, text: String) {
        match self.input {
            Input::Choices {
                selected,
                reply: Reply::Confirm(reply),
                ..
            } => {
                let _ = reply.send(selected == 0);
            }
            Input::Choices {
                selected,
                reply: Reply::Choose(reply),
                ..
            } => {
                let _ = reply.send(selected);
            }
            Input::Text { reply, .. } => {
                let _ = reply.send(text);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use tokio::sync::oneshot::error::TryRecvError;
    use zqa_macros::test_eq;

    use super::*;

    #[test]
    fn confirm_defaults_and_overrides() {
        for (default, selection, expected) in [
            (true, None, true),
            (false, None, false),
            (true, Some(1), false),
            (false, Some(0), true),
        ] {
            let (reply, mut answer) = oneshot::channel();
            let mut prompt = PromptRequest::from_event(EngineEvent::Confirm {
                message: "Continue?".into(),
                default,
                reply,
            })
            .unwrap();
            test_eq!(prompt.message(), "Continue?");
            let options = ["Yes".into(), "No".into()];
            test_eq!(
                prompt.choices(),
                Some((options.as_slice(), usize::from(!default)))
            );
            test_eq!(prompt.is_text(), false);
            test_eq!(prompt.is_secret(), false);
            test_eq!(prompt.is_closed(), false);

            if let Some(index) = selection {
                prompt.select(index);
            }

            test_eq!(answer.try_recv(), Err(TryRecvError::Empty));
            prompt.respond("ignored".into());
            test_eq!(answer.try_recv().unwrap(), expected);
        }
    }

    #[test]
    fn choices_are_zero_based_and_invalid_indices_are_ignored() {
        for (default, expected) in [(0, 0), (1, 1), (2, 0), (usize::MAX, 0)] {
            let (reply, mut answer) = oneshot::channel();
            let mut prompt = PromptRequest::from_event(EngineEvent::Choose {
                message: "Pick".into(),
                options: vec!["First".into(), "Second".into()],
                default,
                reply,
            })
            .unwrap();
            prompt.select(2);
            prompt.select(usize::MAX);
            test_eq!(prompt.choices().unwrap().1, expected);
            test_eq!(answer.try_recv(), Err(TryRecvError::Empty));
            prompt.respond(String::new());
            test_eq!(answer.try_recv().unwrap(), expected);
        }

        let (reply, mut answer) = oneshot::channel();
        let mut prompt = PromptRequest::from_event(EngineEvent::Choose {
            message: "Pick".into(),
            options: vec!["First".into(), "Second".into()],
            default: 0,
            reply,
        })
        .unwrap();
        prompt.select(1);
        prompt.select(2);
        prompt.respond(String::new());
        test_eq!(answer.try_recv().unwrap(), 1);
    }

    #[test]
    fn rejects_empty_choices_and_non_input() {
        let (reply, mut answer) = oneshot::channel();
        let result = PromptRequest::from_event(EngineEvent::Choose {
            message: "Pick".into(),
            options: Vec::new(),
            default: 0,
            reply,
        });
        test_eq!(result.is_err(), true);
        test_eq!(answer.try_recv(), Err(TryRecvError::Closed));
        test_eq!(
            PromptRequest::from_event(EngineEvent::Text {
                message: "Output".into()
            })
            .is_err(),
            true
        );
    }

    #[test]
    fn text_and_secrets_are_sent_unchanged() {
        for secret in [false, true] {
            for text in ["", "  spaces\tand\nnewlines\r\n"] {
                let (reply, mut answer) = oneshot::channel();
                let event = if secret {
                    EngineEvent::Secret {
                        message: "Password".into(),
                        reply,
                    }
                } else {
                    EngineEvent::Line {
                        message: None,
                        reply,
                    }
                };
                let mut prompt = PromptRequest::from_event(event).unwrap();
                test_eq!(
                    prompt.message(),
                    if secret { "Password" } else { "Your response" }
                );
                test_eq!(prompt.is_secret(), secret);
                test_eq!(prompt.is_text(), true);
                test_eq!(prompt.choices(), None);
                prompt.select(usize::MAX);
                test_eq!(answer.try_recv(), Err(TryRecvError::Empty));
                prompt.respond(text.into());
                test_eq!(answer.try_recv().unwrap(), text);
            }
        }
    }

    #[test]
    fn dropping_prompt_disconnects_receiver() {
        let (reply, mut answer) = oneshot::channel();
        let prompt = PromptRequest::from_event(EngineEvent::Line {
            message: Some("Question".into()),
            reply,
        })
        .unwrap();
        test_eq!(prompt.message(), "Question");
        test_eq!(prompt.is_closed(), false);
        drop(prompt);
        test_eq!(answer.try_recv(), Err(TryRecvError::Closed));
    }

    #[test]
    fn cancelled_receivers_allow_submission() {
        let (reply, answer) = oneshot::channel();
        let confirm = EngineEvent::Confirm {
            message: String::new(),
            default: true,
            reply,
        };
        drop(answer);
        let (reply, answer) = oneshot::channel();
        let choose = EngineEvent::Choose {
            message: String::new(),
            options: vec!["Only".into()],
            default: 0,
            reply,
        };
        drop(answer);
        let (reply, answer) = oneshot::channel();
        let line = EngineEvent::Line {
            message: None,
            reply,
        };
        drop(answer);
        let (reply, answer) = oneshot::channel();
        let secret = EngineEvent::Secret {
            message: String::new(),
            reply,
        };
        drop(answer);

        for event in [confirm, choose, line, secret] {
            let prompt = PromptRequest::from_event(event).unwrap();
            test_eq!(prompt.is_closed(), true);
            prompt.respond("discarded".into());
        }
    }
}
