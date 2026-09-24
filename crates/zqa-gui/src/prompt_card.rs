//! A non-blocking question card whose reply sender is owned until submission or dismissal.

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::{ActiveTheme as _, Disableable as _, Icon, IconName, h_flex, v_flex};
use gpui_kit::prelude::*;
use gpui_kit::{
    App, Context, Entity, EventEmitter, FocusHandle, Focusable, IntoElement, Render, Subscription,
    Window, div, px,
};

use crate::prompts::PromptRequest;

pub(crate) enum PromptEvent {
    Submitted,
    Cancelled,
}

/// One pending question. Text is kept in a separate input, not the chat composer or transcript.
/// Dropping the card also drops its input's undo history and any unanswered reply sender.
pub(crate) struct PromptCard {
    request: Option<PromptRequest>,
    input: Option<Entity<InputState>>,
    focus_handle: FocusHandle,
    _subscription: Option<Subscription>,
}

impl EventEmitter<PromptEvent> for PromptCard {}

impl PromptCard {
    pub(crate) fn new(request: PromptRequest, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let input = request.is_text().then(|| {
            cx.new(|cx| {
                InputState::new(window, cx)
                    .placeholder(if request.is_secret() {
                        "Secret..."
                    } else {
                        "Reply..."
                    })
                    .masked(request.is_secret())
            })
        });
        let subscription = input.as_ref().map(|input| {
            cx.subscribe(input, |this, _, event: &InputEvent, cx| match event {
                InputEvent::PressEnter { .. } => this.submit(cx),
                InputEvent::Change => cx.notify(),
                _ => {}
            })
        });

        Self {
            request: Some(request),
            input,
            focus_handle: cx.focus_handle(),
            _subscription: subscription,
        }
    }

    fn can_submit(&self, cx: &App) -> bool {
        self.request
            .as_ref()
            .is_some_and(|request| !request.is_closed())
            && self
                .input
                .as_ref()
                .is_none_or(|input| !input.read(cx).value().is_empty())
    }

    /// Consume the reply once. A cancelled engine may already have dropped its receiver.
    fn submit(&mut self, cx: &mut Context<Self>) {
        if !self.can_submit(cx) {
            return;
        }

        let text = self
            .input
            .as_ref()
            .map(|input| input.read(cx).value().to_string())
            .unwrap_or_default();
        if let Some(request) = self.request.take() {
            request.respond(text);
        }

        self.input = None;
        self._subscription = None;
        cx.emit(PromptEvent::Submitted);
        cx.notify();
    }

    /// Ask the owner to cancel the command before dropping unanswered reply senders.
    fn cancel(&mut self, cx: &mut Context<Self>) {
        cx.emit(PromptEvent::Cancelled);
    }
}

impl Focusable for PromptCard {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        self.input.as_ref().map_or_else(
            || self.focus_handle.clone(),
            |input| input.read(cx).focus_handle(cx),
        )
    }
}

impl Render for PromptCard {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let Some(request) = &self.request else {
            return div().into_any_element();
        };
        let message = request.message().to_owned();
        let mut content = v_flex().gap_2();

        if let Some((options, selected)) = request.choices() {
            for (index, label) in options.iter().enumerate() {
                let selected = index == selected;
                content = content.child(
                    Button::new(("prompt-option", index))
                        .ghost()
                        .accessibility_label(label.clone())
                        .w_full()
                        .h_auto()
                        .justify_start()
                        .items_center()
                        .gap_2()
                        .p_2()
                        .rounded_lg()
                        .when(selected, |row| row.secondary())
                        .child(
                            div()
                                .size(px(24.))
                                .text_size(px(12.))
                                .line_height(px(16.))
                                .flex_shrink_0()
                                .flex()
                                .items_center()
                                .justify_center()
                                .rounded_full()
                                .border_1()
                                .border_color(cx.theme().border)
                                .text_color(cx.theme().muted_foreground)
                                .when(selected, |badge| {
                                    badge
                                        .border_color(cx.theme().primary)
                                        .text_color(cx.theme().primary)
                                })
                                .child((index + 1).to_string()),
                        )
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .text_size(px(13.))
                                .line_height(px(18.))
                                .whitespace_normal()
                                .child(label.trim_end().to_owned()),
                        )
                        .when(selected, |row| {
                            row.child(
                                Icon::new(IconName::Check)
                                    .size_4()
                                    .text_color(cx.theme().primary),
                            )
                        })
                        .on_click(cx.listener(move |this, _, window, cx| {
                            if let Some(request) = &mut this.request {
                                request.select(index);
                            }

                            window.focus(&this.focus_handle, cx);
                            cx.notify();
                        })),
                );
            }
        }

        if let Some(input) = &self.input {
            content = content.child(Input::new(input).h(px(44.)));
        }

        let can_submit = self.can_submit(cx);
        v_flex()
            .id("prompt-card")
            .track_focus(&self.focus_handle)
            .w_full()
            .gap_4()
            .p_4()
            .rounded_2xl()
            .border_1()
            .border_color(cx.theme().border)
            .bg(cx.theme().popover)
            .on_key_down(cx.listener(|this, event: &gpui_kit::KeyDownEvent, _, cx| {
                if this.input.is_some() {
                    return;
                }

                if event.keystroke.key == "enter" {
                    this.submit(cx);
                    cx.stop_propagation();
                    return;
                }

                if let Some(request) = &mut this.request
                    && let Some((options, selected)) = request.choices()
                {
                    let next = match event.keystroke.key.as_str() {
                        "up" => Some(selected.saturating_sub(1)),
                        "down" => Some((selected + 1).min(options.len() - 1)),
                        key => key
                            .parse::<usize>()
                            .ok()
                            .and_then(|number| number.checked_sub(1)),
                    };

                    if let Some(index) = next {
                        request.select(index);
                        cx.stop_propagation();
                        cx.notify();
                    }
                }
            }))
            .child(
                h_flex()
                    .justify_between()
                    .text_color(cx.theme().muted_foreground)
                    .child(
                        h_flex()
                            .gap_2()
                            .child(Icon::new(IconName::Info).size_4())
                            .child("Question"),
                    )
                    .child(
                        Button::new("dismiss-prompt")
                            .ghost()
                            .icon(IconName::Close)
                            .tooltip("Cancel this command")
                            .on_click(cx.listener(|this, _, _, cx| this.cancel(cx))),
                    ),
            )
            .child(div().text_size(px(15.)).child(message))
            .child(
                div()
                    .id("prompt-options")
                    .max_h(px(240.))
                    .overflow_y_scroll()
                    .child(content),
            )
            .child(
                h_flex()
                    .justify_end()
                    .gap_2()
                    .child(
                        Button::new("skip-prompt")
                            .outline()
                            .rounded(px(999.))
                            .label("Skip")
                            .tooltip("Cancel this command")
                            .on_click(cx.listener(|this, _, _, cx| this.cancel(cx))),
                    )
                    .child(
                        Button::new("send-prompt")
                            .primary()
                            .rounded(px(999.))
                            .label("Send")
                            .disabled(!can_submit)
                            .on_click(cx.listener(|this, _, _, cx| this.submit(cx))),
                    ),
            )
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use gpui_kit::TestAppContext;
    use gpui_kit::test::TestWindowExt;
    use tokio::sync::oneshot;
    use zqa::io::EngineEvent;
    use zqa_macros::test_eq;

    use super::*;

    #[gpui_kit::test]
    fn choice_click_then_enter_sends_selected_index(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
        let (reply, mut answer) = oneshot::channel();
        let request = PromptRequest::from_event(EngineEvent::Choose {
            message: "How detailed?".into(),
            options: vec!["Brief".into(), "Detailed".into()],
            default: 0,
            reply,
        })
        .unwrap();
        let (_, cx) = cx.add_window_view(|window, cx| PromptCard::new(request, window, cx));

        cx.update(|window, cx| {
            window.click(("prompt-option", 1_usize), cx);
            window.press("enter", cx);
        });

        test_eq!(answer.try_recv().unwrap(), 1);
    }

    #[gpui_kit::test]
    fn typing_and_sending_preserves_text(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
        let (reply, mut answer) = oneshot::channel();
        let request = PromptRequest::from_event(EngineEvent::Line {
            message: Some("What next?".into()),
            reply,
        })
        .unwrap();
        let (card, cx) = cx.add_window_view(|window, cx| PromptCard::new(request, window, cx));

        cx.update(|window, cx| {
            window.focus(&card.read(cx).focus_handle(cx), cx);
            window.input("A typed response", cx);
            window.click("send-prompt", cx);
        });

        test_eq!(answer.try_recv().unwrap(), "A typed response");
    }

    #[gpui_kit::test]
    fn secret_enter_submits_without_retaining_editor(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
        let (reply, mut answer) = oneshot::channel();
        let request = PromptRequest::from_event(EngineEvent::Secret {
            message: "Test secret".into(),
            reply,
        })
        .unwrap();
        let (card, cx) = cx.add_window_view(|window, cx| PromptCard::new(request, window, cx));

        cx.update(|window, cx| {
            let input = card.read(cx).input.as_ref().unwrap();
            assert!(input.read(cx).presentation().is_masked());
            window.focus(&card.read(cx).focus_handle(cx), cx);
            window.input("synthetic-secret", cx);
            window.press("enter", cx);
        });

        test_eq!(answer.try_recv().unwrap(), "synthetic-secret");
        card.read_with(cx, |card, _| assert!(card.input.is_none()));
    }

    #[gpui_kit::test]
    fn skip_requests_cancellation_without_answering(cx: &mut TestAppContext) {
        cx.update(gpui_kit::init);
        let (reply, mut answer) = oneshot::channel();
        let request = PromptRequest::from_event(EngineEvent::Confirm {
            message: "Continue?".into(),
            default: true,
            reply,
        })
        .unwrap();
        let (card, cx) = cx.add_window_view(|window, cx| PromptCard::new(request, window, cx));
        let cancelled = Rc::new(Cell::new(false));
        let _subscription = cx.update(|_, cx| {
            let cancelled = cancelled.clone();
            cx.subscribe(&card, move |_, event, _| {
                cancelled.set(matches!(event, PromptEvent::Cancelled));
            })
        });

        cx.update(|window, cx| window.click("skip-prompt", cx));
        cx.run_until_parked();
        assert!(cancelled.get());
        assert!(matches!(
            answer.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
    }
}
