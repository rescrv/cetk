use std::ops::Deref;

use utf8path::Path;

////////////////////////////////////////////// PathExt /////////////////////////////////////////////

trait PathExt {
    fn context_file(&self, context_index: u64) -> Path;
}

impl PathExt for Path<'_> {
    fn context_file(&self, context_index: u64) -> Path {
        self.join(format!("{context_index}.jsonl"))
    }
}

/////////////////////////////////////////////// Event //////////////////////////////////////////////

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Probe(String),
}

////////////////////////////////////////// ContextManager //////////////////////////////////////////

pub struct ContextManager {
    root: Path<'static>,
    next_context: u64,
    current: Context,
}

impl ContextManager {
    pub fn new(root: Path) -> Result<Self, claudius::Error> {
        let root = root.into_owned();
        let mut next_context = 0;
        while root.context_file(next_context).exists() {
            next_context += 1;
        }
        // next_context points to the first uninitialized context
        let current = Context::default();
        let mut this = Self {
            root,
            next_context,
            current,
        };
        if next_context > 0 {
            let events = this.load_context(next_context - 1)?;
            this.current = Context::from(events);
        }
        Ok(this)
    }

    pub fn contexts(
        &self,
    ) -> Result<
        impl DoubleEndedIterator<Item = Result<Context, claudius::Error>> + '_,
        claudius::Error,
    > {
        struct Contexts<'a> {
            manager: &'a ContextManager,
            index_start: u64,
            index_limit: u64,
        }
        impl Iterator for Contexts<'_> {
            type Item = Result<Context, claudius::Error>;

            fn next(&mut self) -> Option<Self::Item> {
                if self.index_start < self.index_limit {
                    self.index_limit -= 1;
                    let events = match self.manager.load_context(self.index_limit) {
                        Ok(events) => events,
                        Err(err) => return Some(Err(err)),
                    };
                    Some(Ok(Context::from(events)))
                } else {
                    None
                }
            }
        }
        impl DoubleEndedIterator for Contexts<'_> {
            fn next_back(&mut self) -> Option<Self::Item> {
                if self.index_start < self.index_limit {
                    let events = match self.manager.load_context(self.index_start) {
                        Ok(events) => events,
                        Err(err) => return Some(Err(err)),
                    };
                    self.index_start += 1;
                    Some(Ok(Context::from(events)))
                } else {
                    None
                }
            }
        }
        Ok(Contexts {
            manager: self,
            index_start: 0,
            index_limit: self.next_context.saturating_sub(1),
        })
    }

    pub fn current(&self) -> &Context {
        todo!();
    }

    pub fn unread(&mut self) -> impl DoubleEndedIterator<Item = UnreadMessageRef> {
        std::iter::empty()
    }

    pub fn compact<I: Iterator<Item = Message>, O: Iterator<Item = Message>>(
        &mut self,
        f: impl FnOnce(I) -> O,
    ) {
        todo!();
    }

    pub fn update(&mut self) -> Update<'_> {
        todo!();
    }

    fn load_context(&self, context_index: u64) -> Result<Vec<Event>, claudius::Error> {
        let mut events = vec![];
        let path = self.root.context_file(context_index);
        let jsonl = std::fs::read_to_string(&path)?;
        for json in jsonl.split_terminator('\n') {
            let event: Event = serde_json::from_str(json)?;
            events.push(event);
        }
        Ok(events)
    }
}

////////////////////////////////////////////// Context /////////////////////////////////////////////

#[derive(Clone, Debug, Default)]
pub struct Context {
    messages: Vec<Message>,
}

impl Context {
    pub fn messages(&self) -> impl DoubleEndedIterator<Item = Message> {
        std::iter::empty()
    }
}

impl From<Vec<Event>> for Context {
    fn from(events: Vec<Event>) -> Self {
        todo!();
    }
}

////////////////////////////////////////////// Update //////////////////////////////////////////////

pub struct Update<'a> {
    manager: &'a mut ContextManager,
}

impl Update<'_> {
    pub fn save(self) -> Result<(), claudius::Error> {
        todo!();
    }
}

////////////////////////////////////////////// Message /////////////////////////////////////////////

#[derive(Clone, Debug, Default)]
pub struct Message;

impl Message {}

///////////////////////////////////////// UnreadMessageRef /////////////////////////////////////////

pub struct UnreadMessageRef;

/////////////////////////////////////////// UnreadMessage //////////////////////////////////////////

pub struct UnreadMessage;

impl UnreadMessage {
    pub fn mark_read(&mut self) {}

    pub fn mark_unread(&mut self) {}
}

impl Deref for UnreadMessage {
    type Target = Message;

    fn deref(&self) -> &Message {
        todo!();
    }
}

/////////////////////////////////////////////// tests //////////////////////////////////////////////

#[cfg(test)]
mod tests {}
