#![doc = include_str!("../README.md")]

use std::fmt::Debug;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;

use claudius::MessageParam;
use one_two_eight::generate_id;
use utf8path::Path;

////////////////////////////////////////////// PathExt /////////////////////////////////////////////

/// An internal trait for exposing parameterized paths and path constants.
trait PathExt {
    fn context_file(&self, context_index: u64) -> Path;
}

impl PathExt for Path<'_> {
    /// Given a path/root, compute the path to a JSONL file for the `context_index`'th context.
    fn context_file(&self, context_index: u64) -> Path {
        self.join("contexts").join(format!("{context_index}.jsonl"))
    }
}

///////////////////////////////////////// generate_id_serde ////////////////////////////////////////

/// Generate the serde Deserialize/Serialize routines for a one_two_eight ID.
macro_rules! generate_id_serde {
    ($name:ident, $visitor:ident) => {
        impl serde::Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                let s = self.to_string();
                serializer.serialize_str(&s)
            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<$name, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                deserializer.deserialize_str($visitor)
            }
        }

        struct $visitor;

        impl<'de> serde::de::Visitor<'de> for $visitor {
            type Value = $name;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("an ID")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                $name::from_human_readable(value).ok_or_else(|| E::custom("not a valid tx:UUID"))
            }
        }
    };
}

/////////////////////////////////////////// TransactionID //////////////////////////////////////////

generate_id!(TransactionID, "tx:");
generate_id_serde!(TransactionID, TransactionIDVisitor);

////////////////////////////////////////////// MountID /////////////////////////////////////////////

generate_id!(MountID, "mount:");
generate_id_serde!(MountID, MountIDVisitor);

//////////////////////////////////////////// Transaction ///////////////////////////////////////////

/// A Transaction contains a transaction ID, some application data (usually a pointer to the
/// application state elsewhere, but it could be a state transition for rolling up in a state
/// machine), some messages to append to the conversation, and some filesystem writes.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct Transaction<D: Clone + Debug + Default> {
    pub txid: TransactionID,
    pub data: D,
    pub msgs: Vec<MessageParam>,
    pub writes: Vec<FileWrite>,
}

impl<D: Clone + Debug + Default> Transaction<D> {
    /// Iterate over the messages of this transaction.
    pub fn messages(&self) -> impl DoubleEndedIterator<Item = MessageParam> + '_ {
        self.msgs.iter().cloned()
    }
}

///////////////////////////////////////////// FileWrite ////////////////////////////////////////////

/// Write the complete contents of data to the file at path on mount.
///
/// We assume files in the virtual filesystem should be small.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct FileWrite {
    mount: MountID,
    path: String,
    data: String,
}

////////////////////////////////////////// ContextManager //////////////////////////////////////////

/// The ContextManager gives a single logical entity a linear history.  Instantiate it with a path
/// to the filesystem as the root.  Contexts will be written under the directory returned by
/// [PathExt::context_file] in the order in which they are written.
pub struct ContextManager<
    D: Clone + Debug + Default + for<'a> serde::Deserialize<'a> + serde::Serialize,
> {
    root: Path<'static>,
    curr_context: u64,
    next_context: u64,
    current: Context<D>,
}

impl<D: Clone + Debug + Default + for<'a> serde::Deserialize<'a> + serde::Serialize>
    ContextManager<D>
{
    /// Create a new context manager with the provided root.
    pub fn new(root: Path) -> Result<Self, claudius::Error> {
        let root = root.into_owned();
        let mut next_context = 0;
        while root.context_file(next_context).exists() {
            next_context += 1;
        }
        if next_context == 0 {
            next_context += 1;
        }
        // next_context points to the first uninitialized context
        let curr_context = next_context.saturating_sub(1);
        let current = Context::default();
        let curr_context_exists = root.context_file(curr_context).exists();
        let mut this = Self {
            root,
            curr_context,
            next_context,
            current,
        };
        if curr_context_exists {
            this.current = this.load_context(curr_context)?
        };
        Ok(this)
    }

    /// Iterate over all contexts.  This returns a double ended iterator where forward iteration
    /// goes forward in time and reverse iteration starts at the most recent context and goes
    /// backward in time.
    pub fn contexts(
        &self,
    ) -> Result<
        impl DoubleEndedIterator<Item = Result<Context<D>, claudius::Error>> + '_,
        claudius::Error,
    > {
        struct Contexts<
            'a,
            D: Clone + Debug + Default + for<'b> serde::Deserialize<'b> + serde::Serialize,
        > {
            manager: &'a ContextManager<D>,
            start: u64,
            limit: u64,
        }
        impl<
                'a,
                D: Clone + Debug + Default + for<'b> serde::Deserialize<'b> + serde::Serialize,
            > Iterator for Contexts<'a, D>
        {
            type Item = Result<Context<D>, claudius::Error>;

            fn next(&mut self) -> Option<Self::Item> {
                if self.start < self.limit {
                    let context_index = self.start;
                    self.start += 1;
                    Some(self.manager.load_context(context_index))
                } else {
                    None
                }
            }
        }
        impl<
                'a,
                D: Clone + Debug + Default + for<'b> serde::Deserialize<'b> + serde::Serialize,
            > DoubleEndedIterator for Contexts<'a, D>
        {
            fn next_back(&mut self) -> Option<Self::Item> {
                if self.start < self.limit {
                    self.limit -= 1;
                    Some(self.manager.load_context(self.limit))
                } else {
                    None
                }
            }
        }
        Ok(Contexts {
            manager: self,
            start: 0,
            limit: self.next_context,
        })
    }

    /// Iterate the transactions of all contexts in the order in which they were applied.
    pub fn transactions(
        &self,
    ) -> Result<impl Iterator<Item = Result<Transaction<D>, claudius::Error>> + '_, claudius::Error>
    {
        Ok(self.contexts()?.flat_map(|x| match x {
            Ok(context) => context
                .transactions()
                .map(Ok)
                .collect::<Vec<_>>()
                .into_iter(),
            Err(err) => vec![Err(err)].into_iter(),
        }))
    }

    /// Iterate the messages of all contexts in the order in which they were applied.
    pub fn messages(
        &self,
    ) -> Result<impl Iterator<Item = Result<MessageParam, claudius::Error>> + '_, claudius::Error>
    {
        Ok(self.transactions()?.flat_map(|x| match x {
            Ok(tx) => tx.messages().map(Ok).collect::<Vec<_>>().into_iter(),
            Err(err) => vec![Err(err)].into_iter(),
        }))
    }

    // Transact on the latest context.  This will acquire a lock on the context across all
    // processes, and then synchronously execute the transaction against the context.
    pub fn transact<E: From<claudius::Error>>(
        &mut self,
        transact: impl FnOnce(&Context<D>) -> Result<Transaction<D>, E>,
    ) -> Result<(), E> {
        let mut output = Self::lock(self.root.context_file(self.curr_context))?;
        if self.root.context_file(self.next_context).exists() {
            todo!();
        }
        let xact = transact(&self.current)?;
        let mut json = serde_json::to_string(&xact).map_err(|err| {
            claudius::Error::serialization("could not serialize transaction", Some(Box::new(err)))
        })?;
        json.push('\n');
        output
            .write_all(json.as_bytes())
            .map_err(|err| claudius::Error::io("could not write transaction", err))?;
        self.current.transactions.push(xact);
        Ok(())
    }

    /// Take the latest context and compact it in an arbitrary way according to the `doit`
    /// function.
    pub fn compact<O: Iterator<Item = Transaction<D>>>(
        &mut self,
        doit: impl FnOnce(&mut dyn Iterator<Item = Transaction<D>>) -> O,
    ) -> Result<(), claudius::Error> {
        // NOTE(rescrv): used as a lock.
        let _current = Self::lock(self.root.context_file(self.curr_context))?;
        if self.root.context_file(self.next_context).exists() {
            todo!();
        }
        let mut output = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .mode(0o600)
            .open(self.root.context_file(self.next_context))?;
        let mut buffer = String::new();
        for xact in doit(&mut self.current.transactions()) {
            let json = serde_json::to_string(&xact).map_err(|err| {
                claudius::Error::serialization(
                    "could not serialize transaction",
                    Some(Box::new(err)),
                )
            })?;
            buffer += &json;
            buffer.push('\n');
        }
        output
            .write_all(buffer.as_bytes())
            .map_err(|err| claudius::Error::io("could not write transaction", err))?;
        if self.curr_context == self.next_context {
            self.next_context += 1;
        } else {
            self.curr_context += 1;
            self.next_context += 1;
        }
        self.current = self.load_context(self.curr_context)?;
        Ok(())
    }

    /// Load the `content_index`'th context from durable storage.
    fn load_context(&self, context_index: u64) -> Result<Context<D>, claudius::Error> {
        let mut transactions = vec![];
        let path = self.root.context_file(context_index);
        let jsonl = std::fs::read_to_string(&path)?;
        for json in jsonl.split_terminator('\n') {
            let transaction: Transaction<D> = serde_json::from_str(json)?;
            transactions.push(transaction);
        }
        Ok(Context::from(transactions))
    }

    fn lock(path: Path) -> Result<File, claudius::Error> {
        let what = libc::F_SETLKW;
        // Open the file, creating it if it doesn't exist.
        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .mode(0o600)
            .open(path)?;
        // NOTE(rescrv): l_type,l_whence is 16 bits on some platforms and 32 bits on others.
        // The annotations here are for cross-platform compatibility.
        #[allow(clippy::useless_conversion)]
        #[allow(clippy::unnecessary_cast)]
        let flock = libc::flock {
            l_type: libc::F_WRLCK as i16,
            l_whence: libc::SEEK_SET as i16,
            l_start: 0,
            l_len: 0,
            l_pid: 0,
            #[cfg(target_os = "freebsd")]
            l_sysid: 0,
        };
        loop {
            if unsafe { libc::fcntl(file.as_raw_fd(), what, &flock) < 0 } {
                let err = std::io::Error::last_os_error();
                if let Some(libc::EINTR) = err.raw_os_error() {
                    continue;
                }
                break Err(err.into());
            } else {
                break Ok(file);
            }
        }
    }
}

////////////////////////////////////////////// Context /////////////////////////////////////////////

/// An individual context.  Defined as a sequence of transactions, it can also be seen as a
/// sequence of messages, or a projection of the filesystem.
#[derive(Clone, Debug, Default)]
pub struct Context<D: Clone + Debug + Default + for<'a> serde::Deserialize<'a> + serde::Serialize> {
    transactions: Vec<Transaction<D>>,
}

impl<D: Clone + Debug + Default + for<'a> serde::Deserialize<'a> + serde::Serialize> Context<D> {
    /// Iterate over the transactions of this context.
    pub fn transactions(&self) -> impl DoubleEndedIterator<Item = Transaction<D>> {
        self.transactions.iter().cloned()
    }

    /// Iterate over the messages of this context.
    pub fn messages(&self) -> impl DoubleEndedIterator<Item = MessageParam> {
        self.transactions.iter().flat_map(|tx| tx.messages())
    }
}

impl<D: Clone + Debug + Default + for<'a> serde::Deserialize<'a> + serde::Serialize>
    From<Vec<Transaction<D>>> for Context<D>
{
    /// Create a context from a vector of transactions.
    fn from(transactions: Vec<Transaction<D>>) -> Self {
        Self { transactions }
    }
}

/////////////////////////////////////////////// tests //////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transaction_id_generation() {
        let id1 = TransactionID::generate().unwrap();
        let id2 = TransactionID::generate().unwrap();

        // IDs should be unique
        assert_ne!(id1, id2);

        // Should have correct prefix
        let s1 = id1.to_string();
        assert!(s1.starts_with("tx:"));

        // Should be parseable
        let parsed = TransactionID::from_human_readable(&s1).unwrap();
        assert_eq!(id1, parsed);
    }

    #[test]
    fn mount_id_generation() {
        let id1 = MountID::generate().unwrap();
        let id2 = MountID::generate().unwrap();

        // IDs should be unique
        assert_ne!(id1, id2);

        // Should have correct prefix
        let s1 = id1.to_string();
        assert!(s1.starts_with("mount:"));

        // Should be parseable
        let parsed = MountID::from_human_readable(&s1).unwrap();
        assert_eq!(id1, parsed);
    }

    #[test]
    fn id_serialization() {
        let txid = TransactionID::generate().unwrap();
        let json = serde_json::to_string(&txid).unwrap();
        let deserialized: TransactionID = serde_json::from_str(&json).unwrap();
        assert_eq!(txid, deserialized);

        let mount_id = MountID::generate().unwrap();
        let json = serde_json::to_string(&mount_id).unwrap();
        let deserialized: MountID = serde_json::from_str(&json).unwrap();
        assert_eq!(mount_id, deserialized);
    }

    #[test]
    fn invalid_id_parsing() {
        assert!(TransactionID::from_human_readable("invalid").is_none());
        assert!(TransactionID::from_human_readable("mount:123").is_none());
        assert!(MountID::from_human_readable("tx:123").is_none());
    }

    #[test]
    fn transaction_creation() {
        #[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize, PartialEq)]
        struct TestData {
            value: String,
        }

        let tx = Transaction {
            txid: TransactionID::generate().unwrap(),
            data: TestData {
                value: "test".to_string(),
            },
            msgs: vec![],
            writes: vec![],
        };

        assert_eq!(tx.data.value, "test");
        assert_eq!(tx.messages().count(), 0);
    }

    #[test]
    fn transaction_messages_iterator() {
        use claudius::{MessageParam, MessageRole};

        #[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
        struct TestData;

        let msg1 = MessageParam {
            role: MessageRole::User,
            content: "Hello".to_string().into(),
        };
        let msg2 = MessageParam {
            role: MessageRole::Assistant,
            content: "Hi there".to_string().into(),
        };

        let tx = Transaction {
            txid: TransactionID::generate().unwrap(),
            data: TestData,
            msgs: vec![msg1.clone(), msg2.clone()],
            writes: vec![],
        };

        let messages: Vec<_> = tx.messages().collect();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, msg1.content);
        assert_eq!(messages[1].content, msg2.content);

        // Test reverse iteration
        let rev_messages: Vec<_> = tx.messages().rev().collect();
        assert_eq!(rev_messages.len(), 2);
        assert_eq!(rev_messages[0].content, msg2.content);
        assert_eq!(rev_messages[1].content, msg1.content);
    }

    #[test]
    fn file_write() {
        let fw = FileWrite {
            mount: MountID::generate().unwrap(),
            path: "/test/file.txt".to_string(),
            data: "content".to_string(),
        };

        let json = serde_json::to_string(&fw).unwrap();
        let deserialized: FileWrite = serde_json::from_str(&json).unwrap();
        assert_eq!(fw.path, deserialized.path);
        assert_eq!(fw.data, deserialized.data);
    }

    #[test]
    fn context_from_transactions() {
        #[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
        struct TestData;

        let tx1 = Transaction {
            txid: TransactionID::generate().unwrap(),
            data: TestData,
            msgs: vec![],
            writes: vec![],
        };
        let tx2 = Transaction {
            txid: TransactionID::generate().unwrap(),
            data: TestData,
            msgs: vec![],
            writes: vec![],
        };

        let context = Context::from(vec![tx1.clone(), tx2.clone()]);
        let transactions: Vec<_> = context.transactions().collect();
        assert_eq!(transactions.len(), 2);
        assert_eq!(transactions[0].txid, tx1.txid);
        assert_eq!(transactions[1].txid, tx2.txid);
    }

    #[test]
    fn context_messages_flattening() {
        use claudius::{MessageParam, MessageRole};

        #[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
        struct TestData;

        let msg1 = MessageParam {
            role: MessageRole::User,
            content: "Message 1".to_string().into(),
        };
        let msg2 = MessageParam {
            role: MessageRole::Assistant,
            content: "Message 2".to_string().into(),
        };
        let msg3 = MessageParam {
            role: MessageRole::User,
            content: "Message 3".to_string().into(),
        };

        let tx1 = Transaction {
            txid: TransactionID::generate().unwrap(),
            data: TestData,
            msgs: vec![msg1.clone(), msg2.clone()],
            writes: vec![],
        };
        let tx2 = Transaction {
            txid: TransactionID::generate().unwrap(),
            data: TestData,
            msgs: vec![msg3.clone()],
            writes: vec![],
        };

        let context = Context::from(vec![tx1, tx2]);
        let messages: Vec<_> = context.messages().collect();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content, msg1.content);
        assert_eq!(messages[1].content, msg2.content);
        assert_eq!(messages[2].content, msg3.content);
    }

    #[test]
    fn path_ext_context_file() {
        let root = Path::new("/test/root");
        assert_eq!(root.context_file(0).as_str(), "/test/root/contexts/0.jsonl");
        assert_eq!(
            root.context_file(42).as_str(),
            "/test/root/contexts/42.jsonl"
        );
    }

    #[test]
    fn context_manager_new_empty_dir() {
        #[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
        struct TestData;

        let root = Path::from(".tests").join("context_manager_new_empty_dir");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let manager = ContextManager::<TestData>::new(root).unwrap();
        assert!(manager.contexts().unwrap().count() == 1);
    }

    #[test]
    fn context_manager_transact() {
        use claudius::{MessageParam, MessageParamContent, MessageRole};

        #[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize, PartialEq)]
        struct TestData {
            counter: u32,
        }

        let root = Path::from(".tests").join("context_manager_transact");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        // Create contexts directory
        std::fs::create_dir_all(root.join("contexts")).unwrap();

        let mut manager = ContextManager::<TestData>::new(root).unwrap();

        // Perform a transaction
        manager
            .transact::<claudius::Error>(|_context| {
                let msg = MessageParam {
                    role: MessageRole::User,
                    content: "Test message".to_string().into(),
                };
                Ok(Transaction {
                    txid: TransactionID::generate().unwrap(),
                    data: TestData { counter: 1 },
                    msgs: vec![msg],
                    writes: vec![],
                })
            })
            .unwrap();

        // Verify the transaction was persisted
        let messages: Vec<_> = manager.messages().unwrap().collect();
        assert_eq!(messages.len(), 1);
        match &messages[0] {
            Ok(msg) => {
                if let MessageParamContent::String(text) = &msg.content {
                    assert_eq!(text, "Test message");
                } else {
                    panic!("Expected string content");
                }
            }
            Err(_) => panic!("Expected Ok message"),
        }
    }

    #[test]
    fn context_manager_persistence() {
        #[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize, PartialEq)]
        struct TestData {
            value: String,
        }

        let root = Path::from(".tests").join("context_manager_persistence");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        // Create contexts directory
        std::fs::create_dir_all(root.join("contexts")).unwrap();

        // Create first manager and add transaction
        {
            let mut manager = ContextManager::<TestData>::new(root.clone()).unwrap();
            manager
                .transact::<claudius::Error>(|_| {
                    Ok(Transaction {
                        txid: TransactionID::generate().unwrap(),
                        data: TestData {
                            value: "persisted".to_string(),
                        },
                        msgs: vec![],
                        writes: vec![],
                    })
                })
                .unwrap();
        }

        // Create second manager and verify data is loaded
        {
            let manager = ContextManager::<TestData>::new(root).unwrap();
            let transactions: Vec<_> = manager
                .transactions()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(transactions.len(), 1);
            assert_eq!(transactions[0].data.value, "persisted");
        }
    }

    #[test]
    fn integration_workflow() {
        use claudius::{MessageParam, MessageRole};

        #[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize, PartialEq)]
        struct AppState {
            step: u32,
            description: String,
        }

        let root = Path::from(".tests").join("integration_workflow");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(root.join("contexts")).unwrap();

        let mut manager = ContextManager::<AppState>::new(root).unwrap();

        // Step 1: Initial transaction
        manager
            .transact::<claudius::Error>(|_| {
                let msg = MessageParam {
                    role: MessageRole::User,
                    content: "Initialize system".to_string().into(),
                };
                Ok(Transaction {
                    txid: TransactionID::generate().unwrap(),
                    data: AppState {
                        step: 1,
                        description: "Initialized".to_string(),
                    },
                    msgs: vec![msg],
                    writes: vec![FileWrite {
                        mount: MountID::generate().unwrap(),
                        path: "/state.txt".to_string(),
                        data: "step=1".to_string(),
                    }],
                })
            })
            .unwrap();

        // Step 2: Another transaction
        manager
            .transact::<claudius::Error>(|_| {
                let msg = MessageParam {
                    role: MessageRole::Assistant,
                    content: "System initialized".to_string().into(),
                };
                Ok(Transaction {
                    txid: TransactionID::generate().unwrap(),
                    data: AppState {
                        step: 2,
                        description: "Running".to_string(),
                    },
                    msgs: vec![msg],
                    writes: vec![],
                })
            })
            .unwrap();

        // Verify all messages
        let messages: Vec<_> = manager
            .messages()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(messages.len(), 2);

        // Verify transactions
        let transactions: Vec<_> = manager
            .transactions()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(transactions.len(), 2);
        assert_eq!(transactions[0].data.step, 1);
        assert_eq!(transactions[1].data.step, 2);

        // Test compaction
        manager
            .compact(|iter| {
                // Keep only the last transaction
                let mut last_tx = None;
                for tx in iter {
                    last_tx = Some(tx);
                }
                last_tx.into_iter()
            })
            .unwrap();

        // After compaction, we should have a new context with just one transaction
        let contexts_count = manager.contexts().unwrap().count();
        assert_eq!(contexts_count, 2); // Original + compacted
    }
}
