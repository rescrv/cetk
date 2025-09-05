#![doc = include_str!("../README.md")]

use std::fmt::Debug;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;

use claudius::{Anthropic, MessageParam};
use one_two_eight::generate_id;
use utf8path::Path;

///////////////////////////////////////////// Constants ////////////////////////////////////////////

const FILE_SIZE_LIMIT: usize = 8192;

/////////////////////////////////////////////// Error //////////////////////////////////////////////

pub enum Error {}

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

////////////////////////////////////////////// AgentID /////////////////////////////////////////////

generate_id!(AgentID, "agent:");
generate_id_serde!(AgentID, AgentIDVisitor);

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
pub struct Transaction {
    pub agent_id: AgentID,
    pub context_seq_no: u32,
    pub transaction_seq_no: u64,
    pub msgs: Vec<MessageParam>,
    pub writes: Vec<FileWrite>,
}

impl Transaction {
    /// Iterate over the messages of this transaction.
    pub fn messages(&self) -> impl DoubleEndedIterator<Item = MessageParam> + '_ {
        self.msgs.iter().cloned()
    }

    // TODO(claude):  Make this function return an InvariantViolation enum and not assert.
    fn check_invariants(&self) {
        for w in self.writes.iter() {
            assert!(w.data.len() < FILE_SIZE_LIMIT);
        }
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

////////////////////////////////////////////// Context /////////////////////////////////////////////

/// An individual context.  Defined as a sequence of transactions, it can also be seen as a
/// sequence of messages, or a projection of the filesystem (although we will not pursue this
/// variant now as Context is data rather than code + data).
pub struct Context<'a> {
    manager: &'a ContextManager,
    agent_id: AgentID,
    context_seq_no: u32,
    transactions: Vec<Transaction>,
}

impl Context<'_> {
    /// Iterate over the transactions of this context.
    pub fn transactions(&self) -> impl DoubleEndedIterator<Item = Transaction> {
        self.transactions.iter().cloned()
    }

    /// Iterate over the messages of this context.
    pub fn messages(&self) -> impl DoubleEndedIterator<Item = MessageParam> {
        self.transactions.iter().flat_map(|tx| tx.messages())
    }

    // Insert the next transaction into durable storage.
    pub fn transact(&mut self, mut transaction: Transaction) -> Result<(), Error> {
        self.check_invariants();
        transaction.transaction_seq_no = self
            .transactions
            .len()
            .checked_add(1)
            .unwrap()
            .try_into()
            .unwrap();
        let res = self
            .manager
            .transact(self.agent_id, self.context_seq_no, transaction.clone());
        if res.is_ok() {
            self.transactions.push(transaction);
            self.check_invariants();
        }
        res
    }

    // TODO(claude):  Make this function return an InvariantViolation enum and not assert.
    fn check_invariants(&self) {
        for (idx, (lhs, rhs)) in self
            .transactions
            .iter()
            .zip(self.transactions.iter().skip(1))
            .enumerate()
        {
            assert_eq!(lhs.agent_id, rhs.agent_id);
            assert_eq!(lhs.context_seq_no, rhs.context_seq_no);
            assert_eq!(idx as u64, lhs.transaction_seq_no,);
            assert_eq!(
                lhs.transaction_seq_no.saturating_add(1),
                rhs.transaction_seq_no
            );
        }
    }
}

////////////////////////////////////////// ContextManager //////////////////////////////////////////

pub struct ContextManager {}

impl ContextManager {
    // TODO(claude): () is a placeholder for the chroma client.  Switch it over.
    pub fn new(claudius: Anthropic, chroma: ()) -> Self {
        todo!("CLAUDE: implement this");
    }

    pub fn open(&self, agent_id: AgentID) -> Context {
        todo!("CLAUDE: implement this");
    }

    pub fn transact(
        &self,
        agent_id: AgentID,
        context_seq_no: u32,
        transaction: Transaction,
    ) -> Result<(), Error> {
        todo!("CLAUDE: implement this");
    }
}
