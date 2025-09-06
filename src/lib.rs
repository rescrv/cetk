#![doc = include_str!("../README.md")]

use chromadb::ChromaClient;
use claudius::{Anthropic, MessageParam};
use one_two_eight::generate_id;

mod bullets;

pub use bullets::MarkdownList;

///////////////////////////////////////////// Constants ////////////////////////////////////////////

const CHUNK_SIZE_LIMIT: usize = 8192;

/////////////////////////////////////////////// Error //////////////////////////////////////////////

#[derive(Debug)]
pub enum Error {
    ChromaConnectionError(String),
    TransactionError(String),
    ContextNotFound(String),
    ChunkSizeExceeded(String),
    InvalidSequence(String),
    SealError(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::ChromaConnectionError(msg) => write!(f, "ChromaDB connection error: {msg}"),
            Error::TransactionError(msg) => write!(f, "Transaction error: {msg}"),
            Error::ContextNotFound(msg) => write!(f, "Context not found: {msg}"),
            Error::ChunkSizeExceeded(msg) => write!(f, "Chunk size exceeded: {msg}"),
            Error::InvalidSequence(msg) => write!(f, "Invalid sequence: {msg}"),
            Error::SealError(msg) => write!(f, "Seal error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

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

///////////////////////////////////////////// ContextID ////////////////////////////////////////////

generate_id!(ContextID, "context:");
generate_id_serde!(ContextID, ContextIDVisitor);

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

    /// Generate an embedding summary for this transaction.
    pub fn generate_embedding_summary(&self) -> String {
        let msg_count = self.msgs.len();
        let write_count = self.writes.len();
        format!(
            "Transaction {} for agent {} context {} with {} messages and {} file writes",
            self.transaction_seq_no, self.agent_id, self.context_seq_no, msg_count, write_count
        )
    }

    /// Chunk a transaction if it exceeds the size limit.
    pub fn chunk_transaction(&self) -> Result<Vec<TransactionChunk>, Error> {
        let mut serialized = serde_json::to_string(self)
            .map_err(|e| Error::TransactionError(format!("Failed to serialize: {e}")))?;

        if serialized.len() <= CHUNK_SIZE_LIMIT {
            return Ok(vec![TransactionChunk {
                agent_id: self.agent_id,
                context_seq_no: self.context_seq_no,
                transaction_seq_no: self.transaction_seq_no,
                chunk_seq_no: 0,
                total_chunks: 1,
                data: serialized,
            }]);
        }

        let mut chunks = Vec::new();
        let mut chunk_seq_no = 0;

        while !serialized.is_empty() {
            let chunk = serialized
                .chars()
                .take(CHUNK_SIZE_LIMIT)
                .collect::<String>();
            serialized = serialized
                .chars()
                .skip(CHUNK_SIZE_LIMIT)
                .collect::<String>();
            chunks.push(TransactionChunk {
                agent_id: self.agent_id,
                context_seq_no: self.context_seq_no,
                transaction_seq_no: self.transaction_seq_no,
                chunk_seq_no,
                total_chunks: 0,
                data: chunk,
            });
            chunk_seq_no += 1;
        }

        let total_chunks = chunks.len() as u32;
        for chunk in &mut chunks {
            chunk.total_chunks = total_chunks;
        }

        Ok(chunks)
    }

    fn check_invariants(&self) -> Result<(), Error> {
        for w in self.writes.iter() {
            if w.data.len() >= CHUNK_SIZE_LIMIT {
                return Err(Error::ChunkSizeExceeded(format!(
                    "File write exceeds size limit: {} bytes",
                    w.data.len()
                )));
            }
        }
        Ok(())
    }
}

///////////////////////////////////////// TransactionChunk /////////////////////////////////////////

/// A chunk of a transaction when it exceeds the storage size limit.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct TransactionChunk {
    pub agent_id: AgentID,
    pub context_seq_no: u32,
    pub transaction_seq_no: u64,
    pub chunk_seq_no: u32,
    pub total_chunks: u32,
    pub data: String,
}

///////////////////////////////////////////// FileWrite ////////////////////////////////////////////

/// Write the complete contents of data to the file at path on mount.
///
/// We assume files in the virtual filesystem should be small.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct FileWrite {
    pub mount: MountID,
    pub path: String,
    pub data: String,
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
    pub async fn transact(&mut self, mut transaction: Transaction) -> Result<(), Error> {
        self.check_invariants()?;
        transaction.transaction_seq_no = self
            .transactions
            .len()
            .checked_add(1)
            .ok_or_else(|| Error::InvalidSequence("Transaction count overflow".to_string()))?
            .try_into()
            .map_err(|_| {
                Error::InvalidSequence("Transaction sequence number overflow".to_string())
            })?;
        let res = self
            .manager
            .transact(self.agent_id, self.context_seq_no, transaction.clone())
            .await;
        if res.is_ok() {
            self.transactions.push(transaction);
            self.check_invariants()?;
        }
        res
    }

    // TODO(claude):  Make this function return an InvariantViolation enum and not assert.
    fn check_invariants(&self) -> Result<(), Error> {
        // Check each transaction individually
        for (idx, tx) in self.transactions.iter().enumerate() {
            // Verify transaction sequence number matches its position
            if idx as u64 != tx.transaction_seq_no {
                return Err(Error::InvalidSequence(format!(
                    "Transaction at index {} has sequence number {}, expected {}",
                    idx, tx.transaction_seq_no, idx
                )));
            }

            // Verify agent_id matches the context's agent_id
            if tx.agent_id != self.agent_id {
                return Err(Error::InvalidSequence(format!(
                    "Transaction {} has agent_id {}, expected {}",
                    idx, tx.agent_id, self.agent_id
                )));
            }

            // Verify context_seq_no matches the context's seq_no
            if tx.context_seq_no != self.context_seq_no {
                return Err(Error::InvalidSequence(format!(
                    "Transaction {} has context_seq_no {}, expected {}",
                    idx, tx.context_seq_no, self.context_seq_no
                )));
            }

            // Check transaction's own invariants
            tx.check_invariants()?;
        }

        // Check continuity between adjacent transactions
        for (lhs, rhs) in self
            .transactions
            .iter()
            .zip(self.transactions.iter().skip(1))
        {
            if lhs.transaction_seq_no.saturating_add(1) != rhs.transaction_seq_no {
                return Err(Error::InvalidSequence(format!(
                    "Non-continuous transaction sequence: {} -> {}",
                    lhs.transaction_seq_no, rhs.transaction_seq_no
                )));
            }
        }

        Ok(())
    }
}

////////////////////////////////////////// ContextManager //////////////////////////////////////////

pub struct ContextManager {
    _claudius: Anthropic,
    chroma: ChromaClient,
    collection_name: String,
}

impl ContextManager {
    pub async fn transact(
        &self,
        _agent_id: AgentID,
        _context_seq_no: u32,
        _transaction: Transaction,
    ) -> Result<(), Error> {
        todo!();
    }
}

//////////////////////////////////////////// ContextSeal ///////////////////////////////////////////

/// A seal marks the end of a context and points to the next context in the lineage.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ContextSeal {
    pub context_id: ContextID,
    pub next_context_id: ContextID,
    pub sealed_at_ms: u64,
    pub summary: String,
}

impl ContextSeal {
    /// Create a new context seal.
    pub fn new(context_id: ContextID, next_context_id: ContextID, summary: String) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let sealed_at_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis()
            .try_into()
            .unwrap();

        ContextSeal {
            context_id,
            next_context_id,
            sealed_at_ms,
            summary,
        }
    }
}
