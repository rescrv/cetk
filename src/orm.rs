use std::collections::BTreeMap;

use claudius::MessageParam;
use prototk::FieldNumber;
use tuple_key::{Direction, TupleKey};
use tuple_key_derive::TypedTupleKey;

use crate::transaction::{FileWrite, Transaction};
use crate::{AgentID, CHUNK_SIZE_LIMIT};

////////////////////////////////////////////// Errors //////////////////////////////////////////////

/// Error that occurs during ORM operations.
#[derive(Debug)]
pub enum OrmError {
    /// Invalid key format.
    InvalidKey { key: String, reason: String },
    /// Serialization error.
    Serialization {
        operation: String,
        error: serde_json::Error,
    },
    /// Storage error.
    Storage { operation: String, error: String },
    /// Data invariant violation.
    InvariantViolation { violation: String },
}

impl std::fmt::Display for OrmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrmError::InvalidKey { key, reason } => {
                write!(f, "Invalid key '{}': {}", key, reason)
            }
            OrmError::Serialization { operation, error } => {
                write!(f, "Serialization error in {}: {}", operation, error)
            }
            OrmError::Storage { operation, error } => {
                write!(f, "Storage error in {}: {}", operation, error)
            }
            OrmError::InvariantViolation { violation } => {
                write!(f, "Invariant violation: {}", violation)
            }
        }
    }
}

impl std::error::Error for OrmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OrmError::Serialization { error, .. } => Some(error),
            _ => None,
        }
    }
}

/////////////////////////////////////////// TableSetID ///////////////////////////////////////////

use one_two_eight::generate_id;

generate_id!(TableSetID, "tableset:");

/// Generate the serde Deserialize/Serialize routines for TableSetID.
impl serde::Serialize for TableSetID {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let s = self.to_string();
        serializer.serialize_str(&s)
    }
}

impl<'de> serde::Deserialize<'de> for TableSetID {
    fn deserialize<D>(deserializer: D) -> Result<TableSetID, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_str(TableSetIDVisitor)
    }
}

struct TableSetIDVisitor;

impl<'de> serde::de::Visitor<'de> for TableSetIDVisitor {
    type Value = TableSetID;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a TableSetID")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        TableSetID::from_human_readable(value).ok_or_else(|| E::custom("not a valid tableset:UUID"))
    }
}

////////////////////////////////////////// Key Constants //////////////////////////////////////////

/// Key component constants as defined in EXAMPLE.md
mod key_constants {
    pub const AGENT_TABLE: &str = "1_A";
    pub const AGENT_ID_COLUMN: &str = "1_B";
    pub const CREATED_AT_COLUMN: &str = "2_A";
    pub const UPDATED_AT_COLUMN: &str = "3_A";
    pub const CONTEXT_COLUMN: &str = "4";
    pub const CONTEXT_SEQ_NO_COLUMN: &str = "1_C";
    pub const TRANSACTION_COLUMN: &str = "1_D";
    pub const TRANSACTION_SEQ_NO_COLUMN: &str = "1_E";
    pub const MESSAGES_COLUMN: &str = "2_B";
    pub const WRITES_COLUMN: &str = "3_B";
}

////////////////////////////////////////////// Agent ///////////////////////////////////////////////

/// Agent data structure as defined in the EXAMPLE.md schema.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Agent {
    pub agent_id: AgentID,
    pub created_at: std::time::SystemTime,
    pub updated_at: std::time::SystemTime,
    pub contexts: BTreeMap<u32, Context>,
}

impl Agent {
    /// Create a new agent with the given ID.
    pub fn new(agent_id: AgentID) -> Self {
        let now = std::time::SystemTime::now();
        Self {
            agent_id,
            created_at: now,
            updated_at: now,
            contexts: BTreeMap::new(),
        }
    }

    /// Update the agent's last updated timestamp.
    pub fn touch(&mut self) {
        self.updated_at = std::time::SystemTime::now();
    }

    /// Add or update a context.
    pub fn add_context(&mut self, context_seq_no: u32, context: Context) {
        self.contexts.insert(context_seq_no, context);
        self.touch();
    }

    /// Get a context by sequence number.
    pub fn get_context(&self, context_seq_no: u32) -> Option<&Context> {
        self.contexts.get(&context_seq_no)
    }

    /// Get a mutable context by sequence number.
    pub fn get_context_mut(&mut self, context_seq_no: u32) -> Option<&mut Context> {
        self.contexts.get_mut(&context_seq_no)
    }

    /// List all context sequence numbers.
    pub fn context_seq_nos(&self) -> impl Iterator<Item = u32> + '_ {
        self.contexts.keys().copied()
    }
}

///////////////////////////////////////////// Context //////////////////////////////////////////////

/// Context data structure containing transactions.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Context {
    pub context_seq_no: u32,
    pub transactions: BTreeMap<u32, ContextTransaction>,
}

impl Context {
    /// Create a new context with the given sequence number.
    pub fn new(context_seq_no: u32) -> Self {
        Self {
            context_seq_no,
            transactions: BTreeMap::new(),
        }
    }

    /// Add or update a transaction.
    pub fn add_transaction(&mut self, transaction_seq_no: u32, transaction: ContextTransaction) {
        self.transactions.insert(transaction_seq_no, transaction);
    }

    /// Get a transaction by sequence number.
    pub fn get_transaction(&self, transaction_seq_no: u32) -> Option<&ContextTransaction> {
        self.transactions.get(&transaction_seq_no)
    }

    /// Get a mutable transaction by sequence number.
    pub fn get_transaction_mut(
        &mut self,
        transaction_seq_no: u32,
    ) -> Option<&mut ContextTransaction> {
        self.transactions.get_mut(&transaction_seq_no)
    }

    /// List all transaction sequence numbers.
    pub fn transaction_seq_nos(&self) -> impl Iterator<Item = u32> + '_ {
        self.transactions.keys().copied()
    }
}

/////////////////////////////////////// ContextTransaction ///////////////////////////////////////

/// Transaction data specific to contexts (different from the general Transaction type).
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ContextTransaction {
    pub transaction_seq_no: u32,
    pub messages: Vec<MessageParam>,
    pub writes: Vec<FileWrite>,
}

impl ContextTransaction {
    /// Create a new context transaction.
    pub fn new(transaction_seq_no: u32) -> Self {
        Self {
            transaction_seq_no,
            messages: Vec::new(),
            writes: Vec::new(),
        }
    }

    /// Add a message to the transaction.
    pub fn add_message(&mut self, message: MessageParam) {
        self.messages.push(message);
    }

    /// Add a file write to the transaction.
    pub fn add_write(&mut self, write: FileWrite) {
        self.writes.push(write);
    }

    /// Convert to the general Transaction type.
    pub fn to_transaction(&self, agent_id: AgentID, context_seq_no: u32) -> Transaction {
        Transaction {
            agent_id,
            context_seq_no,
            transaction_seq_no: self.transaction_seq_no as u64,
            msgs: self.messages.clone(),
            writes: self.writes.clone(),
        }
    }

    /// Check if the transaction would exceed chunk size limits.
    pub fn check_chunk_size(&self) -> Result<(), OrmError> {
        let serialized = serde_json::to_string(self).map_err(|e| OrmError::Serialization {
            operation: "check_chunk_size".to_string(),
            error: e,
        })?;

        if serialized.len() >= CHUNK_SIZE_LIMIT {
            return Err(OrmError::InvariantViolation {
                violation: format!(
                    "Transaction size {} exceeds limit {}",
                    serialized.len(),
                    CHUNK_SIZE_LIMIT
                ),
            });
        }

        Ok(())
    }
}

/////////////////////////////////////////////// Keys ///////////////////////////////////////////////

/// Key for agent metadata (created_at, updated_at).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, TypedTupleKey)]
pub struct AgentMetadataKey {
    #[tuple_key(1)]
    pub tableset_id: String,
    #[tuple_key(2)]
    pub agent_table: String,
    #[tuple_key(3)]
    pub agent_id_column: String,
    #[tuple_key(4)]
    pub agent_id: String,
    #[tuple_key(5)]
    pub field: String,
}

impl AgentMetadataKey {
    pub fn new(tableset_id: &TableSetID, agent_id: &AgentID, field: &str) -> Self {
        Self {
            tableset_id: tableset_id.to_string(),
            agent_table: key_constants::AGENT_TABLE.to_string(),
            agent_id_column: key_constants::AGENT_ID_COLUMN.to_string(),
            agent_id: agent_id.to_string(),
            field: field.to_string(),
        }
    }
}

/// Key for message data.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, TypedTupleKey)]
pub struct MessageKey {
    #[tuple_key(1)]
    pub tableset_id: String,
    #[tuple_key(2)]
    pub agent_table: String,
    #[tuple_key(3)]
    pub agent_id_column: String,
    #[tuple_key(4)]
    pub agent_id: String,
    #[tuple_key(5)]
    pub context_column: String,
    #[tuple_key(6)]
    pub context_seq_no_column: String,
    #[tuple_key(7)]
    pub context_seq_no: u32,
    #[tuple_key(8)]
    pub transaction_column: String,
    #[tuple_key(9)]
    pub transaction_seq_no_column: String,
    #[tuple_key(10)]
    pub transaction_seq_no: u32,
    #[tuple_key(11)]
    pub messages_column: String,
    #[tuple_key(12)]
    pub index: u64,
}

impl MessageKey {
    pub fn new(
        tableset_id: &TableSetID,
        agent_id: &AgentID,
        context_seq_no: u32,
        transaction_seq_no: u32,
        index: usize,
    ) -> Self {
        Self {
            tableset_id: tableset_id.to_string(),
            agent_table: key_constants::AGENT_TABLE.to_string(),
            agent_id_column: key_constants::AGENT_ID_COLUMN.to_string(),
            agent_id: agent_id.to_string(),
            context_column: key_constants::CONTEXT_COLUMN.to_string(),
            context_seq_no_column: key_constants::CONTEXT_SEQ_NO_COLUMN.to_string(),
            context_seq_no,
            transaction_column: key_constants::TRANSACTION_COLUMN.to_string(),
            transaction_seq_no_column: key_constants::TRANSACTION_SEQ_NO_COLUMN.to_string(),
            transaction_seq_no,
            messages_column: key_constants::MESSAGES_COLUMN.to_string(),
            index: index as u64,
        }
    }
}

/// Key for file write data.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, TypedTupleKey)]
pub struct WriteKey {
    #[tuple_key(1)]
    pub tableset_id: String,
    #[tuple_key(2)]
    pub agent_table: String,
    #[tuple_key(3)]
    pub agent_id_column: String,
    #[tuple_key(4)]
    pub agent_id: String,
    #[tuple_key(5)]
    pub context_column: String,
    #[tuple_key(6)]
    pub context_seq_no_column: String,
    #[tuple_key(7)]
    pub context_seq_no: u32,
    #[tuple_key(8)]
    pub transaction_column: String,
    #[tuple_key(9)]
    pub transaction_seq_no_column: String,
    #[tuple_key(10)]
    pub transaction_seq_no: u32,
    #[tuple_key(11)]
    pub writes_column: String,
    #[tuple_key(12)]
    pub index: u64,
}

impl WriteKey {
    pub fn new(
        tableset_id: &TableSetID,
        agent_id: &AgentID,
        context_seq_no: u32,
        transaction_seq_no: u32,
        index: usize,
    ) -> Self {
        Self {
            tableset_id: tableset_id.to_string(),
            agent_table: key_constants::AGENT_TABLE.to_string(),
            agent_id_column: key_constants::AGENT_ID_COLUMN.to_string(),
            agent_id: agent_id.to_string(),
            context_column: key_constants::CONTEXT_COLUMN.to_string(),
            context_seq_no_column: key_constants::CONTEXT_SEQ_NO_COLUMN.to_string(),
            context_seq_no,
            transaction_column: key_constants::TRANSACTION_COLUMN.to_string(),
            transaction_seq_no_column: key_constants::TRANSACTION_SEQ_NO_COLUMN.to_string(),
            transaction_seq_no,
            writes_column: key_constants::WRITES_COLUMN.to_string(),
            index: index as u64,
        }
    }
}

/// Generic key type that can hold any key.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Key {
    AgentMetadata(AgentMetadataKey),
    Message(MessageKey),
    Write(WriteKey),
    Raw(TupleKey),
}

impl Key {
    /// Create a key for agent metadata.
    pub fn agent_metadata(tableset_id: &TableSetID, agent_id: &AgentID, field: &str) -> Self {
        Self::AgentMetadata(AgentMetadataKey::new(tableset_id, agent_id, field))
    }

    /// Create a key for message data.
    pub fn message(
        tableset_id: &TableSetID,
        agent_id: &AgentID,
        context_seq_no: u32,
        transaction_seq_no: u32,
        index: usize,
    ) -> Self {
        Self::Message(MessageKey::new(
            tableset_id,
            agent_id,
            context_seq_no,
            transaction_seq_no,
            index,
        ))
    }

    /// Create a key for file write data.
    pub fn write(
        tableset_id: &TableSetID,
        agent_id: &AgentID,
        context_seq_no: u32,
        transaction_seq_no: u32,
        index: usize,
    ) -> Self {
        Self::Write(WriteKey::new(
            tableset_id,
            agent_id,
            context_seq_no,
            transaction_seq_no,
            index,
        ))
    }

    /// Create a prefix key from a raw TupleKey.
    pub fn prefix(tuple_key: TupleKey) -> Self {
        Self::Raw(tuple_key)
    }

    /// Create a key from a raw TupleKey (alias for prefix).
    pub fn new(tuple_key: TupleKey) -> Self {
        Self::Raw(tuple_key)
    }

    /// Get the underlying TupleKey.
    pub fn to_tuple_key(&self) -> TupleKey {
        match self {
            Key::AgentMetadata(k) => k.clone().into(),
            Key::Message(k) => k.clone().into(),
            Key::Write(k) => k.clone().into(),
            Key::Raw(k) => k.clone(),
        }
    }

    /// Get the underlying bytes.
    pub fn as_bytes(&self) -> Vec<u8> {
        self.to_tuple_key().as_bytes().to_vec()
    }

    /// Check if this key is a prefix of another key.
    pub fn is_prefix_of(&self, other: &Key) -> bool {
        let self_bytes = self.as_bytes();
        let other_bytes = other.as_bytes();
        other_bytes.starts_with(&self_bytes)
    }
}

/////////////////////////////////////////// Storage ///////////////////////////////////////////////

/// In-memory storage implementation for the ORM.
#[derive(Clone, Debug, Default)]
pub struct MemoryStorage {
    data: BTreeMap<Vec<u8>, String>,
}

impl MemoryStorage {
    /// Create a new memory storage.
    pub fn new() -> Self {
        Self {
            data: BTreeMap::new(),
        }
    }

    /// Store a key-value pair.
    pub fn put(&mut self, key: Key, value: String) -> Result<(), OrmError> {
        let key_bytes = key.as_bytes();
        self.data.insert(key_bytes, value);
        Ok(())
    }

    /// Retrieve a value by key.
    pub fn get(&self, key: &Key) -> Option<&String> {
        let key_bytes = key.as_bytes();
        self.data.get(&key_bytes)
    }

    /// Delete a key-value pair.
    pub fn delete(&mut self, key: &Key) -> Result<(), OrmError> {
        let key_bytes = key.as_bytes();
        self.data.remove(&key_bytes);
        Ok(())
    }

    /// List all keys with a given prefix.
    pub fn list_keys_with_prefix(&self, prefix: &Key) -> Vec<Key> {
        let prefix_bytes = prefix.as_bytes();
        self.data
            .keys()
            .filter(|key_bytes| key_bytes.starts_with(&prefix_bytes))
            .map(|key_bytes| Key::Raw(TupleKey::from(key_bytes.as_slice())))
            .collect()
    }

    /// Get all data (for debugging).
    pub fn all_data(&self) -> Vec<(Key, &String)> {
        self.data
            .iter()
            .map(|(key_bytes, value)| (Key::Raw(TupleKey::from(key_bytes.as_slice())), value))
            .collect()
    }
}

/////////////////////////////////////////////// ORM ////////////////////////////////////////////////

/// Object-Relational Mapping layer for the agent framework.
#[derive(Clone, Debug)]
pub struct Orm {
    tableset_id: TableSetID,
    storage: MemoryStorage,
}

impl Orm {
    /// Create a new ORM instance.
    pub fn new(tableset_id: TableSetID) -> Self {
        Self {
            tableset_id,
            storage: MemoryStorage::new(),
        }
    }

    /// Store an agent.
    pub fn store_agent(&mut self, agent: &Agent) -> Result<(), OrmError> {
        // Store metadata
        let created_at_key = Key::agent_metadata(
            &self.tableset_id,
            &agent.agent_id,
            key_constants::CREATED_AT_COLUMN,
        );
        let updated_at_key = Key::agent_metadata(
            &self.tableset_id,
            &agent.agent_id,
            key_constants::UPDATED_AT_COLUMN,
        );

        let created_at_timestamp = agent
            .created_at
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| OrmError::Storage {
                operation: "serialize_timestamp".to_string(),
                error: e.to_string(),
            })?
            .as_secs();

        let updated_at_timestamp = agent
            .updated_at
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| OrmError::Storage {
                operation: "serialize_timestamp".to_string(),
                error: e.to_string(),
            })?
            .as_secs();

        self.storage
            .put(created_at_key, created_at_timestamp.to_string())?;
        self.storage
            .put(updated_at_key, updated_at_timestamp.to_string())?;

        // Store contexts and transactions
        for (context_seq_no, context) in &agent.contexts {
            for (transaction_seq_no, transaction) in &context.transactions {
                // Store messages
                for (index, message) in transaction.messages.iter().enumerate() {
                    let key = Key::message(
                        &self.tableset_id,
                        &agent.agent_id,
                        *context_seq_no,
                        *transaction_seq_no,
                        index,
                    );
                    let value =
                        serde_json::to_string(message).map_err(|e| OrmError::Serialization {
                            operation: "serialize_message".to_string(),
                            error: e,
                        })?;
                    self.storage.put(key, value)?;
                }

                // Store writes
                for (index, write) in transaction.writes.iter().enumerate() {
                    let key = Key::write(
                        &self.tableset_id,
                        &agent.agent_id,
                        *context_seq_no,
                        *transaction_seq_no,
                        index,
                    );
                    let value =
                        serde_json::to_string(write).map_err(|e| OrmError::Serialization {
                            operation: "serialize_write".to_string(),
                            error: e,
                        })?;
                    self.storage.put(key, value)?;
                }
            }
        }

        Ok(())
    }

    /// Retrieve an agent.
    pub fn get_agent(&self, agent_id: &AgentID) -> Result<Option<Agent>, OrmError> {
        // Check if agent exists by looking for created_at
        let created_at_key = Key::agent_metadata(
            &self.tableset_id,
            agent_id,
            key_constants::CREATED_AT_COLUMN,
        );

        let created_at_str = match self.storage.get(&created_at_key) {
            Some(s) => s,
            None => return Ok(None),
        };

        let updated_at_key = Key::agent_metadata(
            &self.tableset_id,
            agent_id,
            key_constants::UPDATED_AT_COLUMN,
        );

        let updated_at_str =
            self.storage
                .get(&updated_at_key)
                .ok_or_else(|| OrmError::Storage {
                    operation: "get_updated_at".to_string(),
                    error: "Missing updated_at timestamp".to_string(),
                })?;

        let created_at_secs: u64 = created_at_str.parse().map_err(|e| OrmError::Storage {
            operation: "parse_created_at".to_string(),
            error: format!("Invalid timestamp: {}", e),
        })?;

        let updated_at_secs: u64 = updated_at_str.parse().map_err(|e| OrmError::Storage {
            operation: "parse_updated_at".to_string(),
            error: format!("Invalid timestamp: {}", e),
        })?;

        let created_at = std::time::UNIX_EPOCH + std::time::Duration::from_secs(created_at_secs);
        let updated_at = std::time::UNIX_EPOCH + std::time::Duration::from_secs(updated_at_secs);

        // Build contexts and transactions
        let mut contexts = BTreeMap::new();

        // For now, just return an empty agent. The key parsing with TypedTupleKey
        // is more complex and we can implement it properly later.
        // TODO: Implement proper key parsing to reconstruct contexts and transactions

        Ok(Some(Agent {
            agent_id: *agent_id,
            created_at,
            updated_at,
            contexts,
        }))
    }

    /// List all agent IDs.
    pub fn list_agents(&self) -> Result<Vec<AgentID>, OrmError> {
        // For now, return empty list. Implementing proper prefix search requires
        // more complex tuple key parsing.
        // TODO: Implement proper prefix search to find agents
        Ok(vec![])
    }

    /// Delete an agent and all its data.
    pub fn delete_agent(&mut self, agent_id: &AgentID) -> Result<(), OrmError> {
        // For now, just remove specific keys we know about
        // TODO: Implement proper prefix-based deletion
        let created_at_key = Key::agent_metadata(
            &self.tableset_id,
            agent_id,
            key_constants::CREATED_AT_COLUMN,
        );
        let updated_at_key = Key::agent_metadata(
            &self.tableset_id,
            agent_id,
            key_constants::UPDATED_AT_COLUMN,
        );

        self.storage.delete(&created_at_key)?;
        self.storage.delete(&updated_at_key)?;

        Ok(())
    }

    /// Get the underlying storage (for testing/debugging).
    pub fn storage(&self) -> &MemoryStorage {
        &self.storage
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claudius::MessageRole;

    fn test_tableset_id() -> TableSetID {
        TableSetID::from_human_readable("tableset:00000000-0000-0000-0000-000000000001").unwrap()
    }

    fn test_agent_id() -> AgentID {
        AgentID::from_human_readable("agent:00000000-0000-0000-0000-000000000001").unwrap()
    }

    fn test_mount_id() -> crate::MountID {
        crate::MountID::from_human_readable("mount:00000000-0000-0000-0000-000000000001").unwrap()
    }

    #[test]
    fn tableset_id_creates_from_human_readable() {
        let id_str = "tableset:00000000-0000-0000-0000-000000000001";
        let tableset_id = TableSetID::from_human_readable(id_str);
        assert!(tableset_id.is_some());
        assert_eq!(tableset_id.unwrap().to_string(), id_str);
    }

    #[test]
    fn tableset_id_generates_unique_ids() {
        let id1 = TableSetID::generate().unwrap();
        let id2 = TableSetID::generate().unwrap();
        assert_ne!(id1, id2);
        assert!(id1.to_string().starts_with("tableset:"));
        assert!(id2.to_string().starts_with("tableset:"));
    }

    #[test]
    fn agent_creation() {
        let agent_id = test_agent_id();
        let agent = Agent::new(agent_id);

        assert_eq!(agent.agent_id, agent_id);
        assert!(agent.contexts.is_empty());
        assert!(agent.created_at <= agent.updated_at);
    }

    #[test]
    fn agent_touch_updates_timestamp() {
        let agent_id = test_agent_id();
        let mut agent = Agent::new(agent_id);
        let original_updated = agent.updated_at;

        std::thread::sleep(std::time::Duration::from_millis(1));
        agent.touch();

        assert!(agent.updated_at > original_updated);
    }

    #[test]
    fn context_creation() {
        let context = Context::new(42);
        assert_eq!(context.context_seq_no, 42);
        assert!(context.transactions.is_empty());
    }

    #[test]
    fn context_transaction_creation() {
        let transaction = ContextTransaction::new(100);
        assert_eq!(transaction.transaction_seq_no, 100);
        assert!(transaction.messages.is_empty());
        assert!(transaction.writes.is_empty());
    }

    #[test]
    fn context_transaction_to_transaction_conversion() {
        let agent_id = test_agent_id();
        let mut ctx_transaction = ContextTransaction::new(42);

        ctx_transaction.add_message(MessageParam {
            role: MessageRole::User,
            content: "Hello".into(),
        });

        ctx_transaction.add_write(FileWrite {
            mount: test_mount_id(),
            path: "test.txt".to_string(),
            data: "test content".to_string(),
        });

        let transaction = ctx_transaction.to_transaction(agent_id, 1);

        assert_eq!(transaction.agent_id, agent_id);
        assert_eq!(transaction.context_seq_no, 1);
        assert_eq!(transaction.transaction_seq_no, 42);
        assert_eq!(transaction.msgs.len(), 1);
        assert_eq!(transaction.writes.len(), 1);
    }

    #[test]
    fn key_creation_agent_metadata() {
        let tableset_id = test_tableset_id();
        let agent_id = test_agent_id();

        let key = Key::agent_metadata(&tableset_id, &agent_id, "created_at");

        // Just test that we can create a key - the internal structure is complex
        assert!(!key.as_bytes().is_empty());
    }

    #[test]
    fn key_creation_message() {
        let tableset_id = test_tableset_id();
        let agent_id = test_agent_id();

        let key = Key::message(&tableset_id, &agent_id, 1, 42, 0);

        // Just test that we can create a key
        assert!(!key.as_bytes().is_empty());
    }

    #[test]
    fn key_from_bytes_roundtrip() {
        let tableset_id = test_tableset_id();
        let agent_id = test_agent_id();

        let original_key = Key::agent_metadata(&tableset_id, &agent_id, "created_at");
        let bytes = original_key.as_bytes();
        let restored_key = Key::from_bytes(bytes);

        assert_eq!(original_key.as_bytes(), restored_key.as_bytes());
    }

    #[test]
    fn key_is_prefix_of() {
        let tableset_id = test_tableset_id();
        let agent_id = test_agent_id();

        let prefix_key = Key::agent_metadata(&tableset_id, &agent_id, "created_at");
        let longer_key = Key::message(&tableset_id, &agent_id, 1, 42, 0);

        // The agent metadata key should not be a prefix of a message key since they have different structures
        // But a proper prefix should work
        let mut prefix_tuple = TupleKey::default();
        prefix_tuple.extend_with_key(
            FieldNumber::must(1),
            tableset_id.to_string(),
            Direction::Forward,
        );
        let true_prefix = Key::prefix(prefix_tuple);

        assert!(true_prefix.is_prefix_of(&prefix_key));
        assert!(true_prefix.is_prefix_of(&longer_key));
    }

    #[test]
    fn memory_storage_basic_operations() {
        let mut storage = MemoryStorage::new();
        let mut tuple_key = TupleKey::default();
        tuple_key.extend_with_key(FieldNumber::must(1), "test".to_string(), Direction::Forward);
        let key = Key::new(tuple_key);
        let value = "test_value".to_string();

        assert!(storage.get(&key).is_none());

        storage.put(key.clone(), value.clone()).unwrap();
        assert_eq!(storage.get(&key), Some(&value));

        storage.delete(&key).unwrap();
        assert!(storage.get(&key).is_none());
    }

    #[test]
    fn memory_storage_prefix_listing() {
        let mut storage = MemoryStorage::new();

        let mut key1 = TupleKey::default();
        key1.extend_with_key(
            FieldNumber::must(1),
            "prefix".to_string(),
            Direction::Forward,
        );
        key1.extend_with_key(FieldNumber::must(2), "a".to_string(), Direction::Forward);
        let key1 = Key::new(key1);

        let mut key2 = TupleKey::default();
        key2.extend_with_key(
            FieldNumber::must(1),
            "prefix".to_string(),
            Direction::Forward,
        );
        key2.extend_with_key(FieldNumber::must(2), "b".to_string(), Direction::Forward);
        let key2 = Key::new(key2);

        let mut key3 = TupleKey::default();
        key3.extend_with_key(
            FieldNumber::must(1),
            "other".to_string(),
            Direction::Forward,
        );
        key3.extend_with_key(FieldNumber::must(2), "c".to_string(), Direction::Forward);
        let key3 = Key::new(key3);

        storage.put(key1.clone(), "value1".to_string()).unwrap();
        storage.put(key2.clone(), "value2".to_string()).unwrap();
        storage.put(key3.clone(), "value3".to_string()).unwrap();

        let mut prefix_tuple = TupleKey::default();
        prefix_tuple.extend_with_key(
            FieldNumber::must(1),
            "prefix".to_string(),
            Direction::Forward,
        );
        let prefix = Key::new(prefix_tuple);

        let matching_keys = storage.list_keys_with_prefix(&prefix);

        assert_eq!(matching_keys.len(), 2);
        assert!(
            matching_keys
                .iter()
                .any(|k| k.as_bytes() == key1.as_bytes())
        );
        assert!(
            matching_keys
                .iter()
                .any(|k| k.as_bytes() == key2.as_bytes())
        );
        assert!(
            !matching_keys
                .iter()
                .any(|k| k.as_bytes() == key3.as_bytes())
        );
    }

    #[test]
    fn orm_store_and_get_agent() {
        let tableset_id = test_tableset_id();
        let mut orm = Orm::new(tableset_id);

        let agent_id = test_agent_id();
        let mut agent = Agent::new(agent_id);

        // Add a context with a transaction
        let mut context = Context::new(1);
        let mut transaction = ContextTransaction::new(42);

        transaction.add_message(MessageParam {
            role: MessageRole::User,
            content: "Hello world".into(),
        });

        transaction.add_write(FileWrite {
            mount: test_mount_id(),
            path: "test.txt".to_string(),
            data: "Hello file".to_string(),
        });

        context.add_transaction(42, transaction);
        agent.add_context(1, context);

        // Store the agent
        orm.store_agent(&agent).unwrap();

        // Retrieve the agent
        let retrieved_agent = orm.get_agent(&agent_id).unwrap().unwrap();

        assert_eq!(retrieved_agent.agent_id, agent.agent_id);
        assert_eq!(retrieved_agent.contexts.len(), 1);

        let retrieved_context = retrieved_agent.get_context(1).unwrap();
        assert_eq!(retrieved_context.transactions.len(), 1);

        let retrieved_transaction = retrieved_context.get_transaction(42).unwrap();
        assert_eq!(retrieved_transaction.messages.len(), 1);
        assert_eq!(retrieved_transaction.writes.len(), 1);
    }

    #[test]
    fn orm_get_nonexistent_agent() {
        let tableset_id = test_tableset_id();
        let orm = Orm::new(tableset_id);

        let agent_id = test_agent_id();
        let result = orm.get_agent(&agent_id).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn orm_list_agents() {
        let tableset_id = test_tableset_id();
        let mut orm = Orm::new(tableset_id);

        let agent1_id =
            AgentID::from_human_readable("agent:00000000-0000-0000-0000-000000000001").unwrap();
        let agent2_id =
            AgentID::from_human_readable("agent:00000000-0000-0000-0000-000000000002").unwrap();

        let agent1 = Agent::new(agent1_id);
        let agent2 = Agent::new(agent2_id);

        orm.store_agent(&agent1).unwrap();
        orm.store_agent(&agent2).unwrap();

        let agent_list = orm.list_agents().unwrap();
        assert_eq!(agent_list.len(), 2);
        assert!(agent_list.contains(&agent1_id));
        assert!(agent_list.contains(&agent2_id));
    }

    #[test]
    fn orm_delete_agent() {
        let tableset_id = test_tableset_id();
        let mut orm = Orm::new(tableset_id);

        let agent_id = test_agent_id();
        let agent = Agent::new(agent_id);

        orm.store_agent(&agent).unwrap();
        assert!(orm.get_agent(&agent_id).unwrap().is_some());

        orm.delete_agent(&agent_id).unwrap();
        assert!(orm.get_agent(&agent_id).unwrap().is_none());
    }

    #[test]
    fn context_transaction_chunk_size_check() {
        let mut transaction = ContextTransaction::new(1);

        // Small transaction should pass
        transaction.add_message(MessageParam {
            role: MessageRole::User,
            content: "Small message".into(),
        });

        assert!(transaction.check_chunk_size().is_ok());

        // Large transaction should fail
        let large_content = "x".repeat(CHUNK_SIZE_LIMIT);
        transaction.add_message(MessageParam {
            role: MessageRole::User,
            content: large_content.into(),
        });

        assert!(matches!(
            transaction.check_chunk_size(),
            Err(OrmError::InvariantViolation { .. })
        ));
    }

    #[test]
    fn key_ordering() {
        let mut key1_tuple = TupleKey::default();
        key1_tuple.extend_with_key(FieldNumber::must(1), "a".to_string(), Direction::Forward);
        let key1 = Key::new(key1_tuple);

        let mut key2_tuple = TupleKey::default();
        key2_tuple.extend_with_key(FieldNumber::must(1), "b".to_string(), Direction::Forward);
        let key2 = Key::new(key2_tuple);

        let mut key3_tuple = TupleKey::default();
        key3_tuple.extend_with_key(FieldNumber::must(1), "a".to_string(), Direction::Forward);
        key3_tuple.extend_with_key(FieldNumber::must(2), "b".to_string(), Direction::Forward);
        let key3 = Key::new(key3_tuple);

        assert!(key1 < key2);
        assert!(key1 < key3);
        assert!(key2 > key1);
    }

    #[test]
    fn agent_context_operations() {
        let agent_id = test_agent_id();
        let mut agent = Agent::new(agent_id);

        let context1 = Context::new(1);
        let context2 = Context::new(2);

        agent.add_context(1, context1);
        agent.add_context(2, context2);

        assert_eq!(agent.contexts.len(), 2);
        assert!(agent.get_context(1).is_some());
        assert!(agent.get_context(2).is_some());
        assert!(agent.get_context(3).is_none());

        let seq_nos: Vec<u32> = agent.context_seq_nos().collect();
        assert_eq!(seq_nos, vec![1, 2]);
    }

    #[test]
    fn context_transaction_operations() {
        let mut context = Context::new(1);

        let transaction1 = ContextTransaction::new(10);
        let transaction2 = ContextTransaction::new(20);

        context.add_transaction(10, transaction1);
        context.add_transaction(20, transaction2);

        assert_eq!(context.transactions.len(), 2);
        assert!(context.get_transaction(10).is_some());
        assert!(context.get_transaction(20).is_some());
        assert!(context.get_transaction(30).is_none());

        let seq_nos: Vec<u32> = context.transaction_seq_nos().collect();
        assert_eq!(seq_nos, vec![10, 20]);
    }
}
