//! Transaction data structures and processing.
//!
//! This module provides the core transaction system for CETK, including:
//!
//! - **Atomic Transactions**: [`Transaction`] represents atomic units of agent state changes
//! - **Chunking System**: Automatic splitting of large transactions for storage efficiency
//! - **Virtual File Operations**: [`FileWrite`] for virtual filesystem modifications
//! - **Invariant Checking**: Built-in validation to ensure data integrity
//!
//! ## Transaction Structure
//!
//! A [`Transaction`] contains:
//! - Agent, context, and transaction identifiers for hierarchy management
//! - Conversation messages that extend the agent's dialogue history
//! - File system writes that modify virtual filesystem state
//!
//! ## Size Management
//!
//! Large transactions are automatically split into [`TransactionChunk`]s when they exceed
//! [`CHUNK_SIZE_LIMIT`]. This ensures efficient storage and retrieval while maintaining
//! atomicity guarantees through proper reassembly.
//!
//! ## Examples
//!
//! ### Creating a Transaction
//!
//! ```rust
//! use cetk::{Transaction, AgentID, FileWrite, MountID};
//! use claudius::{MessageParam, MessageRole};
//!
//! let agent_id = AgentID::generate().unwrap();
//! let mount_id = MountID::generate().unwrap();
//!
//! let transaction = Transaction {
//!     agent_id,
//!     context_seq_no: 1,
//!     transaction_seq_no: 1,
//!     msgs: vec![
//!         MessageParam::user("Hello"),
//!         MessageParam::assistant("Hi there!"),
//!     ],
//!     writes: vec![FileWrite {
//!         mount: mount_id,
//!         path: "/notes.txt".to_string(),
//!         data: "Meeting notes...".to_string(),
//!     }],
//! };
//!
//! // Validate transaction invariants
//! transaction.check_invariants().unwrap();
//! ```
//!
//! ### Chunking Large Transactions
//!
//! ```rust
//! use cetk::{Transaction, CHUNK_SIZE_LIMIT};
//!
//! # fn example() -> Result<(), cetk::TransactionSerializationError> {
//! # let transaction = Transaction::default();
//! // Transactions exceeding CHUNK_SIZE_LIMIT are automatically chunked
//! let chunks = transaction.chunk_transaction()?;
//!
//! // Chunks can be reassembled back to the original transaction
//! let reassembled = Transaction::from_chunks(chunks).unwrap();
//! # Ok(())
//! # }
//! ```
//!
//! ## Error Types
//!
//! The module provides comprehensive error types for different failure scenarios:
//!
//! - [`TransactionSerializationError`]: JSON serialization failures
//! - [`FromChunksError`]: Chunk reassembly failures
//! - [`ChunkSizeExceededError`]: Individual item size violations
//! - [`InvariantViolation`]: Data integrity violations
//!
//! ## Invariant Checking
//!
//! Transactions automatically validate:
//! - File write sizes don't exceed chunk limits
//! - Messages don't have consecutive duplicate roles (conversation flow)
//! - All required fields are properly populated

use claudius::{MessageParam, MessageRole};

use crate::{AgentID, CHUNK_SIZE_LIMIT, MountID};

////////////////////////////////////////////// Errors //////////////////////////////////////////////

/// Error that occurs during transaction serialization operations.
#[derive(Debug)]
pub struct TransactionSerializationError {
    /// The agent ID associated with the failed serialization
    pub agent_id: AgentID,
    /// The context sequence number where serialization failed
    pub context_seq_no: u32,
    /// The transaction sequence number where serialization failed
    pub transaction_seq_no: u64,
    /// Description of the operation that failed
    pub operation: String,
    /// The underlying JSON serialization error
    pub error: serde_json::Error,
}

impl std::fmt::Display for TransactionSerializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Serialization error in {} for transaction {}/{}/{}: {}",
            self.operation, self.agent_id, self.context_seq_no, self.transaction_seq_no, self.error
        )
    }
}

impl std::error::Error for TransactionSerializationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Error that occurs when reconstructing a transaction from chunks.
#[derive(Debug)]
pub enum FromChunksError {
    /// No chunks provided to reconstruct transaction.
    NoChunks,
    /// Chunks belong to different transactions.
    MismatchedTransaction {
        /// The expected agent ID
        agent_id: AgentID,
        /// The expected context sequence number
        context_seq_no: u32,
        /// The expected transaction sequence number
        transaction_seq_no: u64,
    },
    /// Missing chunks in the sequence.
    MissingChunks {
        /// The agent ID for the incomplete transaction
        agent_id: AgentID,
        /// The context sequence number for the incomplete transaction
        context_seq_no: u32,
        /// The transaction sequence number for the incomplete transaction
        transaction_seq_no: u64,
        /// The expected number of chunks
        expected: u32,
        /// The actual number of chunks received
        actual: u32,
    },
    /// Extra chunks beyond what total_chunks indicates.
    ExtraChunks {
        /// The agent ID for the transaction with extra chunks
        agent_id: AgentID,
        /// The context sequence number for the transaction with extra chunks
        context_seq_no: u32,
        /// The transaction sequence number for the transaction with extra chunks
        transaction_seq_no: u64,
        /// The expected number of chunks
        expected: u32,
        /// The actual number of chunks received
        actual: u32,
    },
    /// Chunk sequence numbers are not consecutive.
    InvalidSequence {
        /// The agent ID for the transaction with invalid sequence
        agent_id: AgentID,
        /// The context sequence number for the transaction with invalid sequence
        context_seq_no: u32,
        /// The transaction sequence number for the transaction with invalid sequence
        transaction_seq_no: u64,
        /// The expected chunk sequence number
        expected: u32,
        /// The actual chunk sequence number found
        actual: u32,
    },
    /// Failed to deserialize the reconstructed data.
    Deserialization {
        /// The agent ID for the failed deserialization
        agent_id: AgentID,
        /// The context sequence number for the failed deserialization
        context_seq_no: u32,
        /// The transaction sequence number for the failed deserialization
        transaction_seq_no: u64,
        /// The underlying deserialization error
        error: serde_json::Error,
    },
}

impl std::fmt::Display for FromChunksError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FromChunksError::NoChunks => write!(f, "No chunks provided to reconstruct transaction"),
            FromChunksError::MismatchedTransaction {
                agent_id,
                context_seq_no,
                transaction_seq_no,
            } => {
                write!(
                    f,
                    "Chunks belong to different transactions than {}/{}/{}",
                    agent_id, context_seq_no, transaction_seq_no
                )
            }
            FromChunksError::MissingChunks {
                agent_id,
                context_seq_no,
                transaction_seq_no,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Missing chunks for transaction {}/{}/{}: expected {}, got {}",
                    agent_id, context_seq_no, transaction_seq_no, expected, actual
                )
            }
            FromChunksError::ExtraChunks {
                agent_id,
                context_seq_no,
                transaction_seq_no,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Extra chunks for transaction {}/{}/{}: expected {}, got {}",
                    agent_id, context_seq_no, transaction_seq_no, expected, actual
                )
            }
            FromChunksError::InvalidSequence {
                agent_id,
                context_seq_no,
                transaction_seq_no,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Invalid chunk sequence for transaction {}/{}/{}: expected {}, got {}",
                    agent_id, context_seq_no, transaction_seq_no, expected, actual
                )
            }
            FromChunksError::Deserialization {
                agent_id,
                context_seq_no,
                transaction_seq_no,
                error,
            } => {
                write!(
                    f,
                    "Failed to deserialize transaction {}/{}/{}: {}",
                    agent_id, context_seq_no, transaction_seq_no, error
                )
            }
        }
    }
}

impl std::error::Error for FromChunksError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FromChunksError::Deserialization { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// Error indicating that an item exceeds the maximum chunk size.
#[derive(Debug)]
pub struct ChunkSizeExceededError {
    /// Description of the type of item that exceeded the limit
    pub item_type: String,
    /// The actual size in bytes of the oversized item
    pub actual_size: usize,
    /// The maximum allowed size limit in bytes
    pub limit: usize,
}

impl std::fmt::Display for ChunkSizeExceededError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} size {} exceeds limit {}",
            self.item_type, self.actual_size, self.limit
        )
    }
}

impl std::error::Error for ChunkSizeExceededError {}

/// Violations of transaction invariants that indicate invalid state.
#[derive(Debug)]
pub enum InvariantViolation {
    /// An item exceeds the maximum allowed chunk size.
    ChunkSizeExceeded(ChunkSizeExceededError),
    /// Two consecutive messages have the same role, which violates conversation flow.
    /// This typically indicates a problem with message sequencing or role assignment.
    ConsecutiveMessagesWithSameRole {
        /// The agent ID where the violation occurred
        agent_id: AgentID,
        /// The context sequence number where the violation occurred
        context_seq_no: u32,
        /// The transaction sequence number where the violation occurred
        transaction_seq_no: u64,
        /// The role that appears consecutively
        role: MessageRole,
        /// The positions (indices) of the consecutive messages with same role
        positions: (usize, usize),
    },
}

impl std::fmt::Display for InvariantViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InvariantViolation::ChunkSizeExceeded(err) => write!(f, "Chunk size exceeded: {}", err),
            InvariantViolation::ConsecutiveMessagesWithSameRole {
                agent_id,
                context_seq_no,
                transaction_seq_no,
                role,
                positions,
            } => {
                write!(
                    f,
                    "Consecutive messages with same role {:?} at positions {} and {} in transaction {}/{}/{}",
                    role, positions.0, positions.1, agent_id, context_seq_no, transaction_seq_no
                )
            }
        }
    }
}

impl std::error::Error for InvariantViolation {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            InvariantViolation::ChunkSizeExceeded(err) => Some(err),
            _ => None,
        }
    }
}

//////////////////////////////////////////// Transaction ///////////////////////////////////////////

/// A Transaction contains a transaction ID, some application data (usually a pointer to the
/// application state elsewhere, but it could be a state transition for rolling up in a state
/// machine), some messages to append to the conversation, and some filesystem writes.
///
/// Transactions represent atomic units of work within an agent's context. Each transaction
/// is uniquely identified by its agent_id, context_seq_no, and transaction_seq_no.
/// Transactions can be chunked if they exceed size limits and later reassembled.
///
/// # Examples
///
/// ```rust
/// use cetk::{Transaction, AgentID, FileWrite, MountID};
/// use claudius::{MessageParam, MessageRole};
///
/// let agent_id = AgentID::generate().unwrap();
/// let mount_id = MountID::generate().unwrap();
///
/// let transaction = Transaction {
///     agent_id,
///     context_seq_no: 1,
///     transaction_seq_no: 1,
///     msgs: vec![MessageParam {
///         role: MessageRole::User,
///         content: "Hello".into(),
///     }],
///     writes: vec![FileWrite {
///         mount: mount_id,
///         path: "/test.txt".to_string(),
///         data: "test content".to_string(),
///     }],
/// };
/// ```
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct Transaction {
    /// The unique identifier of the agent that created this transaction
    pub agent_id: AgentID,
    /// The sequence number of the context containing this transaction
    pub context_seq_no: u32,
    /// The sequence number of this transaction within its context
    pub transaction_seq_no: u64,
    /// Messages to be added to the conversation as part of this transaction
    pub msgs: Vec<MessageParam>,
    /// File system writes to be performed as part of this transaction
    pub writes: Vec<FileWrite>,
}

impl Transaction {
    /// Iterate over the messages of this transaction.
    pub fn messages(&self) -> impl DoubleEndedIterator<Item = MessageParam> + '_ {
        self.msgs.iter().cloned()
    }

    /// Chunk a transaction if it exceeds the size limit.
    pub fn chunk_transaction(
        &self,
    ) -> Result<Vec<TransactionChunk>, TransactionSerializationError> {
        let mut serialized =
            serde_json::to_string(self).map_err(|e| TransactionSerializationError {
                agent_id: self.agent_id,
                context_seq_no: self.context_seq_no,
                transaction_seq_no: self.transaction_seq_no,
                operation: "serialize".to_string(),
                error: e,
            })?;

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
            // Find a safe UTF-8 boundary within the chunk size limit
            let chunk_end = if serialized.len() <= CHUNK_SIZE_LIMIT {
                serialized.len()
            } else {
                // Find the last character boundary within the limit
                let mut end = CHUNK_SIZE_LIMIT;
                while end > 0 && !serialized.is_char_boundary(end) {
                    end -= 1;
                }
                // If we can't find a boundary, take at least one character to avoid infinite loop
                if end == 0 {
                    serialized.chars().next().unwrap().len_utf8()
                } else {
                    end
                }
            };

            let chunk = serialized[..chunk_end].to_string();
            serialized = serialized[chunk_end..].to_string();

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

    /// Check transaction invariants to ensure data integrity.
    ///
    /// This method validates:
    /// - File write data doesn't exceed chunk size limits
    /// - Messages don't have consecutive same roles (which violates conversation flow)
    ///
    /// # Errors
    ///
    /// Returns `InvariantViolation` if any invariant is violated.
    pub fn check_invariants(&self) -> Result<(), InvariantViolation> {
        // Check file write size limits
        for w in self.writes.iter() {
            if w.data.len() >= CHUNK_SIZE_LIMIT {
                return Err(InvariantViolation::ChunkSizeExceeded(
                    ChunkSizeExceededError {
                        item_type: "File write".to_string(),
                        actual_size: w.data.len(),
                        limit: CHUNK_SIZE_LIMIT,
                    },
                ));
            }
        }

        // Check for consecutive messages with same role
        // This is important because conversation flows should alternate between roles
        // (e.g., user -> assistant -> user), and consecutive same roles often indicate
        // a bug in message construction or merging logic.
        for (i, window) in self.msgs.windows(2).enumerate() {
            if window[0].role == window[1].role {
                return Err(InvariantViolation::ConsecutiveMessagesWithSameRole {
                    agent_id: self.agent_id,
                    context_seq_no: self.context_seq_no,
                    transaction_seq_no: self.transaction_seq_no,
                    role: window[0].role,
                    positions: (i, i + 1),
                });
            }
        }

        Ok(())
    }
}

impl Transaction {
    /// Reconstruct a transaction from a set of chunks.
    pub fn from_chunks(chunks: Vec<TransactionChunk>) -> Result<Transaction, FromChunksError> {
        if chunks.is_empty() {
            return Err(FromChunksError::NoChunks);
        }

        // Validate all chunks belong to the same transaction
        let first = chunks[0].clone();
        for chunk in &chunks {
            if chunk.agent_id != first.agent_id
                || chunk.context_seq_no != first.context_seq_no
                || chunk.transaction_seq_no != first.transaction_seq_no
                || chunk.total_chunks != first.total_chunks
            {
                return Err(FromChunksError::MismatchedTransaction {
                    agent_id: first.agent_id,
                    context_seq_no: first.context_seq_no,
                    transaction_seq_no: first.transaction_seq_no,
                });
            }
        }

        // Sort chunks by sequence number
        let mut sorted_chunks = chunks;
        sorted_chunks.sort_by_key(|chunk| chunk.chunk_seq_no);

        // Validate chunk sequence numbers are consecutive
        for (i, chunk) in sorted_chunks.iter().enumerate() {
            if chunk.chunk_seq_no != i as u32 {
                return Err(FromChunksError::InvalidSequence {
                    agent_id: first.agent_id,
                    context_seq_no: first.context_seq_no,
                    transaction_seq_no: first.transaction_seq_no,
                    expected: i as u32,
                    actual: chunk.chunk_seq_no,
                });
            }
        }

        // Validate we have the correct number of chunks (after confirming sequence is valid)
        if sorted_chunks.len() != first.total_chunks as usize {
            if sorted_chunks.len() > first.total_chunks as usize {
                return Err(FromChunksError::ExtraChunks {
                    agent_id: first.agent_id,
                    context_seq_no: first.context_seq_no,
                    transaction_seq_no: first.transaction_seq_no,
                    expected: first.total_chunks,
                    actual: sorted_chunks.len() as u32,
                });
            } else {
                return Err(FromChunksError::MissingChunks {
                    agent_id: first.agent_id,
                    context_seq_no: first.context_seq_no,
                    transaction_seq_no: first.transaction_seq_no,
                    expected: first.total_chunks,
                    actual: sorted_chunks.len() as u32,
                });
            }
        }

        // Reconstruct the serialized data
        let mut reconstructed_data = String::new();
        for chunk in sorted_chunks {
            reconstructed_data.push_str(&chunk.data);
        }

        // Deserialize back to Transaction
        let transaction: Transaction = serde_json::from_str(&reconstructed_data).map_err(|e| {
            FromChunksError::Deserialization {
                agent_id: first.agent_id,
                context_seq_no: first.context_seq_no,
                transaction_seq_no: first.transaction_seq_no,
                error: e,
            }
        })?;

        // Validate that the deserialized transaction matches the chunk metadata
        if transaction.agent_id != first.agent_id
            || transaction.context_seq_no != first.context_seq_no
            || transaction.transaction_seq_no != first.transaction_seq_no
        {
            return Err(FromChunksError::MismatchedTransaction {
                agent_id: transaction.agent_id,
                context_seq_no: transaction.context_seq_no,
                transaction_seq_no: transaction.transaction_seq_no,
            });
        }

        Ok(transaction)
    }
}

///////////////////////////////////////// TransactionChunk /////////////////////////////////////////

/// A chunk of a transaction when it exceeds the storage size limit.
///
/// When a transaction's serialized representation exceeds [`CHUNK_SIZE_LIMIT`],
/// it is split into multiple chunks for storage. Each chunk contains part of the
/// serialized transaction data along with metadata needed for reassembly.
///
/// Chunks are identified by their sequence number within the transaction and
/// can be reassembled in the correct order to reconstruct the original transaction.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct TransactionChunk {
    /// The agent ID that owns this transaction chunk
    pub agent_id: AgentID,
    /// The context sequence number this chunk belongs to
    pub context_seq_no: u32,
    /// The transaction sequence number this chunk belongs to
    pub transaction_seq_no: u64,
    /// The sequence number of this chunk within the transaction (0-based)
    pub chunk_seq_no: u32,
    /// The total number of chunks for the complete transaction
    pub total_chunks: u32,
    /// The serialized data contained in this chunk
    pub data: String,
}

///////////////////////////////////////////// FileWrite ////////////////////////////////////////////

/// Write the complete contents of data to the file at path on mount.
///
/// We assume files in the virtual filesystem should be small. FileWrites represent
/// complete overwrites of file content - they are not incremental patches but rather
/// full replacements of the file's contents.
///
/// # Examples
///
/// ```rust
/// use cetk::{FileWrite, MountID};
///
/// let mount_id = MountID::generate().unwrap();
/// let write = FileWrite {
///     mount: mount_id,
///     path: "/config/settings.json".to_string(),
///     data: r#"{"theme": "dark", "autosave": true}"#.to_string(),
/// };
/// ```
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct FileWrite {
    /// The mount identifier where the file should be written
    pub mount: MountID,
    /// The absolute path to the file within the mount
    pub path: String,
    /// The complete file content to write
    pub data: String,
}

/////////////////////////////////////////////// tests //////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use claudius::MessageParamContent;

    // Helper function to create a test AgentID
    fn test_agent_id() -> AgentID {
        AgentID::from_human_readable("agent:00000000-0000-0000-0000-000000000001").unwrap()
    }

    // Helper function to create a test MountID
    fn test_mount_id() -> MountID {
        MountID::from_human_readable("mount:00000000-0000-0000-0000-000000000001").unwrap()
    }

    // Helper function to create a test Transaction
    fn create_test_transaction() -> Transaction {
        Transaction {
            agent_id: test_agent_id(),
            context_seq_no: 1,
            transaction_seq_no: 42,
            msgs: vec![
                MessageParam {
                    role: MessageRole::User,
                    content: "Hello".into(),
                },
                MessageParam {
                    role: MessageRole::Assistant,
                    content: "Hi there!".into(),
                },
            ],
            writes: vec![FileWrite {
                mount: test_mount_id(),
                path: "test.txt".to_string(),
                data: "test data".to_string(),
            }],
        }
    }

    #[test]
    fn transaction_chunks_correctly_for_small_data() {
        let transaction = create_test_transaction();
        let chunks = transaction.chunk_transaction().unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_seq_no, 0);
        assert_eq!(chunks[0].total_chunks, 1);
        assert_eq!(chunks[0].agent_id, transaction.agent_id);
        assert_eq!(chunks[0].context_seq_no, transaction.context_seq_no);
        assert_eq!(chunks[0].transaction_seq_no, transaction.transaction_seq_no);
    }

    #[test]
    fn transaction_chunks_correctly_for_large_data() {
        let mut transaction = create_test_transaction();
        // Create a large message that will require chunking
        let large_content = "x".repeat(CHUNK_SIZE_LIMIT * 2);
        transaction.msgs.push(MessageParam::new(
            MessageParamContent::String(large_content),
            MessageRole::User,
        ));

        let chunks = transaction.chunk_transaction().unwrap();

        assert!(chunks.len() > 1);
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_seq_no, i as u32);
            assert_eq!(chunk.total_chunks, chunks.len() as u32);
            assert_eq!(chunk.agent_id, transaction.agent_id);
        }
    }

    #[test]
    fn transaction_reconstructs_correctly_from_chunks() {
        let original = create_test_transaction();
        let chunks = original.chunk_transaction().unwrap();
        let reconstructed = Transaction::from_chunks(chunks).unwrap();

        assert_eq!(original.agent_id, reconstructed.agent_id);
        assert_eq!(original.context_seq_no, reconstructed.context_seq_no);
        assert_eq!(
            original.transaction_seq_no,
            reconstructed.transaction_seq_no
        );
        assert_eq!(original.msgs.len(), reconstructed.msgs.len());
        assert_eq!(original.writes.len(), reconstructed.writes.len());
    }

    #[test]
    fn from_chunks_fails_with_no_chunks() {
        let result = Transaction::from_chunks(vec![]);
        assert!(matches!(result, Err(FromChunksError::NoChunks)));
    }

    #[test]
    fn from_chunks_fails_with_mismatched_transaction_ids() {
        let chunk1 = TransactionChunk {
            agent_id: test_agent_id(),
            context_seq_no: 1,
            transaction_seq_no: 1,
            chunk_seq_no: 0,
            total_chunks: 2,
            data: "{}".to_string(),
        };
        let mut chunk2 = chunk1.clone();
        chunk2.transaction_seq_no = 2; // Different transaction
        chunk2.chunk_seq_no = 1;

        let result = Transaction::from_chunks(vec![chunk1, chunk2]);
        assert!(matches!(
            result,
            Err(FromChunksError::MismatchedTransaction { .. })
        ));
    }

    #[test]
    fn from_chunks_fails_with_missing_chunks() {
        let chunk = TransactionChunk {
            agent_id: test_agent_id(),
            context_seq_no: 1,
            transaction_seq_no: 1,
            chunk_seq_no: 0,
            total_chunks: 2, // Claims 2 chunks but we only provide 1
            data: "{}".to_string(),
        };

        let result = Transaction::from_chunks(vec![chunk]);
        assert!(matches!(result, Err(FromChunksError::MissingChunks { .. })));
    }

    #[test]
    fn from_chunks_fails_with_extra_chunks() {
        // Create chunks with sequence numbers [0, 1, 2, 3, 4, 5, 6] but total_chunks = 6
        // This should fail with ExtraChunks because we have 7 chunks but total_chunks says 6
        let chunks = (0..=6)
            .map(|i| TransactionChunk {
                agent_id: test_agent_id(),
                context_seq_no: 1,
                transaction_seq_no: 42,
                chunk_seq_no: i,
                total_chunks: 6, // Claims 6 chunks but we provide 7 (0-6)
                data: format!("{{\"chunk\":{}}}", i),
            })
            .collect();

        match Transaction::from_chunks(chunks) {
            Err(FromChunksError::ExtraChunks {
                expected, actual, ..
            }) => {
                assert_eq!(expected, 6);
                assert_eq!(actual, 7);
            }
            _ => panic!("Expected ExtraChunks error"),
        }
    }

    #[test]
    fn check_invariants_passes_for_valid_transaction() {
        let transaction = create_test_transaction();
        assert!(transaction.check_invariants().is_ok());
    }

    #[test]
    fn check_invariants_fails_for_consecutive_same_role_messages() {
        let mut transaction = create_test_transaction();
        // Add another assistant message after the existing assistant message
        transaction
            .msgs
            .push(MessageParam::assistant("Another assistant message"));

        let result = transaction.check_invariants();
        assert!(matches!(
            result,
            Err(InvariantViolation::ConsecutiveMessagesWithSameRole { .. })
        ));

        if let Err(InvariantViolation::ConsecutiveMessagesWithSameRole {
            role, positions, ..
        }) = result
        {
            assert_eq!(role, MessageRole::Assistant); // The repeated role should be Assistant
            assert_eq!(positions, (1, 2)); // Positions of the consecutive same-role messages
        }
    }

    #[test]
    fn check_invariants_fails_for_oversized_file_write() {
        let mut transaction = create_test_transaction();
        transaction.writes[0].data = "x".repeat(CHUNK_SIZE_LIMIT + 1);

        let result = transaction.check_invariants();
        assert!(matches!(
            result,
            Err(InvariantViolation::ChunkSizeExceeded(_))
        ));
    }

    #[test]
    fn messages_iterator_works() {
        let transaction = create_test_transaction();
        let messages: Vec<_> = transaction.messages().collect();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, MessageRole::User);
        assert_eq!(messages[1].role, MessageRole::Assistant);
    }

    #[test]
    fn utf8_chunking_preserves_boundaries() {
        let mut transaction = create_test_transaction();
        // Create content with multi-byte UTF-8 characters near the chunk boundary
        let mut content = "a".repeat(CHUNK_SIZE_LIMIT - 10);
        content.push_str("🦀🦀🦀🦀🦀"); // Multi-byte emoji characters
        content.push_str(&"b".repeat(100));

        transaction.msgs.push(MessageParam::user(content));

        let chunks = transaction.chunk_transaction().unwrap();
        let reconstructed = Transaction::from_chunks(chunks).unwrap();

        // The reconstructed transaction should have valid UTF-8 and match the original
        assert_eq!(transaction.msgs.len(), reconstructed.msgs.len());
        for (orig, recon) in transaction.msgs.iter().zip(reconstructed.msgs.iter()) {
            assert_eq!(orig.role, recon.role);
            assert_eq!(orig.content, recon.content);
        }
    }

    #[test]
    fn check_invariants_valid_transaction() {
        let transaction = create_test_transaction();
        assert!(transaction.check_invariants().is_ok());
    }

    #[test]
    fn check_invariants_consecutive_messages_same_role_user_user() {
        let transaction = Transaction {
            agent_id: test_agent_id(),
            context_seq_no: 1,
            transaction_seq_no: 42,
            msgs: vec![
                MessageParam {
                    role: MessageRole::User,
                    content: "First user message".into(),
                },
                MessageParam {
                    role: MessageRole::User,
                    content: "Second user message".into(),
                },
            ],
            writes: vec![],
        };

        match transaction.check_invariants() {
            Err(InvariantViolation::ConsecutiveMessagesWithSameRole {
                agent_id,
                context_seq_no,
                transaction_seq_no,
                role,
                positions,
            }) => {
                assert_eq!(agent_id, test_agent_id());
                assert_eq!(context_seq_no, 1);
                assert_eq!(transaction_seq_no, 42);
                assert_eq!(role, MessageRole::User);
                assert_eq!(positions, (0, 1));
            }
            _ => panic!("Expected ConsecutiveMessagesWithSameRole error"),
        }
    }

    #[test]
    fn check_invariants_consecutive_messages_same_role_assistant_assistant() {
        let transaction = Transaction {
            agent_id: test_agent_id(),
            context_seq_no: 2,
            transaction_seq_no: 100,
            msgs: vec![
                MessageParam {
                    role: MessageRole::User,
                    content: "User message".into(),
                },
                MessageParam {
                    role: MessageRole::Assistant,
                    content: "First assistant message".into(),
                },
                MessageParam {
                    role: MessageRole::Assistant,
                    content: "Second assistant message".into(),
                },
            ],
            writes: vec![],
        };

        match transaction.check_invariants() {
            Err(InvariantViolation::ConsecutiveMessagesWithSameRole {
                agent_id,
                context_seq_no,
                transaction_seq_no,
                role,
                positions,
            }) => {
                assert_eq!(agent_id, test_agent_id());
                assert_eq!(context_seq_no, 2);
                assert_eq!(transaction_seq_no, 100);
                assert_eq!(role, MessageRole::Assistant);
                assert_eq!(positions, (1, 2));
            }
            _ => panic!("Expected ConsecutiveMessagesWithSameRole error"),
        }
    }

    #[test]
    fn check_invariants_multiple_consecutive_violations_reports_first() {
        let transaction = Transaction {
            agent_id: test_agent_id(),
            context_seq_no: 1,
            transaction_seq_no: 42,
            msgs: vec![
                MessageParam {
                    role: MessageRole::User,
                    content: "First user message".into(),
                },
                MessageParam {
                    role: MessageRole::User,
                    content: "Second user message".into(),
                },
                MessageParam {
                    role: MessageRole::User,
                    content: "Third user message".into(),
                },
            ],
            writes: vec![],
        };

        match transaction.check_invariants() {
            Err(InvariantViolation::ConsecutiveMessagesWithSameRole { positions, .. }) => {
                // Should report the first violation (0, 1), not (1, 2)
                assert_eq!(positions, (0, 1));
            }
            _ => panic!("Expected ConsecutiveMessagesWithSameRole error"),
        }
    }

    #[test]
    fn check_invariants_alternating_roles_valid() {
        let transaction = Transaction {
            agent_id: test_agent_id(),
            context_seq_no: 1,
            transaction_seq_no: 42,
            msgs: vec![
                MessageParam {
                    role: MessageRole::User,
                    content: "User 1".into(),
                },
                MessageParam {
                    role: MessageRole::Assistant,
                    content: "Assistant 1".into(),
                },
                MessageParam {
                    role: MessageRole::User,
                    content: "User 2".into(),
                },
                MessageParam {
                    role: MessageRole::Assistant,
                    content: "Assistant 2".into(),
                },
            ],
            writes: vec![],
        };

        assert!(transaction.check_invariants().is_ok());
    }

    #[test]
    fn check_invariants_single_message_valid() {
        let transaction = Transaction {
            agent_id: test_agent_id(),
            context_seq_no: 1,
            transaction_seq_no: 42,
            msgs: vec![MessageParam {
                role: MessageRole::User,
                content: "Only message".into(),
            }],
            writes: vec![],
        };

        assert!(transaction.check_invariants().is_ok());
    }

    #[test]
    fn check_invariants_empty_messages_valid() {
        let transaction = Transaction {
            agent_id: test_agent_id(),
            context_seq_no: 1,
            transaction_seq_no: 42,
            msgs: vec![],
            writes: vec![],
        };

        assert!(transaction.check_invariants().is_ok());
    }

    #[test]
    fn check_invariants_file_write_exceeds_chunk_size() {
        let large_data = "x".repeat(CHUNK_SIZE_LIMIT);
        let transaction = Transaction {
            agent_id: test_agent_id(),
            context_seq_no: 1,
            transaction_seq_no: 42,
            msgs: vec![],
            writes: vec![FileWrite {
                mount: test_mount_id(),
                path: "large.txt".to_string(),
                data: large_data.clone(),
            }],
        };

        match transaction.check_invariants() {
            Err(InvariantViolation::ChunkSizeExceeded(error)) => {
                assert_eq!(error.item_type, "File write");
                assert_eq!(error.actual_size, large_data.len());
                assert_eq!(error.limit, CHUNK_SIZE_LIMIT);
            }
            _ => panic!("Expected ChunkSizeExceeded error"),
        }
    }

    #[test]
    fn check_invariants_file_write_at_chunk_size_limit_valid() {
        let data_at_limit = "x".repeat(CHUNK_SIZE_LIMIT - 1);
        let transaction = Transaction {
            agent_id: test_agent_id(),
            context_seq_no: 1,
            transaction_seq_no: 42,
            msgs: vec![],
            writes: vec![FileWrite {
                mount: test_mount_id(),
                path: "at_limit.txt".to_string(),
                data: data_at_limit,
            }],
        };

        assert!(transaction.check_invariants().is_ok());
    }

    #[test]
    fn check_invariants_multiple_file_writes_one_exceeds() {
        let large_data = "x".repeat(CHUNK_SIZE_LIMIT);
        let transaction = Transaction {
            agent_id: test_agent_id(),
            context_seq_no: 1,
            transaction_seq_no: 42,
            msgs: vec![],
            writes: vec![
                FileWrite {
                    mount: test_mount_id(),
                    path: "small.txt".to_string(),
                    data: "small".to_string(),
                },
                FileWrite {
                    mount: test_mount_id(),
                    path: "large.txt".to_string(),
                    data: large_data,
                },
            ],
        };

        match transaction.check_invariants() {
            Err(InvariantViolation::ChunkSizeExceeded(_)) => {
                // Expected - should catch the first file that exceeds the limit
            }
            _ => panic!("Expected ChunkSizeExceeded error"),
        }
    }

    #[test]
    fn chunk_transaction_small_transaction_single_chunk() {
        let transaction = create_test_transaction();
        let chunks = transaction.chunk_transaction().unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].agent_id, transaction.agent_id);
        assert_eq!(chunks[0].context_seq_no, transaction.context_seq_no);
        assert_eq!(chunks[0].transaction_seq_no, transaction.transaction_seq_no);
        assert_eq!(chunks[0].chunk_seq_no, 0);
        assert_eq!(chunks[0].total_chunks, 1);

        // Verify the data is the serialized transaction
        let deserialized: Transaction = serde_json::from_str(&chunks[0].data).unwrap();
        assert_eq!(deserialized.agent_id, transaction.agent_id);
        assert_eq!(deserialized.msgs.len(), transaction.msgs.len());
    }

    #[test]
    fn chunk_transaction_large_transaction_multiple_chunks() {
        // Create a transaction with very large message content to exceed CHUNK_SIZE_LIMIT
        let large_content = "x".repeat(CHUNK_SIZE_LIMIT);
        let transaction = Transaction {
            agent_id: test_agent_id(),
            context_seq_no: 1,
            transaction_seq_no: 42,
            msgs: vec![
                MessageParam {
                    role: MessageRole::User,
                    content: large_content.into(),
                },
                MessageParam {
                    role: MessageRole::Assistant,
                    content: "Response".into(),
                },
            ],
            writes: vec![],
        };

        let chunks = transaction.chunk_transaction().unwrap();

        assert!(chunks.len() > 1);

        // Verify all chunks have consistent metadata
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.agent_id, transaction.agent_id);
            assert_eq!(chunk.context_seq_no, transaction.context_seq_no);
            assert_eq!(chunk.transaction_seq_no, transaction.transaction_seq_no);
            assert_eq!(chunk.chunk_seq_no, i as u32);
            assert_eq!(chunk.total_chunks, chunks.len() as u32);
            assert!(chunk.data.len() <= CHUNK_SIZE_LIMIT);
        }

        // Verify we can reconstruct the original data
        let reconstructed_data: String = chunks.iter().map(|c| c.data.as_str()).collect();
        let deserialized: Transaction = serde_json::from_str(&reconstructed_data).unwrap();
        assert_eq!(deserialized.agent_id, transaction.agent_id);
        assert_eq!(deserialized.msgs.len(), transaction.msgs.len());
    }

    #[test]
    fn chunk_transaction_exactly_at_limit() {
        // Create a transaction that serializes to exactly CHUNK_SIZE_LIMIT bytes
        let mut transaction = create_test_transaction();

        // Adjust the content to get close to the limit
        let serialized_base = serde_json::to_string(&transaction).unwrap();
        let needed_padding = CHUNK_SIZE_LIMIT - serialized_base.len();

        if needed_padding > 0 {
            transaction.msgs[0].content = format!("Hello{}", "x".repeat(needed_padding - 5)).into();
        }

        let chunks = transaction.chunk_transaction().unwrap();

        if serde_json::to_string(&transaction).unwrap().len() <= CHUNK_SIZE_LIMIT {
            assert_eq!(chunks.len(), 1);
        } else {
            assert!(chunks.len() > 1);
        }
    }

    #[test]
    fn from_chunks_single_chunk() {
        let original = create_test_transaction();
        let chunks = original.chunk_transaction().unwrap();

        let reconstructed = Transaction::from_chunks(chunks).unwrap();

        assert_eq!(reconstructed.agent_id, original.agent_id);
        assert_eq!(reconstructed.context_seq_no, original.context_seq_no);
        assert_eq!(
            reconstructed.transaction_seq_no,
            original.transaction_seq_no
        );
        assert_eq!(reconstructed.msgs.len(), original.msgs.len());
        assert_eq!(reconstructed.writes.len(), original.writes.len());
    }

    #[test]
    fn from_chunks_multiple_chunks() {
        // Create a large transaction that will be chunked
        let large_content = "x".repeat(CHUNK_SIZE_LIMIT);
        let original = Transaction {
            agent_id: test_agent_id(),
            context_seq_no: 5,
            transaction_seq_no: 123,
            msgs: vec![MessageParam {
                role: MessageRole::User,
                content: large_content.into(),
            }],
            writes: vec![],
        };

        let chunks = original.chunk_transaction().unwrap();
        assert!(chunks.len() > 1); // Verify we actually have multiple chunks

        let reconstructed = Transaction::from_chunks(chunks).unwrap();

        assert_eq!(reconstructed.agent_id, original.agent_id);
        assert_eq!(reconstructed.context_seq_no, original.context_seq_no);
        assert_eq!(
            reconstructed.transaction_seq_no,
            original.transaction_seq_no
        );
        assert_eq!(reconstructed.msgs.len(), original.msgs.len());
    }

    #[test]
    fn from_chunks_unordered_chunks() {
        let large_content = "x".repeat(CHUNK_SIZE_LIMIT);
        let original = Transaction {
            agent_id: test_agent_id(),
            context_seq_no: 1,
            transaction_seq_no: 42,
            msgs: vec![MessageParam {
                role: MessageRole::User,
                content: large_content.into(),
            }],
            writes: vec![],
        };

        let mut chunks = original.chunk_transaction().unwrap();
        assert!(chunks.len() > 1); // Ensure multiple chunks

        // Reverse the order of chunks
        chunks.reverse();

        let reconstructed = Transaction::from_chunks(chunks).unwrap();
        assert_eq!(reconstructed.agent_id, original.agent_id);
    }

    #[test]
    fn from_chunks_empty_chunks() {
        match Transaction::from_chunks(vec![]) {
            Err(FromChunksError::NoChunks) => {
                // Expected
            }
            _ => panic!("Expected NoChunks error"),
        }
    }

    #[test]
    fn from_chunks_mismatched_agent_id() {
        let original = create_test_transaction();
        let mut chunks = original.chunk_transaction().unwrap();

        // Modify one chunk to have a different agent_id
        let different_agent_id =
            AgentID::from_human_readable("agent:00000000-0000-0000-0000-000000000002").unwrap();
        chunks[0].agent_id = different_agent_id;

        match Transaction::from_chunks(chunks) {
            Err(FromChunksError::MismatchedTransaction {
                agent_id,
                context_seq_no,
                transaction_seq_no,
            }) => {
                assert_eq!(agent_id, original.agent_id);
                assert_eq!(context_seq_no, original.context_seq_no);
                assert_eq!(transaction_seq_no, original.transaction_seq_no);
            }
            _ => panic!("Expected MismatchedTransaction error"),
        }
    }

    #[test]
    fn from_chunks_mismatched_context_seq_no() {
        let original = create_test_transaction();
        let mut chunks = original.chunk_transaction().unwrap();

        // Modify one chunk to have a different context_seq_no
        chunks[0].context_seq_no = 999;

        match Transaction::from_chunks(chunks) {
            Err(FromChunksError::MismatchedTransaction { .. }) => {
                // Expected
            }
            _ => panic!("Expected MismatchedTransaction error"),
        }
    }

    #[test]
    fn from_chunks_mismatched_transaction_seq_no() {
        let original = create_test_transaction();
        let mut chunks = original.chunk_transaction().unwrap();

        // Modify one chunk to have a different transaction_seq_no
        chunks[0].transaction_seq_no = 999;

        match Transaction::from_chunks(chunks) {
            Err(FromChunksError::MismatchedTransaction { .. }) => {
                // Expected
            }
            _ => panic!("Expected MismatchedTransaction error"),
        }
    }

    #[test]
    fn from_chunks_mismatched_total_chunks() {
        let original = create_test_transaction();
        let mut chunks = original.chunk_transaction().unwrap();

        // Modify one chunk to have a different total_chunks
        chunks[0].total_chunks = 999;

        match Transaction::from_chunks(chunks) {
            Err(FromChunksError::MissingChunks {
                expected, actual, ..
            }) => {
                assert_eq!(expected, 999);
                assert_eq!(actual, 1);
            }
            _ => panic!("Expected MissingChunks error"),
        }
    }

    #[test]
    fn from_chunks_missing_chunks() {
        let large_content = "x".repeat(CHUNK_SIZE_LIMIT);
        let original = Transaction {
            agent_id: test_agent_id(),
            context_seq_no: 1,
            transaction_seq_no: 42,
            msgs: vec![MessageParam {
                role: MessageRole::User,
                content: large_content.into(),
            }],
            writes: vec![],
        };

        let mut chunks = original.chunk_transaction().unwrap();
        assert!(chunks.len() > 1); // Ensure multiple chunks

        // Remove one chunk
        let expected_count = chunks.len();
        chunks.pop();

        match Transaction::from_chunks(chunks) {
            Err(FromChunksError::MissingChunks {
                expected, actual, ..
            }) => {
                assert_eq!(expected, expected_count as u32);
                assert_eq!(actual, (expected_count - 1) as u32);
            }
            _ => panic!("Expected MissingChunks error"),
        }
    }

    #[test]
    fn from_chunks_invalid_sequence_gap() {
        let large_content = "x".repeat(CHUNK_SIZE_LIMIT * 3);
        let original = Transaction {
            agent_id: test_agent_id(),
            context_seq_no: 1,
            transaction_seq_no: 42,
            msgs: vec![MessageParam {
                role: MessageRole::User,
                content: large_content.into(),
            }],
            writes: vec![],
        };

        let mut chunks = original.chunk_transaction().unwrap();
        assert!(chunks.len() > 2); // Need at least 3 chunks for this test

        // Create a gap in sequence numbers
        chunks[1].chunk_seq_no = 5;

        match Transaction::from_chunks(chunks) {
            Err(FromChunksError::InvalidSequence {
                expected, actual, ..
            }) => {
                assert_eq!(expected, 1);
                assert_eq!(actual, 2);
            }
            _ => panic!("Expected InvalidSequence error"),
        }
    }

    #[test]
    fn from_chunks_invalid_sequence_duplicate() {
        let large_content = "x".repeat(CHUNK_SIZE_LIMIT);
        let original = Transaction {
            agent_id: test_agent_id(),
            context_seq_no: 1,
            transaction_seq_no: 42,
            msgs: vec![MessageParam {
                role: MessageRole::User,
                content: large_content.into(),
            }],
            writes: vec![],
        };

        let mut chunks = original.chunk_transaction().unwrap();
        assert!(chunks.len() > 1); // Ensure multiple chunks

        // Duplicate a sequence number
        chunks[1].chunk_seq_no = 0;

        match Transaction::from_chunks(chunks) {
            Err(FromChunksError::InvalidSequence {
                expected, actual, ..
            }) => {
                assert_eq!(expected, 1);
                assert_eq!(actual, 0);
            }
            _ => panic!("Expected InvalidSequence error"),
        }
    }

    #[test]
    fn from_chunks_corrupted_json_data() {
        let original = create_test_transaction();
        let mut chunks = original.chunk_transaction().unwrap();

        // Corrupt the JSON data
        chunks[0].data = "invalid json {".to_string();

        match Transaction::from_chunks(chunks) {
            Err(FromChunksError::Deserialization { .. }) => {
                // Expected
            }
            _ => panic!("Expected Deserialization error"),
        }
    }

    #[test]
    fn messages_iterator_forward() {
        let transaction = create_test_transaction();
        let messages: Vec<MessageParam> = transaction.messages().collect();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, MessageRole::User);
        assert_eq!(messages[1].role, MessageRole::Assistant);
    }

    #[test]
    fn messages_iterator_reverse() {
        let transaction = create_test_transaction();
        let messages: Vec<MessageParam> = transaction.messages().rev().collect();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, MessageRole::Assistant);
        assert_eq!(messages[1].role, MessageRole::User);
    }

    #[test]
    fn messages_iterator_empty() {
        let transaction = Transaction {
            agent_id: test_agent_id(),
            context_seq_no: 1,
            transaction_seq_no: 42,
            msgs: vec![],
            writes: vec![],
        };

        let messages: Vec<MessageParam> = transaction.messages().collect();
        assert_eq!(messages.len(), 0);
    }

    #[test]
    fn transaction_with_unicode_content() {
        let transaction = Transaction {
            agent_id: test_agent_id(),
            context_seq_no: 1,
            transaction_seq_no: 42,
            msgs: vec![
                MessageParam {
                    role: MessageRole::User,
                    content: "Hello 🌍 世界 🚀".into(),
                },
                MessageParam {
                    role: MessageRole::Assistant,
                    content: "こんにちは! 🎉".into(),
                },
            ],
            writes: vec![],
        };

        assert!(transaction.check_invariants().is_ok());

        let chunks = transaction.chunk_transaction().unwrap();
        let reconstructed = Transaction::from_chunks(chunks).unwrap();

        // Check the content by pattern matching on the enum
        match &reconstructed.msgs[0].content {
            claudius::MessageParamContent::String(s) => assert_eq!(s, "Hello 🌍 世界 🚀"),
            _ => panic!("Expected String content"),
        }
        match &reconstructed.msgs[1].content {
            claudius::MessageParamContent::String(s) => assert_eq!(s, "こんにちは! 🎉"),
            _ => panic!("Expected String content"),
        }
    }

    #[test]
    fn transaction_zero_sequence_numbers() {
        let transaction = Transaction {
            agent_id: test_agent_id(),
            context_seq_no: 0,
            transaction_seq_no: 0,
            msgs: vec![],
            writes: vec![],
        };

        assert!(transaction.check_invariants().is_ok());

        let chunks = transaction.chunk_transaction().unwrap();
        let reconstructed = Transaction::from_chunks(chunks).unwrap();

        assert_eq!(reconstructed.context_seq_no, 0);
        assert_eq!(reconstructed.transaction_seq_no, 0);
    }

    #[test]
    fn transaction_max_sequence_numbers() {
        let transaction = Transaction {
            agent_id: test_agent_id(),
            context_seq_no: u32::MAX,
            transaction_seq_no: u64::MAX,
            msgs: vec![],
            writes: vec![],
        };

        assert!(transaction.check_invariants().is_ok());

        let chunks = transaction.chunk_transaction().unwrap();
        let reconstructed = Transaction::from_chunks(chunks).unwrap();

        assert_eq!(reconstructed.context_seq_no, u32::MAX);
        assert_eq!(reconstructed.transaction_seq_no, u64::MAX);
    }
}
