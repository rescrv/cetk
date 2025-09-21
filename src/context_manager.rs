use std::collections::HashMap;

use chromadb::{
    MetadataValue,
    collection::{ChromaCollection, CollectionEntries, GetOptions},
    filters::{self, MetadataFilter},
};

use crate::{AgentID, EmbeddingService, FileWrite, Transaction, TransactionChunk, TransactionID};
use claudius::MessageParam;

////////////////////////////////////////////// Errors //////////////////////////////////////////////

/// Error that can occur during context management operations.
#[derive(Debug)]
pub enum ContextManagerError {
    /// Error connecting to or communicating with ChromaDB.
    ChromaError(String),
    /// Error chunking the transaction.
    ChunkingError(crate::TransactionSerializationError),
    /// GUID generation failed.
    GuidError,
    /// Error generating embeddings.
    EmbeddingError(anyhow::Error),
    /// Error loading agent data.
    LoadAgentError(String),
}

impl std::fmt::Display for ContextManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContextManagerError::ChromaError(e) => write!(f, "ChromaDB error: {}", e),
            ContextManagerError::ChunkingError(e) => {
                write!(f, "Transaction chunking error: {}", e)
            }
            ContextManagerError::GuidError => write!(f, "Failed to generate GUID"),
            ContextManagerError::EmbeddingError(e) => write!(f, "Embedding error: {}", e),
            ContextManagerError::LoadAgentError(e) => write!(f, "Agent loading error: {}", e),
        }
    }
}

impl std::error::Error for ContextManagerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ContextManagerError::ChromaError(_) => None,
            ContextManagerError::ChunkingError(e) => Some(e),
            ContextManagerError::GuidError => None,
            ContextManagerError::EmbeddingError(e) => Some(e.as_ref()),
            ContextManagerError::LoadAgentError(_) => None,
        }
    }
}

impl From<crate::TransactionSerializationError> for ContextManagerError {
    fn from(error: crate::TransactionSerializationError) -> Self {
        ContextManagerError::ChunkingError(error)
    }
}

impl From<anyhow::Error> for ContextManagerError {
    fn from(error: anyhow::Error) -> Self {
        ContextManagerError::EmbeddingError(error)
    }
}

//////////////////////////////////////// Helper Functions /////////////////////////////////////////

/// Generate a consistent chunk ID from transaction metadata.
/// Format: agent_id:context_seq_no:transaction_seq_no:chunk_seq_no
fn generate_chunk_id(
    agent_id: AgentID,
    context_seq_no: u32,
    transaction_seq_no: u64,
    chunk_seq_no: u32,
) -> String {
    format!(
        "{}:{}:{}:{}",
        agent_id, context_seq_no, transaction_seq_no, chunk_seq_no
    )
}

//////////////////////////////////////// Agent Data Structures ////////////////////////////////////////

/// Represents a complete context with its transactions for an agent
#[derive(Debug, Clone)]
pub struct AgentContext {
    pub agent_id: AgentID,
    pub context_seq_no: u32,
    pub transactions: Vec<Transaction>,
}

/// Complete agent data with all contexts and transactions
#[derive(Debug, Clone)]
pub struct AgentData {
    pub agent_id: AgentID,
    pub contexts: Vec<AgentContext>,
}

impl AgentData {
    /// Get all transactions across all contexts, in order
    pub fn all_transactions(&self) -> Vec<&Transaction> {
        let mut transactions = Vec::new();
        for context in &self.contexts {
            transactions.extend(context.transactions.iter());
        }
        transactions
    }

    /// Get the latest context (highest context_seq_no)
    pub fn latest_context(&self) -> Option<&AgentContext> {
        self.contexts.iter().max_by_key(|c| c.context_seq_no)
    }

    /// Get a specific context by sequence number
    pub fn get_context(&self, context_seq_no: u32) -> Option<&AgentContext> {
        self.contexts
            .iter()
            .find(|c| c.context_seq_no == context_seq_no)
    }

    /// Start building the next transaction in the current context.
    ///
    /// This creates a fluent TransactionBuilder that automatically determines
    /// the correct sequence numbers for continuing in the latest context.
    pub fn next_transaction<'a>(
        &'a mut self,
        context_manager: &'a ContextManager,
    ) -> TransactionBuilder<'a> {
        TransactionBuilder::new_in_current_context(self, context_manager)
    }

    /// Start building the first transaction in a new context.
    ///
    /// This creates a fluent TransactionBuilder for starting a fresh context,
    /// typically used for major conversation restarts or compaction.
    pub fn new_context<'a>(
        &'a mut self,
        context_manager: &'a ContextManager,
    ) -> TransactionBuilder<'a> {
        TransactionBuilder::new_in_next_context(self, context_manager)
    }
}

//////////////////////////////////////// TransactionBuilder ////////////////////////////////////////

/// A fluent builder for creating the next transaction from existing agent data.
///
/// This builder automatically determines the correct sequence numbers and provides
/// a fluent API for adding messages and file writes.
pub struct TransactionBuilder<'a> {
    agent_data: &'a mut AgentData,
    context_manager: &'a ContextManager,
    context_seq_no: u32,
    transaction_seq_no: u64,
    msgs: Vec<MessageParam>,
    writes: Vec<FileWrite>,
}

impl<'a> TransactionBuilder<'a> {
    /// Create a new transaction builder for the next transaction in the current context
    fn new_in_current_context(
        agent_data: &'a mut AgentData,
        context_manager: &'a ContextManager,
    ) -> Self {
        let (context_seq_no, transaction_seq_no) =
            if let Some(latest_context) = agent_data.latest_context() {
                let next_transaction_seq = latest_context
                    .transactions
                    .iter()
                    .map(|t| t.transaction_seq_no)
                    .max()
                    .unwrap_or(0)
                    + 1;
                (latest_context.context_seq_no, next_transaction_seq)
            } else {
                // No contexts exist, start with context 1, transaction 1
                (1, 1)
            };

        TransactionBuilder {
            agent_data,
            context_manager,
            context_seq_no,
            transaction_seq_no,
            msgs: Vec::new(),
            writes: Vec::new(),
        }
    }

    /// Create a new transaction builder for the next transaction in a new context
    fn new_in_next_context(
        agent_data: &'a mut AgentData,
        context_manager: &'a ContextManager,
    ) -> Self {
        let next_context_seq = agent_data
            .contexts
            .iter()
            .map(|c| c.context_seq_no)
            .max()
            .unwrap_or(0)
            + 1;

        TransactionBuilder {
            agent_data,
            context_manager,
            context_seq_no: next_context_seq,
            transaction_seq_no: 1, // First transaction in new context
            msgs: Vec::new(),
            writes: Vec::new(),
        }
    }

    /// Add a message to this transaction
    pub fn message(mut self, message: MessageParam) -> Self {
        self.msgs.push(message);
        self
    }

    /// Add multiple messages to this transaction
    pub fn messages(mut self, messages: Vec<MessageParam>) -> Self {
        self.msgs.extend(messages);
        self
    }

    /// Add a file write to this transaction
    pub fn write_file<P: Into<String>, D: Into<String>>(
        mut self,
        mount: crate::MountID,
        path: P,
        data: D,
    ) -> Self {
        self.writes.push(FileWrite {
            mount,
            path: path.into(),
            data: data.into(),
        });
        self
    }

    /// Add multiple file writes to this transaction
    pub fn write_files(mut self, writes: Vec<FileWrite>) -> Self {
        self.writes.extend(writes);
        self
    }

    /// Complete the transaction builder and persist it to ChromaDB.
    ///
    /// This will:
    /// 1. Build the Transaction struct
    /// 2. Persist it to ChromaDB with verification
    /// 3. Update the AgentData with the new transaction
    /// 4. Return the persistence nonce
    pub async fn save(mut self) -> Result<String, ContextManagerError> {
        // Build the transaction
        let transaction = Transaction {
            agent_id: self.agent_data.agent_id,
            context_seq_no: self.context_seq_no,
            transaction_seq_no: self.transaction_seq_no,
            msgs: self.msgs.clone(),
            writes: self.writes.clone(),
        };

        // Persist to ChromaDB
        let nonce = self
            .context_manager
            .persist_transaction(&transaction)
            .await?;

        // Update the agent data
        self.update_agent_data(transaction);

        Ok(nonce)
    }

    /// Build the transaction without persisting it.
    ///
    /// This is useful for testing or when you want to inspect the transaction
    /// before persisting it.
    pub fn build(self) -> Transaction {
        Transaction {
            agent_id: self.agent_data.agent_id,
            context_seq_no: self.context_seq_no,
            transaction_seq_no: self.transaction_seq_no,
            msgs: self.msgs,
            writes: self.writes,
        }
    }

    /// Update the AgentData with the new transaction
    fn update_agent_data(&mut self, transaction: Transaction) {
        // Find or create the context
        if let Some(context) = self
            .agent_data
            .contexts
            .iter_mut()
            .find(|c| c.context_seq_no == self.context_seq_no)
        {
            // Add to existing context
            context.transactions.push(transaction);
            // Keep transactions sorted
            context.transactions.sort_by_key(|t| t.transaction_seq_no);
        } else {
            // Create new context
            let new_context = AgentContext {
                agent_id: self.agent_data.agent_id,
                context_seq_no: self.context_seq_no,
                transactions: vec![transaction],
            };
            self.agent_data.contexts.push(new_context);
            // Keep contexts sorted
            self.agent_data.contexts.sort_by_key(|c| c.context_seq_no);
        }
    }
}

//////////////////////////////////////// ContextManager ////////////////////////////////////////

/// Manages agent contexts and transaction persistence using ChromaDB collections.
///
/// The ContextManager handles:
/// 1. Agent context management and transaction persistence
/// 2. Atomically adds all transaction chunks to a chroma collection with nonce metadata
/// 3. Can verify transaction persistence by checking if chunk[0] has the expected nonce
pub struct ContextManager {
    collection: ChromaCollection,
    embedding_service: EmbeddingService,
}

impl ContextManager {
    /// Create a new ContextManager with a ChromaDB collection.
    pub fn new(collection: ChromaCollection) -> Result<Self, ContextManagerError> {
        let embedding_service = EmbeddingService::new()?;
        Ok(ContextManager {
            collection,
            embedding_service,
        })
    }

    /// Persist a transaction by chunking it and atomically storing all chunks with a nonce.
    ///
    /// Automatically verifies persistence before returning success.
    /// Returns the GUID that was used as the nonce if persistence and verification succeed.
    pub async fn persist_transaction(
        &self,
        transaction: &Transaction,
    ) -> Result<String, ContextManagerError> {
        // Generate a unique GUID for this persistence operation using TransactionID
        let nonce = TransactionID::generate()
            .ok_or(ContextManagerError::GuidError)?
            .to_string();

        // Chunk the transaction
        let chunks = transaction.chunk_transaction()?;

        // Create IDs for the chunks
        let chunk_ids: Vec<String> = chunks
            .iter()
            .map(|chunk| {
                generate_chunk_id(
                    chunk.agent_id,
                    chunk.context_seq_no,
                    chunk.transaction_seq_no,
                    chunk.chunk_seq_no,
                )
            })
            .collect();

        let chunk_id_refs: Vec<&str> = chunk_ids.iter().map(|s| s.as_str()).collect();

        // Create metadata with the nonce for each chunk
        let metadatas: Vec<HashMap<String, MetadataValue>> = chunks
            .iter()
            .map(|chunk| {
                let mut metadata = HashMap::new();
                metadata.insert("nonce".to_string(), MetadataValue::Str(nonce.clone()));
                metadata.insert(
                    "agent_id".to_string(),
                    MetadataValue::Str(chunk.agent_id.to_string()),
                );
                metadata.insert(
                    "context_seq_no".to_string(),
                    MetadataValue::Int(chunk.context_seq_no as i64),
                );
                metadata.insert(
                    "transaction_seq_no".to_string(),
                    MetadataValue::Int(chunk.transaction_seq_no as i64),
                );
                metadata.insert(
                    "chunk_seq_no".to_string(),
                    MetadataValue::Int(chunk.chunk_seq_no as i64),
                );
                metadata.insert(
                    "total_chunks".to_string(),
                    MetadataValue::Int(chunk.total_chunks as i64),
                );
                metadata
            })
            .collect();

        // Create documents from chunk data
        let documents: Vec<&str> = chunks.iter().map(|chunk| chunk.data.as_str()).collect();

        // Generate real embeddings for each chunk using the embedding service
        let embeddings = self.embedding_service.embed(&documents)?;

        let collection_entries = CollectionEntries {
            ids: chunk_id_refs,
            metadatas: Some(metadatas),
            documents: Some(documents),
            embeddings: Some(embeddings),
        };

        // Atomically add all chunks to the collection
        self.collection
            .add(collection_entries, None)
            .await
            .map_err(|e| ContextManagerError::ChromaError(e.to_string()))?;

        // Verify persistence before returning success
        let verification_successful = self.verify_persistence(transaction, &nonce).await?;
        if !verification_successful {
            return Err(ContextManagerError::ChromaError(
                "Transaction persistence verification failed".to_string(),
            ));
        }

        Ok(nonce)
    }

    /// Verify that a transaction was successfully persisted by checking chunk[0] for the expected nonce.
    ///
    /// Returns `true` if chunk[0] exists and has the specified nonce, `false` otherwise.
    pub async fn verify_persistence(
        &self,
        transaction: &Transaction,
        expected_nonce: &str,
    ) -> Result<bool, ContextManagerError> {
        // Construct the ID for chunk 0
        let chunk_0_id = generate_chunk_id(
            transaction.agent_id,
            transaction.context_seq_no,
            transaction.transaction_seq_no,
            0,
        );

        // Try to get chunk 0 from the collection
        let get_options = chromadb::collection::GetOptions::new()
            .ids(vec![chunk_0_id])
            .include(vec!["metadatas".to_string()]);

        let result = self
            .collection
            .get(get_options)
            .await
            .map_err(|e| ContextManagerError::ChromaError(e.to_string()))?;

        // Check if we got exactly one result
        if result.ids.len() != 1 {
            return Ok(false);
        }

        // Check if we have metadata and verify nonce
        if let Some(metadatas) = result.metadatas
            && let Some(Some(metadata)) = metadatas.first()
            && let Some(MetadataValue::Str(nonce_str)) = metadata.get("nonce")
        {
            return Ok(nonce_str == expected_nonce);
        }

        Ok(false)
    }

    /// Load all transaction data for a given agent, organizing it by contexts and transactions.
    ///
    /// This method:
    /// 1. Queries ChromaDB for all chunks belonging to the agent
    /// 2. Groups chunks by context and transaction
    /// 3. Assembles complete transactions from their chunks
    /// 4. Returns organized agent data with contexts and transactions
    pub async fn load_agent(&self, agent_id: AgentID) -> Result<AgentData, ContextManagerError> {
        // Query all chunks for this agent
        let agent_filter = filters::eq("agent_id", agent_id.to_string());
        let get_options = GetOptions::new()
            .where_metadata(agent_filter)
            .include(vec!["metadatas".to_string(), "documents".to_string()]);

        let result = self
            .collection
            .get(get_options)
            .await
            .map_err(|e| ContextManagerError::ChromaError(e.to_string()))?;

        // Convert ChromaDB result to TransactionChunks
        let chunks = self.convert_chroma_result_to_chunks(result)?;

        // Organize chunks by context and transaction, then assemble
        let agent_data = self.assemble_agent_data(agent_id, chunks)?;

        Ok(agent_data)
    }

    /// Convert ChromaDB GetResult to TransactionChunks
    fn convert_chroma_result_to_chunks(
        &self,
        result: chromadb::collection::GetResult,
    ) -> Result<Vec<TransactionChunk>, ContextManagerError> {
        let chromadb::collection::GetResult {
            ids,
            metadatas,
            documents,
            ..
        } = result;

        let Some(metadatas) = metadatas else {
            return Err(ContextManagerError::LoadAgentError(
                "No metadata returned from ChromaDB".to_string(),
            ));
        };

        let Some(documents) = documents else {
            return Err(ContextManagerError::LoadAgentError(
                "No documents returned from ChromaDB".to_string(),
            ));
        };

        let mut chunks = Vec::new();
        for (i, id) in ids.iter().enumerate() {
            let metadata = metadatas.get(i).and_then(|m| m.as_ref()).ok_or_else(|| {
                ContextManagerError::LoadAgentError(format!("Missing metadata for chunk {}", id))
            })?;

            let document = documents.get(i).and_then(|d| d.as_ref()).ok_or_else(|| {
                ContextManagerError::LoadAgentError(format!("Missing document for chunk {}", id))
            })?;

            // Extract metadata fields
            let agent_id_str = metadata
                .get("agent_id")
                .and_then(|v| match v {
                    MetadataValue::Str(s) => Some(s.as_str()),
                    _ => None,
                })
                .ok_or_else(|| {
                    ContextManagerError::LoadAgentError(format!(
                        "Missing or invalid agent_id in chunk {}",
                        id
                    ))
                })?;

            let agent_id = AgentID::from_human_readable(agent_id_str).ok_or_else(|| {
                ContextManagerError::LoadAgentError(format!(
                    "Invalid agent_id format in chunk {}: {}",
                    id, agent_id_str
                ))
            })?;

            let context_seq_no = metadata
                .get("context_seq_no")
                .and_then(|v| match v {
                    MetadataValue::Int(i) => Some(*i as u32),
                    _ => None,
                })
                .ok_or_else(|| {
                    ContextManagerError::LoadAgentError(format!(
                        "Missing or invalid context_seq_no in chunk {}",
                        id
                    ))
                })?;

            let transaction_seq_no = metadata
                .get("transaction_seq_no")
                .and_then(|v| match v {
                    MetadataValue::Int(i) => Some(*i as u64),
                    _ => None,
                })
                .ok_or_else(|| {
                    ContextManagerError::LoadAgentError(format!(
                        "Missing or invalid transaction_seq_no in chunk {}",
                        id
                    ))
                })?;

            let chunk_seq_no = metadata
                .get("chunk_seq_no")
                .and_then(|v| match v {
                    MetadataValue::Int(i) => Some(*i as u32),
                    _ => None,
                })
                .ok_or_else(|| {
                    ContextManagerError::LoadAgentError(format!(
                        "Missing or invalid chunk_seq_no in chunk {}",
                        id
                    ))
                })?;

            let total_chunks = metadata
                .get("total_chunks")
                .and_then(|v| match v {
                    MetadataValue::Int(i) => Some(*i as u32),
                    _ => None,
                })
                .ok_or_else(|| {
                    ContextManagerError::LoadAgentError(format!(
                        "Missing or invalid total_chunks in chunk {}",
                        id
                    ))
                })?;

            chunks.push(TransactionChunk {
                agent_id,
                context_seq_no,
                transaction_seq_no,
                chunk_seq_no,
                total_chunks,
                data: document.clone(),
            });
        }

        Ok(chunks)
    }

    /// Organize chunks into agent data with contexts and transactions
    fn assemble_agent_data(
        &self,
        agent_id: AgentID,
        chunks: Vec<TransactionChunk>,
    ) -> Result<AgentData, ContextManagerError> {
        use std::collections::BTreeMap;

        // Group chunks by (context_seq_no, transaction_seq_no)
        let mut context_transaction_chunks: BTreeMap<(u32, u64), Vec<TransactionChunk>> =
            BTreeMap::new();

        for chunk in chunks {
            let key = (chunk.context_seq_no, chunk.transaction_seq_no);
            context_transaction_chunks
                .entry(key)
                .or_default()
                .push(chunk);
        }

        // Group by context_seq_no to build contexts
        let mut contexts_map: BTreeMap<u32, Vec<Transaction>> = BTreeMap::new();

        for ((context_seq_no, _transaction_seq_no), mut transaction_chunks) in
            context_transaction_chunks
        {
            // Sort chunks by chunk_seq_no to ensure correct order
            transaction_chunks.sort_by_key(|c| c.chunk_seq_no);

            // Assemble transaction from chunks
            let transaction = Transaction::from_chunks(transaction_chunks).map_err(|e| {
                ContextManagerError::LoadAgentError(format!(
                    "Failed to assemble transaction: {}",
                    e
                ))
            })?;

            contexts_map
                .entry(context_seq_no)
                .or_default()
                .push(transaction);
        }

        // Build final AgentContext structs, sorting transactions within each context
        let mut contexts = Vec::new();
        for (context_seq_no, mut transactions) in contexts_map {
            // Sort transactions by transaction_seq_no
            transactions.sort_by_key(|t| t.transaction_seq_no);

            contexts.push(AgentContext {
                agent_id,
                context_seq_no,
                transactions,
            });
        }

        // Sort contexts by context_seq_no
        contexts.sort_by_key(|c| c.context_seq_no);

        Ok(AgentData { agent_id, contexts })
    }
}

/////////////////////////////////////////////// tests //////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentID;
    use chromadb::ChromaClient;
    use claudius::{MessageParam, MessageRole};

    async fn create_test_client() -> ChromaClient {
        let mut options = chromadb::client::ChromaClientOptions::new()
            .tenant(std::env::var("CHROMA_TENANT").unwrap_or("default_tenant".to_string()))
            .database(std::env::var("CHROMA_DATABASE").unwrap_or("default_database".to_string()));

        if let Ok(host) = std::env::var("CHROMA_HOST") {
            options = options.url(host);
        }

        let options = if let Ok(api_key) = std::env::var("CHROMA_API_KEY") {
            options.x_chroma_token(api_key)
        } else {
            options
        };

        ChromaClient::new(options)
            .await
            .expect("Failed to create ChromaDB client")
    }

    fn create_test_transaction() -> Transaction {
        Transaction {
            agent_id: AgentID::generate().unwrap(),
            context_seq_no: 1,
            transaction_seq_no: 42,
            msgs: vec![MessageParam {
                role: MessageRole::User,
                content: "Test message".into(),
            }],
            writes: vec![],
        }
    }

    #[tokio::test]
    async fn context_manager_creation() {
        let client = create_test_client().await;
        let collection = client
            .get_or_create_collection("test_transactions", None, None)
            .await
            .expect("Failed to create ChromaDB collection");
        let _context_manager =
            ContextManager::new(collection).expect("Failed to create ContextManager");
        // Test passes if we can create the context manager without panicking
    }

    #[tokio::test]
    async fn persist_and_verify_transaction() {
        let client = create_test_client().await;
        let collection = client
            .get_or_create_collection("test_transactions", None, None)
            .await
            .expect("Failed to create ChromaDB collection");
        let context_manager =
            ContextManager::new(collection).expect("Failed to create ContextManager");
        let transaction = create_test_transaction();

        // Persist the transaction
        let nonce_result = context_manager.persist_transaction(&transaction).await;
        assert!(nonce_result.is_ok());

        let nonce = nonce_result.unwrap();
        assert!(!nonce.is_empty());

        // Verify the persistence
        let verification_result = context_manager
            .verify_persistence(&transaction, &nonce)
            .await;
        assert!(verification_result.is_ok());
        assert!(verification_result.unwrap());

        // Verify with wrong nonce should return false
        let wrong_nonce = TransactionID::generate().unwrap().to_string();
        let wrong_verification = context_manager
            .verify_persistence(&transaction, &wrong_nonce)
            .await;
        assert!(wrong_verification.is_ok());
        assert!(!wrong_verification.unwrap());
    }

    #[tokio::test]
    async fn persist_transaction_with_multiple_chunks() {
        let client = create_test_client().await;
        let collection = client
            .get_or_create_collection("test_transactions", None, None)
            .await
            .expect("Failed to create ChromaDB collection");
        let context_manager =
            ContextManager::new(collection).expect("Failed to create ContextManager");

        // Create a large transaction that will be chunked
        let large_content = "x".repeat(crate::CHUNK_SIZE_LIMIT * 2);
        let mut transaction = create_test_transaction();
        transaction.msgs.push(MessageParam {
            role: MessageRole::Assistant,
            content: large_content.into(),
        });

        let nonce = context_manager
            .persist_transaction(&transaction)
            .await
            .unwrap();
        let verification = context_manager
            .verify_persistence(&transaction, &nonce)
            .await
            .unwrap();
        assert!(verification);
    }

    #[tokio::test]
    async fn verify_nonexistent_transaction() {
        let client = create_test_client().await;
        let collection = client
            .get_or_create_collection("test_transactions", None, None)
            .await
            .expect("Failed to create ChromaDB collection");
        let context_manager =
            ContextManager::new(collection).expect("Failed to create ContextManager");
        let transaction = create_test_transaction();
        let fake_nonce = TransactionID::generate().unwrap().to_string();

        let verification = context_manager
            .verify_persistence(&transaction, &fake_nonce)
            .await
            .unwrap();
        assert!(!verification);
    }

    #[tokio::test]
    async fn load_agent_single_context_single_transaction() {
        let client = create_test_client().await;
        let collection = client
            .get_or_create_collection("test_transactions", None, None)
            .await
            .expect("Failed to create ChromaDB collection");
        let context_manager =
            ContextManager::new(collection).expect("Failed to create ContextManager");

        let agent_id = AgentID::generate().unwrap();
        let transaction = Transaction {
            agent_id,
            context_seq_no: 1,
            transaction_seq_no: 1,
            msgs: vec![MessageParam {
                role: MessageRole::User,
                content: "Test message for load_agent".into(),
            }],
            writes: vec![],
        };

        // Persist the transaction first
        let _nonce = context_manager
            .persist_transaction(&transaction)
            .await
            .unwrap();

        // Load the agent data
        let agent_data = context_manager.load_agent(agent_id).await.unwrap();

        // Verify the loaded data
        assert_eq!(agent_data.agent_id, agent_id);
        assert_eq!(agent_data.contexts.len(), 1);

        let context = &agent_data.contexts[0];
        assert_eq!(context.agent_id, agent_id);
        assert_eq!(context.context_seq_no, 1);
        assert_eq!(context.transactions.len(), 1);

        let loaded_transaction = &context.transactions[0];
        assert_eq!(loaded_transaction.agent_id, transaction.agent_id);
        assert_eq!(
            loaded_transaction.context_seq_no,
            transaction.context_seq_no
        );
        assert_eq!(
            loaded_transaction.transaction_seq_no,
            transaction.transaction_seq_no
        );
        assert_eq!(loaded_transaction.msgs.len(), transaction.msgs.len());
    }

    #[tokio::test]
    async fn load_agent_multiple_contexts_multiple_transactions() {
        let client = create_test_client().await;
        let collection = client
            .get_or_create_collection("test_transactions", None, None)
            .await
            .expect("Failed to create ChromaDB collection");
        let context_manager =
            ContextManager::new(collection).expect("Failed to create ContextManager");

        let agent_id = AgentID::generate().unwrap();

        // Create transactions in different contexts
        let transactions = vec![
            Transaction {
                agent_id,
                context_seq_no: 1,
                transaction_seq_no: 1,
                msgs: vec![MessageParam {
                    role: MessageRole::User,
                    content: "Context 1, Transaction 1".into(),
                }],
                writes: vec![],
            },
            Transaction {
                agent_id,
                context_seq_no: 1,
                transaction_seq_no: 2,
                msgs: vec![MessageParam {
                    role: MessageRole::Assistant,
                    content: "Context 1, Transaction 2".into(),
                }],
                writes: vec![],
            },
            Transaction {
                agent_id,
                context_seq_no: 2,
                transaction_seq_no: 1,
                msgs: vec![MessageParam {
                    role: MessageRole::User,
                    content: "Context 2, Transaction 1".into(),
                }],
                writes: vec![],
            },
        ];

        // Persist all transactions
        for transaction in &transactions {
            let _nonce = context_manager
                .persist_transaction(transaction)
                .await
                .unwrap();
        }

        // Load the agent data
        let agent_data = context_manager.load_agent(agent_id).await.unwrap();

        // Verify the loaded data structure
        assert_eq!(agent_data.agent_id, agent_id);
        assert_eq!(agent_data.contexts.len(), 2);

        // Check context 1
        let context1 = &agent_data.contexts[0];
        assert_eq!(context1.context_seq_no, 1);
        assert_eq!(context1.transactions.len(), 2);
        assert_eq!(context1.transactions[0].transaction_seq_no, 1);
        assert_eq!(context1.transactions[1].transaction_seq_no, 2);

        // Check context 2
        let context2 = &agent_data.contexts[1];
        assert_eq!(context2.context_seq_no, 2);
        assert_eq!(context2.transactions.len(), 1);
        assert_eq!(context2.transactions[0].transaction_seq_no, 1);

        // Test helper methods
        let all_transactions = agent_data.all_transactions();
        assert_eq!(all_transactions.len(), 3);

        let latest_context = agent_data.latest_context().unwrap();
        assert_eq!(latest_context.context_seq_no, 2);

        let specific_context = agent_data.get_context(1).unwrap();
        assert_eq!(specific_context.transactions.len(), 2);
    }

    #[tokio::test]
    async fn load_agent_with_chunked_transactions() {
        let client = create_test_client().await;
        let collection = client
            .get_or_create_collection("test_transactions", None, None)
            .await
            .expect("Failed to create ChromaDB collection");
        let context_manager =
            ContextManager::new(collection).expect("Failed to create ContextManager");

        let agent_id = AgentID::generate().unwrap();

        // Create a large transaction that will be chunked
        let large_content = "x".repeat(crate::CHUNK_SIZE_LIMIT * 2);
        let transaction = Transaction {
            agent_id,
            context_seq_no: 1,
            transaction_seq_no: 1,
            msgs: vec![MessageParam {
                role: MessageRole::User,
                content: large_content.into(),
            }],
            writes: vec![],
        };

        // Persist the transaction (this will create multiple chunks)
        let _nonce = context_manager
            .persist_transaction(&transaction)
            .await
            .unwrap();

        // Load the agent data
        let agent_data = context_manager.load_agent(agent_id).await.unwrap();

        // Verify the transaction was properly reassembled
        assert_eq!(agent_data.contexts.len(), 1);
        let context = &agent_data.contexts[0];
        assert_eq!(context.transactions.len(), 1);

        let loaded_transaction = &context.transactions[0];
        assert_eq!(loaded_transaction.msgs.len(), transaction.msgs.len());
        assert_eq!(
            loaded_transaction.msgs[0].content,
            transaction.msgs[0].content
        );
    }

    #[tokio::test]
    async fn load_nonexistent_agent() {
        let client = create_test_client().await;
        let collection = client
            .get_or_create_collection("test_transactions", None, None)
            .await
            .expect("Failed to create ChromaDB collection");
        let context_manager =
            ContextManager::new(collection).expect("Failed to create ContextManager");

        let nonexistent_agent_id = AgentID::generate().unwrap();

        // Load data for agent that doesn't exist
        let agent_data = context_manager
            .load_agent(nonexistent_agent_id)
            .await
            .unwrap();

        // Should return empty but valid agent data
        assert_eq!(agent_data.agent_id, nonexistent_agent_id);
        assert!(agent_data.contexts.is_empty());
        assert!(agent_data.all_transactions().is_empty());
        assert!(agent_data.latest_context().is_none());
    }

    #[tokio::test]
    async fn fluent_transaction_building_next_transaction() {
        let client = create_test_client().await;
        let collection = client
            .get_or_create_collection("test_transactions", None, None)
            .await
            .expect("Failed to create ChromaDB collection");
        let context_manager =
            ContextManager::new(collection).expect("Failed to create ContextManager");

        let agent_id = AgentID::generate().unwrap();

        // Create initial transaction
        let transaction = Transaction {
            agent_id,
            context_seq_no: 1,
            transaction_seq_no: 1,
            msgs: vec![MessageParam {
                role: MessageRole::User,
                content: "Initial message".into(),
            }],
            writes: vec![],
        };

        context_manager
            .persist_transaction(&transaction)
            .await
            .unwrap();

        // Load agent data
        let mut agent_data = context_manager.load_agent(agent_id).await.unwrap();

        // Use fluent API to add next transaction
        let nonce = agent_data
            .next_transaction(&context_manager)
            .message(MessageParam {
                role: MessageRole::Assistant,
                content: "Response message".into(),
            })
            .save()
            .await
            .unwrap();

        assert!(!nonce.is_empty());

        // Verify the agent data was updated
        assert_eq!(agent_data.contexts.len(), 1);
        let context = &agent_data.contexts[0];
        assert_eq!(context.transactions.len(), 2);
        assert_eq!(context.transactions[1].transaction_seq_no, 2);
        assert_eq!(context.transactions[1].msgs.len(), 1);

        // Verify it was persisted to ChromaDB
        let reloaded_data = context_manager.load_agent(agent_id).await.unwrap();
        assert_eq!(reloaded_data.contexts[0].transactions.len(), 2);
    }

    #[tokio::test]
    async fn fluent_transaction_building_new_context() {
        let client = create_test_client().await;
        let collection = client
            .get_or_create_collection("test_transactions", None, None)
            .await
            .expect("Failed to create ChromaDB collection");
        let context_manager =
            ContextManager::new(collection).expect("Failed to create ContextManager");

        let agent_id = AgentID::generate().unwrap();

        // Create initial transaction in context 1
        let transaction = Transaction {
            agent_id,
            context_seq_no: 1,
            transaction_seq_no: 1,
            msgs: vec![MessageParam {
                role: MessageRole::User,
                content: "Initial message".into(),
            }],
            writes: vec![],
        };

        context_manager
            .persist_transaction(&transaction)
            .await
            .unwrap();

        // Load agent data
        let mut agent_data = context_manager.load_agent(agent_id).await.unwrap();

        // Use fluent API to create transaction in new context
        let nonce = agent_data
            .new_context(&context_manager)
            .message(MessageParam {
                role: MessageRole::User,
                content: "New context message".into(),
            })
            .save()
            .await
            .unwrap();

        assert!(!nonce.is_empty());

        // Verify the agent data was updated with new context
        assert_eq!(agent_data.contexts.len(), 2);
        let new_context = &agent_data.contexts[1];
        assert_eq!(new_context.context_seq_no, 2);
        assert_eq!(new_context.transactions.len(), 1);
        assert_eq!(new_context.transactions[0].transaction_seq_no, 1);

        // Verify it was persisted to ChromaDB
        let reloaded_data = context_manager.load_agent(agent_id).await.unwrap();
        assert_eq!(reloaded_data.contexts.len(), 2);
        assert_eq!(reloaded_data.contexts[1].context_seq_no, 2);
    }

    #[tokio::test]
    async fn fluent_transaction_building_with_file_writes() {
        let client = create_test_client().await;
        let collection = client
            .get_or_create_collection("test_transactions", None, None)
            .await
            .expect("Failed to create ChromaDB collection");
        let context_manager =
            ContextManager::new(collection).expect("Failed to create ContextManager");

        let agent_id = AgentID::generate().unwrap();
        let mount_id = crate::MountID::generate().unwrap();

        // Start with empty agent
        let mut agent_data = AgentData {
            agent_id,
            contexts: vec![],
        };

        // Use fluent API to create transaction with messages and file writes
        let nonce = agent_data
            .next_transaction(&context_manager)
            .message(MessageParam {
                role: MessageRole::User,
                content: "Create some files".into(),
            })
            .write_file(mount_id, "/test.txt", "Hello, world!")
            .write_file(mount_id, "/config.json", r#"{"setting": "value"}"#)
            .save()
            .await
            .unwrap();

        assert!(!nonce.is_empty());

        // Verify the transaction was created with file writes
        assert_eq!(agent_data.contexts.len(), 1);
        let context = &agent_data.contexts[0];
        assert_eq!(context.transactions.len(), 1);
        let transaction = &context.transactions[0];
        assert_eq!(transaction.writes.len(), 2);
        assert_eq!(transaction.writes[0].path, "/test.txt");
        assert_eq!(transaction.writes[0].data, "Hello, world!");
        assert_eq!(transaction.writes[1].path, "/config.json");

        // Verify persistence
        let reloaded_data = context_manager.load_agent(agent_id).await.unwrap();
        let reloaded_transaction = &reloaded_data.contexts[0].transactions[0];
        assert_eq!(reloaded_transaction.writes.len(), 2);
    }

    #[tokio::test]
    async fn fluent_transaction_building_multiple_messages() {
        let client = create_test_client().await;
        let collection = client
            .get_or_create_collection("test_transactions", None, None)
            .await
            .expect("Failed to create ChromaDB collection");
        let context_manager =
            ContextManager::new(collection).expect("Failed to create ContextManager");

        let agent_id = AgentID::generate().unwrap();

        // Start with empty agent
        let mut agent_data = AgentData {
            agent_id,
            contexts: vec![],
        };

        let messages = vec![
            MessageParam {
                role: MessageRole::User,
                content: "First message".into(),
            },
            MessageParam {
                role: MessageRole::Assistant,
                content: "Second message".into(),
            },
        ];

        // Use fluent API with multiple messages
        let nonce = agent_data
            .next_transaction(&context_manager)
            .messages(messages.clone())
            .message(MessageParam {
                role: MessageRole::User,
                content: "Third message".into(),
            })
            .save()
            .await
            .unwrap();

        assert!(!nonce.is_empty());

        // Verify all messages were added
        let context = &agent_data.contexts[0];
        let transaction = &context.transactions[0];
        assert_eq!(transaction.msgs.len(), 3);

        // Verify roles are correct
        assert_eq!(transaction.msgs[0].role, MessageRole::User);
        assert_eq!(transaction.msgs[1].role, MessageRole::Assistant);
        assert_eq!(transaction.msgs[2].role, MessageRole::User);

        // Verify content matches by directly comparing the content fields
        assert_eq!(transaction.msgs[0].content, messages[0].content);
        assert_eq!(transaction.msgs[1].content, messages[1].content);
    }

    #[tokio::test]
    async fn fluent_transaction_building_build_without_save() {
        let client = create_test_client().await;
        let collection = client
            .get_or_create_collection("test_transactions", None, None)
            .await
            .expect("Failed to create ChromaDB collection");
        let context_manager =
            ContextManager::new(collection).expect("Failed to create ContextManager");

        let agent_id = AgentID::generate().unwrap();
        let mut agent_data = AgentData {
            agent_id,
            contexts: vec![],
        };

        // Build without saving
        let transaction = agent_data
            .next_transaction(&context_manager)
            .message(MessageParam {
                role: MessageRole::User,
                content: "Test message".into(),
            })
            .build();

        // Verify transaction structure
        assert_eq!(transaction.agent_id, agent_id);
        assert_eq!(transaction.context_seq_no, 1);
        assert_eq!(transaction.transaction_seq_no, 1);
        assert_eq!(transaction.msgs.len(), 1);

        // Verify agent_data was not modified (build() doesn't update it)
        assert!(agent_data.contexts.is_empty());

        // Verify nothing was persisted
        let loaded_data = context_manager.load_agent(agent_id).await.unwrap();
        assert!(loaded_data.contexts.is_empty());
    }

    #[tokio::test]
    async fn fluent_transaction_building_sequence_numbers() {
        let client = create_test_client().await;
        let collection = client
            .get_or_create_collection("test_transactions", None, None)
            .await
            .expect("Failed to create ChromaDB collection");
        let context_manager =
            ContextManager::new(collection).expect("Failed to create ContextManager");

        let agent_id = AgentID::generate().unwrap();
        let mut agent_data = AgentData {
            agent_id,
            contexts: vec![],
        };

        // First transaction - should be context 1, transaction 1
        let tx1 = agent_data
            .next_transaction(&context_manager)
            .message(MessageParam {
                role: MessageRole::User,
                content: "Message 1".into(),
            })
            .build();

        assert_eq!(tx1.context_seq_no, 1);
        assert_eq!(tx1.transaction_seq_no, 1);

        // Manually add to agent_data to simulate persistence
        agent_data.contexts.push(AgentContext {
            agent_id,
            context_seq_no: 1,
            transactions: vec![tx1],
        });

        // Second transaction - should be context 1, transaction 2
        let tx2 = agent_data
            .next_transaction(&context_manager)
            .message(MessageParam {
                role: MessageRole::Assistant,
                content: "Message 2".into(),
            })
            .build();

        assert_eq!(tx2.context_seq_no, 1);
        assert_eq!(tx2.transaction_seq_no, 2);

        // New context - should be context 2, transaction 1
        let tx3 = agent_data
            .new_context(&context_manager)
            .message(MessageParam {
                role: MessageRole::User,
                content: "Message 3".into(),
            })
            .build();

        assert_eq!(tx3.context_seq_no, 2);
        assert_eq!(tx3.transaction_seq_no, 1);
    }
}
