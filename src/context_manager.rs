use std::collections::HashMap;

use chromadb::{
    MetadataValue,
    collection::{ChromaCollection, CollectionEntries, GetOptions},
    filters,
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
    /// Error with file operations.
    FileError(FileSystemError),
}

/// Specific errors for file system operations
#[derive(Debug)]
pub enum FileSystemError {
    /// Invalid file path
    InvalidPath(String),
    /// File not found
    FileNotFound(String),
    /// Empty or invalid search pattern
    InvalidPattern(String),
    /// Content not found for replacement
    ContentNotFound(String),
}

impl std::fmt::Display for FileSystemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileSystemError::InvalidPath(msg) => write!(f, "Invalid file path: {}", msg),
            FileSystemError::FileNotFound(path) => write!(f, "File not found: {}", path),
            FileSystemError::InvalidPattern(msg) => write!(f, "Invalid search pattern: {}", msg),
            FileSystemError::ContentNotFound(msg) => write!(f, "Content not found: {}", msg),
        }
    }
}

impl std::error::Error for FileSystemError {}

impl From<FileSystemError> for ContextManagerError {
    fn from(error: FileSystemError) -> Self {
        ContextManagerError::FileError(error)
    }
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
            ContextManagerError::FileError(e) => write!(f, "File system error: {}", e),
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
            ContextManagerError::FileError(e) => Some(e),
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

/// Extract a string field from ChromaDB metadata with error handling
fn extract_metadata_string(
    metadata: &HashMap<String, MetadataValue>,
    field_name: &str,
    chunk_id: &str,
) -> Result<String, ContextManagerError> {
    metadata
        .get(field_name)
        .and_then(|v| match v {
            MetadataValue::Str(s) => Some(s.clone()),
            _ => None,
        })
        .ok_or_else(|| {
            ContextManagerError::LoadAgentError(format!(
                "Missing or invalid {} in chunk {}",
                field_name, chunk_id
            ))
        })
}

/// Extract a u32 field from ChromaDB metadata with error handling
fn extract_metadata_u32(
    metadata: &HashMap<String, MetadataValue>,
    field_name: &str,
    chunk_id: &str,
) -> Result<u32, ContextManagerError> {
    metadata
        .get(field_name)
        .and_then(|v| match v {
            MetadataValue::Int(i) => Some(*i as u32),
            _ => None,
        })
        .ok_or_else(|| {
            ContextManagerError::LoadAgentError(format!(
                "Missing or invalid {} in chunk {}",
                field_name, chunk_id
            ))
        })
}

/// Extract a u64 field from ChromaDB metadata with error handling
fn extract_metadata_u64(
    metadata: &HashMap<String, MetadataValue>,
    field_name: &str,
    chunk_id: &str,
) -> Result<u64, ContextManagerError> {
    metadata
        .get(field_name)
        .and_then(|v| match v {
            MetadataValue::Int(i) => Some(*i as u64),
            _ => None,
        })
        .ok_or_else(|| {
            ContextManagerError::LoadAgentError(format!(
                "Missing or invalid {} in chunk {}",
                field_name, chunk_id
            ))
        })
}

/// Validates a file path for security and correctness
fn validate_file_path(path: &str) -> Result<(), FileSystemError> {
    if path.is_empty() {
        return Err(FileSystemError::InvalidPath(
            "File path cannot be empty".to_string(),
        ));
    }

    if path.contains('\0') {
        return Err(FileSystemError::InvalidPath(
            "File path cannot contain null bytes".to_string(),
        ));
    }

    // Check for path traversal attempts
    if path.contains("..") {
        return Err(FileSystemError::InvalidPath(
            "File path cannot contain '..' (path traversal)".to_string(),
        ));
    }

    // Ensure path starts with /
    if !path.starts_with('/') {
        return Err(FileSystemError::InvalidPath(
            "File path must start with '/'".to_string(),
        ));
    }

    // Check for reasonable length (prevent DoS)
    if path.len() > 4096 {
        return Err(FileSystemError::InvalidPath(
            "File path too long (max 4096 characters)".to_string(),
        ));
    }

    Ok(())
}

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

/// Search for the most recent file content in a collection of contexts
/// Returns the content if found, None if not found
fn find_most_recent_file_content(
    contexts: &[AgentContext],
    mount: crate::MountID,
    path: &str,
) -> Option<String> {
    // Search through all contexts in reverse chronological order
    // to find the most recent write to this file
    for context in contexts.iter().rev() {
        for transaction in context.transactions.iter().rev() {
            for write in transaction.writes.iter().rev() {
                if write.mount == mount && write.path == path {
                    return Some(write.data.clone());
                }
            }
        }
    }
    None
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

    /// Get the content of a file from the most recent version in transaction history
    pub fn get_file_content(
        &self,
        mount: crate::MountID,
        path: &str,
    ) -> Result<Option<String>, FileSystemError> {
        validate_file_path(path)?;
        Ok(find_most_recent_file_content(&self.contexts, mount, path))
    }

    /// List all files that have been written in transaction history for a mount
    pub fn list_files(&self, mount: crate::MountID) -> Vec<String> {
        use std::collections::HashSet;

        // Collect file paths using iterator chain to reduce intermediate allocations
        let mut files: HashSet<String> = self
            .contexts
            .iter()
            .flat_map(|context| &context.transactions)
            .flat_map(|transaction| &transaction.writes)
            .filter(|write| write.mount == mount)
            .map(|write| write.path.clone())
            .collect();

        // Convert to sorted Vec - can't avoid this allocation as we need ownership
        let mut result: Vec<String> = files.drain().collect();
        result.sort();
        result
    }

    /// Search for files containing a pattern in their content
    pub fn search_file_contents(
        &self,
        mount: crate::MountID,
        pattern: &str,
    ) -> Result<Vec<(String, Vec<String>)>, FileSystemError> {
        if pattern.is_empty() {
            return Err(FileSystemError::InvalidPattern(
                "Search pattern cannot be empty".to_string(),
            ));
        }

        let files = self.list_files(mount);

        // Use iterator with filter_map to avoid intermediate allocations
        let matches = files
            .into_iter()
            .filter_map(|file_path| {
                // Only process files that exist and have valid content
                match self.get_file_content(mount, &file_path) {
                    Ok(Some(content)) => {
                        // Use iterator to avoid intermediate Vec allocation
                        let matching_lines: Vec<String> = content
                            .lines()
                            .enumerate()
                            .filter_map(|(line_num, line)| {
                                if line.contains(pattern) {
                                    Some(format!("{}:{}", line_num + 1, line))
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if matching_lines.is_empty() {
                            None
                        } else {
                            Some((file_path, matching_lines))
                        }
                    }
                    Ok(None) | Err(_) => None, // File doesn't exist or invalid path, skip
                }
            })
            .collect();

        Ok(matches)
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
    /// Helper to create a Transaction struct from the current builder state
    fn create_transaction(&self) -> Transaction {
        Transaction {
            agent_id: self.agent_data.agent_id,
            context_seq_no: self.context_seq_no,
            transaction_seq_no: self.transaction_seq_no,
            msgs: self.msgs.clone(),
            writes: self.writes.clone(),
        }
    }

    /// Helper to create and add a FileWrite after validation
    fn add_validated_write<P: Into<String>, D: Into<String>>(
        &mut self,
        mount: crate::MountID,
        path: P,
        data: D,
    ) -> Result<(), FileSystemError> {
        let path_str = path.into();
        validate_file_path(&path_str)?;

        self.writes.push(FileWrite {
            mount,
            path: path_str,
            data: data.into(),
        });
        Ok(())
    }

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
    ) -> Result<Self, FileSystemError> {
        self.add_validated_write(mount, path, data)?;
        Ok(self)
    }

    /// Add multiple file writes to this transaction
    pub fn write_files(mut self, writes: Vec<FileWrite>) -> Result<Self, FileSystemError> {
        // Validate all paths before adding any writes
        for write in &writes {
            validate_file_path(&write.path)?;
        }

        self.writes.extend(writes);
        Ok(self)
    }

    /// Perform a string replacement in an existing file and add the result as a write
    pub fn str_replace_file<P: Into<String>>(
        mut self,
        mount: crate::MountID,
        path: P,
        old_content: &str,
        new_content: &str,
    ) -> Result<Self, FileSystemError> {
        let path_str = path.into();
        match self.agent_data.get_file_content(mount, &path_str)? {
            Some(current_content) => {
                if !current_content.contains(old_content) {
                    return Err(FileSystemError::ContentNotFound(format!(
                        "Content '{}' not found in file {}",
                        old_content, path_str
                    )));
                }
                let updated_content = current_content.replace(old_content, new_content);
                self.writes.push(FileWrite {
                    mount,
                    path: path_str,
                    data: updated_content,
                });
                Ok(self)
            }
            None => Err(FileSystemError::FileNotFound(path_str)),
        }
    }

    /// Insert content into a file (creates new file or overwrites existing)
    pub fn insert_file<P: Into<String>, D: Into<String>>(
        mut self,
        mount: crate::MountID,
        path: P,
        content: D,
    ) -> Result<Self, FileSystemError> {
        self.add_validated_write(mount, path, content)?;
        Ok(self)
    }

    /// Get the current buffered content for a file (considering both transaction history and pending writes)
    pub fn get_buffered_content(
        &self,
        mount: crate::MountID,
        path: &str,
    ) -> Result<Option<String>, FileSystemError> {
        validate_file_path(path)?;

        // First check pending writes in reverse order (most recent first)
        for write in self.writes.iter().rev() {
            if write.mount == mount && write.path == path {
                return Ok(Some(write.data.clone()));
            }
        }
        // Fall back to transaction history
        self.agent_data.get_file_content(mount, path)
    }

    /// View file content (same as get_buffered_content but returns Option instead of Result<Option>)
    pub fn view_file(&self, mount: crate::MountID, path: &str) -> Option<String> {
        self.get_buffered_content(mount, path).ok().flatten()
    }

    /// List all files for a mount (combines transaction history and pending writes)
    pub fn list_files(&self, mount: crate::MountID) -> Vec<String> {
        use std::collections::HashSet;

        // Start with files from transaction history
        let mut files: HashSet<String> = self.agent_data.list_files(mount).into_iter().collect();

        // Add files from pending writes
        for write in &self.writes {
            if write.mount == mount {
                files.insert(write.path.clone());
            }
        }

        let mut result: Vec<String> = files.into_iter().collect();
        result.sort();
        result
    }

    /// Search for files containing a pattern (combines transaction history and pending writes)
    pub fn search_files(&self, mount: crate::MountID, pattern: &str) -> Vec<(String, Vec<String>)> {
        let files = self.list_files(mount);

        files
            .into_iter()
            .filter_map(|file_path| {
                // Get content using get_buffered_content to check pending writes too
                match self.get_buffered_content(mount, &file_path) {
                    Ok(Some(content)) => {
                        let matching_lines: Vec<String> = content
                            .lines()
                            .enumerate()
                            .filter_map(|(line_num, line)| {
                                if line.contains(pattern) {
                                    Some(format!("{}:{}", line_num + 1, line))
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if matching_lines.is_empty() {
                            None
                        } else {
                            Some((file_path, matching_lines))
                        }
                    }
                    Ok(None) | Err(_) => None,
                }
            })
            .collect()
    }

    /// Get a summary of all pending writes
    pub fn get_write_summary(&self) -> Vec<(crate::MountID, String, usize)> {
        let mut summary = Vec::new();
        for write in &self.writes {
            summary.push((write.mount, write.path.clone(), write.data.len()));
        }
        summary
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
        let transaction = self.create_transaction();

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
        self.create_transaction()
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

            // Extract metadata fields using helper functions
            let agent_id_str = extract_metadata_string(metadata, "agent_id", id)?;

            let agent_id = AgentID::from_human_readable(&agent_id_str).ok_or_else(|| {
                ContextManagerError::LoadAgentError(format!(
                    "Invalid agent_id format in chunk {}: {}",
                    id, agent_id_str
                ))
            })?;

            let context_seq_no = extract_metadata_u32(metadata, "context_seq_no", id)?;
            let transaction_seq_no = extract_metadata_u64(metadata, "transaction_seq_no", id)?;
            let chunk_seq_no = extract_metadata_u32(metadata, "chunk_seq_no", id)?;
            let total_chunks = extract_metadata_u32(metadata, "total_chunks", id)?;

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

    async fn create_test_context_manager() -> ContextManager {
        let client = create_test_client().await;
        let collection = client
            .get_or_create_collection("test_transactions", None, None)
            .await
            .expect("Failed to create ChromaDB collection");
        ContextManager::new(collection).expect("Failed to create ContextManager")
    }

    #[tokio::test]
    async fn context_manager_creation() {
        let _context_manager = create_test_context_manager().await;
        // Test passes if we can create the context manager without panicking
    }

    #[tokio::test]
    async fn persist_and_verify_transaction() {
        let context_manager = create_test_context_manager().await;
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
            .unwrap()
            .write_file(mount_id, "/config.json", r#"{"setting": "value"}"#)
            .unwrap()
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

    #[tokio::test]
    async fn agent_data_file_reading() {
        let client = create_test_client().await;

        if let Err(e) = client.heartbeat().await {
            println!("TODO(claude): cleanup this output");
            println!("Skipping test - ChromaDB not available: {:?}", e);
            return;
        }

        let collection_name = format!("test_agent_file_reading_{}", rand::random::<u32>());
        let collection = client
            .get_or_create_collection(&collection_name, None, None)
            .await
            .unwrap();

        let context_manager = ContextManager::new(collection).unwrap();
        let agent_id = AgentID::generate().unwrap();
        let mount_id = crate::MountID::generate().unwrap();

        // Create initial transaction with some files
        let mut agent_data = AgentData {
            agent_id,
            contexts: Vec::new(),
        };

        let _nonce = agent_data
            .new_context(&context_manager)
            .write_file(mount_id, "/file1.txt", "First file content")
            .unwrap()
            .write_file(mount_id, "/file2.txt", "Second file content")
            .unwrap()
            .write_file(mount_id, "/subdir/file3.txt", "Third file content")
            .unwrap()
            .save()
            .await
            .unwrap();

        // Test get_file_content
        assert_eq!(
            agent_data.get_file_content(mount_id, "/file1.txt").unwrap(),
            Some("First file content".to_string())
        );
        assert_eq!(
            agent_data
                .get_file_content(mount_id, "/nonexistent.txt")
                .unwrap(),
            None
        );

        // Test list_files
        let files = agent_data.list_files(mount_id);
        assert_eq!(files.len(), 3);
        assert!(files.contains(&"/file1.txt".to_string()));
        assert!(files.contains(&"/file2.txt".to_string()));
        assert!(files.contains(&"/subdir/file3.txt".to_string()));

        // Test search_file_contents
        let matches = agent_data.search_file_contents(mount_id, "file").unwrap();
        assert_eq!(matches.len(), 3);
        assert!(matches.iter().any(|(path, _)| path == "/file1.txt"));

        let matches = agent_data.search_file_contents(mount_id, "Third").unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].0, "/subdir/file3.txt");
    }

    #[tokio::test]
    async fn transaction_builder_filesystem_methods() {
        let client = create_test_client().await;

        if let Err(e) = client.heartbeat().await {
            println!("TODO(claude): cleanup this output");
            println!("Skipping test - ChromaDB not available: {:?}", e);
            return;
        }

        let collection_name = format!("test_builder_fs_{}", rand::random::<u32>());
        let collection = client
            .get_or_create_collection(&collection_name, None, None)
            .await
            .unwrap();

        let context_manager = ContextManager::new(collection).unwrap();
        let agent_id = AgentID::generate().unwrap();
        let mount_id = crate::MountID::generate().unwrap();

        // Create initial transaction
        let mut agent_data = AgentData {
            agent_id,
            contexts: Vec::new(),
        };

        let _nonce = agent_data
            .new_context(&context_manager)
            .write_file(mount_id, "/original.txt", "Original content")
            .unwrap()
            .save()
            .await
            .unwrap();

        // Test TransactionBuilder filesystem methods
        let builder = agent_data.next_transaction(&context_manager);

        // Test view_file
        assert_eq!(
            builder.view_file(mount_id, "/original.txt"),
            Some("Original content".to_string())
        );
        assert_eq!(builder.view_file(mount_id, "/nonexistent.txt"), None);

        // Test list_files
        let files = builder.list_files(mount_id);
        assert_eq!(files.len(), 1);
        assert!(files.contains(&"/original.txt".to_string()));

        // Test search_files
        let matches = builder.search_files(mount_id, "Original");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].0, "/original.txt");

        // Test str_replace_file
        let builder = builder
            .str_replace_file(mount_id, "/original.txt", "Original", "Modified")
            .unwrap();

        // Test get_buffered_content (should show the modified version)
        assert_eq!(
            builder
                .get_buffered_content(mount_id, "/original.txt")
                .unwrap(),
            Some("Modified content".to_string())
        );

        // Test str_replace_file error cases
        let builder2 = agent_data.next_transaction(&context_manager);
        let result = builder2.str_replace_file(mount_id, "/nonexistent.txt", "old", "new");
        assert!(result.is_err());
        if let Err(error) = result {
            assert!(error.to_string().contains("not found"));
        }

        let builder3 = agent_data.next_transaction(&context_manager);
        let result = builder3.str_replace_file(mount_id, "/original.txt", "nonexistent", "new");
        assert!(result.is_err());
        if let Err(error) = result {
            assert!(error.to_string().contains("not found"));
        }

        // Test insert_file
        let builder4 = agent_data
            .next_transaction(&context_manager)
            .insert_file(mount_id, "/new_file.txt", "New file content")
            .unwrap();

        assert_eq!(
            builder4
                .get_buffered_content(mount_id, "/new_file.txt")
                .unwrap(),
            Some("New file content".to_string())
        );
    }

    #[tokio::test]
    async fn write_buffering_latest_wins() {
        let client = create_test_client().await;

        if let Err(e) = client.heartbeat().await {
            println!("TODO(claude): cleanup this output");
            println!("Skipping test - ChromaDB not available: {:?}", e);
            return;
        }

        let collection_name = format!("test_write_buffering_{}", rand::random::<u32>());
        let collection = client
            .get_or_create_collection(&collection_name, None, None)
            .await
            .unwrap();

        let context_manager = ContextManager::new(collection).unwrap();
        let agent_id = AgentID::generate().unwrap();
        let mount_id = crate::MountID::generate().unwrap();

        let mut agent_data = AgentData {
            agent_id,
            contexts: Vec::new(),
        };

        // Test multiple writes to same file - latest should win
        let builder = agent_data
            .new_context(&context_manager)
            .write_file(mount_id, "/multi_write.txt", "First write")
            .unwrap()
            .write_file(mount_id, "/multi_write.txt", "Second write")
            .unwrap()
            .write_file(mount_id, "/multi_write.txt", "Third write")
            .unwrap();

        // Test get_buffered_content returns the latest write
        assert_eq!(
            builder
                .get_buffered_content(mount_id, "/multi_write.txt")
                .unwrap(),
            Some("Third write".to_string())
        );

        // Test write summary shows all writes
        let summary = builder.get_write_summary();
        assert_eq!(summary.len(), 3); // All three writes present
        assert!(
            summary
                .iter()
                .all(|(_, path, _)| path == "/multi_write.txt")
        );

        // Save and verify the latest write persists
        let _nonce = builder.save().await.unwrap();

        // Verify that reading from agent data returns the latest write
        assert_eq!(
            agent_data
                .get_file_content(mount_id, "/multi_write.txt")
                .unwrap(),
            Some("Third write".to_string())
        );
    }

    #[tokio::test]
    async fn multiple_writes_same_file_appears_once() {
        let client = create_test_client().await;

        if let Err(e) = client.heartbeat().await {
            println!("TODO(claude): cleanup this output");
            println!("Skipping test - ChromaDB not available: {:?}", e);
            return;
        }

        let collection_name = format!("test_multiple_writes_{}", rand::random::<u32>());
        let collection = client
            .get_or_create_collection(&collection_name, None, None)
            .await
            .unwrap();

        let context_manager = ContextManager::new(collection).unwrap();
        let agent_id = AgentID::generate().unwrap();
        let mount_id = crate::MountID::generate().unwrap();

        let mut agent_data = AgentData {
            agent_id,
            contexts: Vec::new(),
        };

        // Write the same file multiple times across different transactions
        let _nonce1 = agent_data
            .new_context(&context_manager)
            .write_file(mount_id, "/config.txt", "Version 1 content")
            .unwrap()
            .write_file(mount_id, "/other.txt", "Other file content")
            .unwrap()
            .save()
            .await
            .unwrap();

        let _nonce2 = agent_data
            .next_transaction(&context_manager)
            .write_file(mount_id, "/config.txt", "Version 2 content")
            .unwrap()
            .save()
            .await
            .unwrap();

        let _nonce3 = agent_data
            .next_transaction(&context_manager)
            .write_file(mount_id, "/config.txt", "Version 3 final content")
            .unwrap()
            .save()
            .await
            .unwrap();

        // Test 1: list_files should show each file only once, despite multiple writes
        let files = agent_data.list_files(mount_id);
        assert_eq!(files.len(), 2); // Should have "/config.txt" and "/other.txt"
        assert!(files.contains(&"/config.txt".to_string()));
        assert!(files.contains(&"/other.txt".to_string()));

        // Count occurrences - each file should appear exactly once
        let config_count = files.iter().filter(|&f| f == "/config.txt").count();
        let other_count = files.iter().filter(|&f| f == "/other.txt").count();
        assert_eq!(
            config_count, 1,
            "config.txt should appear exactly once in file list"
        );
        assert_eq!(
            other_count, 1,
            "other.txt should appear exactly once in file list"
        );

        // Test 2: get_file_content should return the most recent version
        let content = agent_data
            .get_file_content(mount_id, "/config.txt")
            .unwrap();
        assert_eq!(content, Some("Version 3 final content".to_string()));

        let other_content = agent_data.get_file_content(mount_id, "/other.txt").unwrap();
        assert_eq!(other_content, Some("Other file content".to_string()));

        // Test 3: search_file_contents should find the file only once with latest content
        let matches = agent_data.search_file_contents(mount_id, "final").unwrap();
        assert_eq!(matches.len(), 1, "Should find config.txt exactly once");
        assert_eq!(matches[0].0, "/config.txt");
        assert!(matches[0].1[0].contains("Version 3 final content"));

        // Test 4: search should not find old versions
        let old_matches = agent_data
            .search_file_contents(mount_id, "Version 1")
            .unwrap();
        assert_eq!(
            old_matches.len(),
            0,
            "Should not find old Version 1 content"
        );

        let old_matches2 = agent_data
            .search_file_contents(mount_id, "Version 2")
            .unwrap();
        assert_eq!(
            old_matches2.len(),
            0,
            "Should not find old Version 2 content"
        );

        // Test 5: TransactionBuilder should also handle this correctly with pending writes
        let builder = agent_data
            .next_transaction(&context_manager)
            .write_file(mount_id, "/config.txt", "Pending version")
            .unwrap();

        // TransactionBuilder's list_files should still show each file once
        let builder_files = builder.list_files(mount_id);
        let config_count_builder = builder_files.iter().filter(|&f| f == "/config.txt").count();
        assert_eq!(
            config_count_builder, 1,
            "TransactionBuilder should show config.txt exactly once"
        );

        // get_buffered_content should return the pending write (most recent)
        assert_eq!(
            builder
                .get_buffered_content(mount_id, "/config.txt")
                .unwrap(),
            Some("Pending version".to_string())
        );

        // search_files should find the pending version
        let builder_matches = builder.search_files(mount_id, "Pending");
        assert_eq!(
            builder_matches.len(),
            1,
            "Should find file with pending write exactly once"
        );
        assert_eq!(builder_matches[0].0, "/config.txt");
    }
}
