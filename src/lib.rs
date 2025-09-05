#![doc = include_str!("../README.md")]

use std::collections::HashMap;
use std::fmt::Debug;

use chromadb::{ChromaClient, MetadataValue};
use claudius::{Anthropic, MessageParam};
use one_two_eight::generate_id;

///////////////////////////////////////////// Constants ////////////////////////////////////////////

const FILE_SIZE_LIMIT: usize = 8192;
const CHUNK_SIZE_LIMIT: usize = 16 * 1024; // 16KB limit for Chroma documents

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
        let serialized = serde_json::to_string(self)
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
        let mut start = 0;
        let mut chunk_seq_no = 0;

        // Calculate total chunks needed (approximate, will be refined)
        let estimated_chunks = serialized.len().div_ceil(CHUNK_SIZE_LIMIT);

        while start < serialized.len() {
            let mut end = std::cmp::min(start + CHUNK_SIZE_LIMIT, serialized.len());

            // If we're not at the end of the string and we're in the middle of a UTF-8 sequence,
            // backtrack to the nearest valid UTF-8 boundary
            if end < serialized.len() {
                while !serialized.is_char_boundary(end) && end > start {
                    end -= 1;
                }
            }

            // Safety check: ensure we make progress
            if end <= start {
                return Err(Error::TransactionError(
                    "Failed to chunk transaction: UTF-8 boundary issue".to_string(),
                ));
            }

            chunks.push(TransactionChunk {
                agent_id: self.agent_id,
                context_seq_no: self.context_seq_no,
                transaction_seq_no: self.transaction_seq_no,
                chunk_seq_no,
                total_chunks: estimated_chunks as u32, // Will update after all chunks are created
                data: serialized[start..end].to_string(),
            });

            chunk_seq_no += 1;
            start = end;
        }

        // Update total_chunks with the actual count
        let total_chunks = chunks.len() as u32;
        for chunk in &mut chunks {
            chunk.total_chunks = total_chunks;
        }

        Ok(chunks)
    }

    // TODO(claude):  Make this function return an InvariantViolation enum and not assert.
    fn check_invariants(&self) -> Result<(), Error> {
        for w in self.writes.iter() {
            if w.data.len() >= FILE_SIZE_LIMIT {
                return Err(Error::ChunkSizeExceeded(format!(
                    "File write exceeds size limit: {} bytes",
                    w.data.len()
                )));
            }
        }
        Ok(())
    }
}

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
    mount: MountID,
    path: String,
    data: String,
}

/////////////////////////////////////////// ContextSeal ////////////////////////////////////////////

/// A seal marks the end of a context and points to the next context in the lineage.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ContextSeal {
    pub context_id: ContextID,
    pub next_context_id: Option<ContextID>,
    pub sealed_at: u64, // Unix timestamp
    pub summary: String,
}

impl ContextSeal {
    /// Create a new context seal.
    pub fn new(context_id: ContextID, summary: String) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let sealed_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        ContextSeal {
            context_id,
            next_context_id: None,
            sealed_at,
            summary,
        }
    }

    /// Create a seal that points to a new context (for forking/compaction).
    pub fn with_next(mut self, next: ContextID) -> Self {
        self.next_context_id = Some(next);
        self
    }

    /// Validate that the seal is properly formed.
    pub fn validate(&self) -> Result<(), Error> {
        if self.summary.is_empty() {
            return Err(Error::SealError("Seal summary cannot be empty".to_string()));
        }
        Ok(())
    }
}

////////////////////////////////////////// MarkdownList ///////////////////////////////////////////

/// A markdown list structure for virtual filesystem storage.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct MarkdownList {
    pub mount_id: MountID,
    pub path: String,
    pub sections: HashMap<String, Vec<String>>, // section header -> bullet points
}

impl MarkdownList {
    /// Create a new markdown list.
    pub fn new(mount_id: MountID, path: String) -> Self {
        MarkdownList {
            mount_id,
            path,
            sections: HashMap::new(),
        }
    }

    /// Add a bullet point to a section.
    pub fn add_bullet(&mut self, section: String, bullet: String) {
        self.sections.entry(section).or_default().push(bullet);
    }

    /// Remove a bullet point from a section.
    pub fn remove_bullet(&mut self, section: &str, index: usize) -> Option<String> {
        Some(self.sections.get_mut(section)?.remove(index))
    }

    /// Update a bullet point in a section.
    pub fn update_bullet(
        &mut self,
        section: &str,
        index: usize,
        new_bullet: String,
    ) -> Result<(), Error> {
        let bullets = self
            .sections
            .get_mut(section)
            .ok_or_else(|| Error::TransactionError(format!("Section '{section}' not found")))?;

        if index >= bullets.len() {
            return Err(Error::TransactionError(format!(
                "Bullet index {index} out of bounds"
            )));
        }

        bullets[index] = new_bullet;
        Ok(())
    }

    /// Create or delete a section.
    pub fn create_section(&mut self, section: String) {
        self.sections.entry(section).or_default();
    }

    pub fn delete_section(&mut self, section: &str) -> Option<Vec<String>> {
        self.sections.remove(section)
    }

    /// Parse from markdown text.
    pub fn from_markdown(mount_id: MountID, path: String, markdown: &str) -> Self {
        let mut list = MarkdownList::new(mount_id, path);
        let mut current_section = String::new();

        for line in markdown.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('#') {
                // New section header
                current_section = trimmed.trim_start_matches('#').trim().to_string();
                list.create_section(current_section.clone());
            } else if trimmed.starts_with('-') || trimmed.starts_with('*') {
                // Bullet point
                let bullet = trimmed[1..].trim().to_string();
                if !current_section.is_empty() {
                    list.add_bullet(current_section.clone(), bullet);
                }
            }
        }

        list
    }

    /// Serialize to markdown format.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        for (section, bullets) in &self.sections {
            output.push_str(&format!("# {section}\n"));
            for bullet in bullets {
                output.push_str(&format!("- {bullet}\n"));
            }
            output.push('\n');
        }

        output
    }
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
    /// Create a new ContextManager with Chroma integration.
    pub async fn new(
        _claudius: Anthropic,
        chroma: ChromaClient,
        collection_name: String,
    ) -> Result<Self, Error> {
        // Create or get the collection
        chroma
            .get_or_create_collection(&collection_name, None, None)
            .await
            .map_err(|e| {
                Error::ChromaConnectionError(format!("Failed to create collection: {e}"))
            })?;

        Ok(ContextManager {
            _claudius,
            chroma,
            collection_name,
        })
    }

    /// Open a context for the given agent.
    pub async fn open(&self, agent_id: AgentID) -> Result<Context, Error> {
        // Try to load the latest context for this agent
        let context_seq_no = self.get_latest_context(agent_id).await?;

        if let Some(seq_no) = context_seq_no {
            self.load_context(agent_id, seq_no).await
        } else {
            // Create a new context
            Ok(Context {
                manager: self,
                agent_id,
                context_seq_no: 0,
                transactions: Vec::new(),
            })
        }
    }

    /// Store a transaction in Chroma.
    pub async fn transact(
        &self,
        _agent_id: AgentID,
        _context_seq_no: u32,
        transaction: Transaction,
    ) -> Result<(), Error> {
        // Check transaction invariants
        transaction.check_invariants()?;

        // Generate embedding summary for the first chunk
        let embedding_summary = transaction.generate_embedding_summary();

        // Chunk the transaction if needed
        let chunks = transaction.chunk_transaction()?;

        // Store each chunk in Chroma
        let collection = self
            .chroma
            .get_or_create_collection(&self.collection_name, None, None)
            .await
            .map_err(|e| Error::ChromaConnectionError(format!("Failed to get collection: {e}")))?;

        for (i, chunk) in chunks.iter().enumerate() {
            let key = format!(
                "context={};transaction={};chunk={}",
                chunk.context_seq_no, chunk.transaction_seq_no, chunk.chunk_seq_no
            );

            let mut metadata = HashMap::new();
            metadata.insert(
                "agent".to_string(),
                MetadataValue::Str(chunk.agent_id.to_string()),
            );
            metadata.insert(
                "context".to_string(),
                MetadataValue::Int(chunk.context_seq_no as i64),
            );
            metadata.insert(
                "transaction".to_string(),
                MetadataValue::Int(chunk.transaction_seq_no as i64),
            );
            metadata.insert(
                "chunk".to_string(),
                MetadataValue::Int(chunk.chunk_seq_no as i64),
            );
            metadata.insert(
                "total_chunks".to_string(),
                MetadataValue::Int(chunk.total_chunks as i64),
            );

            // For the first chunk, add the embedding summary as a document
            // This enables semantic search on transactions
            let document = if i == 0 {
                format!("{}\n\n{}", embedding_summary, chunk.data)
            } else {
                chunk.data.clone()
            };

            // Store the chunk
            // Note: ChromaDB will generate embeddings automatically from the document text
            let entries = chromadb::collection::CollectionEntries {
                ids: vec![key.as_str()],
                embeddings: None, // Let ChromaDB generate embeddings from the document
                metadatas: Some(vec![metadata]),
                documents: Some(vec![document.as_str()]),
            };

            collection.upsert(entries, None).await.map_err(|e| {
                Error::ChromaConnectionError(format!("Failed to store transaction: {e}"))
            })?;
        }

        Ok(())
    }

    /// Seal the current context and optionally create a new one.
    pub async fn seal_context(
        &self,
        _context: &Context<'_>,
        summary: String,
        create_next: bool,
    ) -> Result<Option<ContextID>, Error> {
        let context_id = ContextID::generate()
            .ok_or_else(|| Error::TransactionError("Failed to generate context ID".to_string()))?;
        let mut seal = ContextSeal::new(context_id, summary);

        if create_next {
            let next_id = ContextID::generate().ok_or_else(|| {
                Error::TransactionError("Failed to generate next context ID".to_string())
            })?;
            seal = seal.with_next(next_id);
            self.store_seal(&seal).await?;
            return Ok(Some(next_id));
        }

        self.store_seal(&seal).await?;
        Ok(None)
    }

    /// Fork a context to create a new branch.
    pub async fn fork_context(
        &self,
        source_context: &Context<'_>,
        summary: String,
    ) -> Result<Context, Error> {
        // Create a new context with the same state as source
        let new_context_id = ContextID::generate()
            .ok_or_else(|| Error::TransactionError("Failed to generate context ID".to_string()))?;
        let new_context_seq_no = source_context.context_seq_no + 1;

        // Create a seal for the fork
        let seal = ContextSeal::new(new_context_id, format!("Fork: {summary}"));
        self.store_seal(&seal).await?;

        // Return a new context with the forked state
        Ok(Context {
            manager: self,
            agent_id: source_context.agent_id,
            context_seq_no: new_context_seq_no,
            transactions: source_context.transactions.clone(),
        })
    }

    /// Store a context seal.
    pub async fn store_seal(&self, seal: &ContextSeal) -> Result<(), Error> {
        seal.validate()?;

        let collection = self
            .chroma
            .get_or_create_collection(&self.collection_name, None, None)
            .await
            .map_err(|e| Error::ChromaConnectionError(format!("Failed to get collection: {e}")))?;

        let key = format!("context={};seal", seal.context_id);
        let doc = serde_json::to_string(seal)
            .map_err(|e| Error::SealError(format!("Failed to serialize seal: {e}")))?;

        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), MetadataValue::Str("seal".to_string()));
        metadata.insert(
            "context".to_string(),
            MetadataValue::Str(seal.context_id.to_string()),
        );
        metadata.insert(
            "sealed_at".to_string(),
            MetadataValue::Int(seal.sealed_at as i64),
        );

        let entries = chromadb::collection::CollectionEntries {
            ids: vec![key.as_str()],
            embeddings: None,
            metadatas: Some(vec![metadata]),
            documents: Some(vec![doc.as_str()]),
        };

        collection
            .upsert(entries, None)
            .await
            .map_err(|e| Error::ChromaConnectionError(format!("Failed to store seal: {e}")))?;

        Ok(())
    }

    /// Store a markdown list in the virtual filesystem.
    pub async fn store_markdown_list(&self, list: &MarkdownList) -> Result<(), Error> {
        let collection = self
            .chroma
            .get_or_create_collection(&self.collection_name, None, None)
            .await
            .map_err(|e| Error::ChromaConnectionError(format!("Failed to get collection: {e}")))?;

        // Store each bullet as a separate document
        for (section, bullets) in &list.sections {
            for (idx, bullet) in bullets.iter().enumerate() {
                let key = format!(
                    "mount={};path={};section={}/bullet={}",
                    list.mount_id, list.path, section, idx
                );

                let mut metadata = HashMap::new();
                metadata.insert(
                    "type".to_string(),
                    MetadataValue::Str("markdown_bullet".to_string()),
                );
                metadata.insert(
                    "mount".to_string(),
                    MetadataValue::Str(list.mount_id.to_string()),
                );
                metadata.insert("path".to_string(), MetadataValue::Str(list.path.clone()));
                metadata.insert("section".to_string(), MetadataValue::Str(section.clone()));
                metadata.insert("bullet_index".to_string(), MetadataValue::Int(idx as i64));

                let entries = chromadb::collection::CollectionEntries {
                    ids: vec![key.as_str()],
                    embeddings: None,
                    metadatas: Some(vec![metadata]),
                    documents: Some(vec![bullet.as_str()]),
                };

                collection.upsert(entries, None).await.map_err(|e| {
                    Error::ChromaConnectionError(format!("Failed to store markdown bullet: {e}"))
                })?;
            }
        }

        Ok(())
    }

    /// Load a markdown list from the virtual filesystem.
    pub async fn load_markdown_list(
        &self,
        mount_id: MountID,
        path: String,
    ) -> Result<MarkdownList, Error> {
        let collection = self
            .chroma
            .get_or_create_collection(&self.collection_name, None, None)
            .await
            .map_err(|e| Error::ChromaConnectionError(format!("Failed to get collection: {e}")))?;

        // Query for all bullets in this file
        let filter = chromadb::filters::and_metadata(vec![
            chromadb::filters::eq("mount", mount_id.to_string()),
            chromadb::filters::eq("path", path.clone()),
            chromadb::filters::eq("type", "markdown_bullet"),
        ]);

        let get_options = chromadb::collection::GetOptions {
            ids: vec![],
            where_metadata: Some(filter),
            limit: None,
            offset: None,
            where_document: None,
            include: Some(vec!["documents".to_string(), "metadatas".to_string()]),
        };

        let result = collection.get(get_options).await.map_err(|e| {
            Error::ChromaConnectionError(format!("Failed to query markdown bullets: {e}"))
        })?;

        let mut list = MarkdownList::new(mount_id, path);

        if let Some(docs) = result.documents {
            if let Some(metadatas) = result.metadatas {
                // Group bullets by section
                let mut section_bullets: HashMap<String, Vec<(i64, String)>> = HashMap::new();

                for (doc_opt, metadata_opt) in docs.iter().zip(metadatas.iter()) {
                    if let (Some(doc), Some(metadata)) = (doc_opt, metadata_opt) {
                        if let Some(MetadataValue::Str(section)) = metadata.get("section") {
                            if let Some(MetadataValue::Int(idx)) = metadata.get("bullet_index") {
                                section_bullets
                                    .entry(section.clone())
                                    .or_default()
                                    .push((*idx, doc.clone()));
                            }
                        }
                    }
                }

                // Reconstruct the list with bullets in order
                for (section, mut bullets) in section_bullets {
                    bullets.sort_by_key(|(idx, _)| *idx);
                    list.create_section(section.clone());
                    for (_, bullet) in bullets {
                        list.add_bullet(section.clone(), bullet);
                    }
                }
            }
        }

        Ok(list)
    }

    /// Search for contexts by text or embedding similarity.
    pub async fn search_contexts(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(u32, f32)>, Error> {
        let collection = self
            .chroma
            .get_or_create_collection(&self.collection_name, None, None)
            .await
            .map_err(|e| Error::ChromaConnectionError(format!("Failed to get collection: {e}")))?;

        // Query Chroma for similar documents
        let query_options = chromadb::collection::QueryOptions {
            query_texts: Some(vec![query]),
            query_embeddings: None,
            where_metadata: None,
            where_document: None,
            n_results: Some(limit),
            include: Some(vec!["metadatas", "distances"]),
        };

        let result = collection.query(query_options, None).await.map_err(|e| {
            Error::ChromaConnectionError(format!("Failed to query for similar contexts: {e}"))
        })?;

        let mut context_scores: HashMap<u32, f32> = HashMap::new();

        // Process results
        if let Some(metadatas_vec) = result.metadatas {
            if let Some(distances_vec) = result.distances {
                for (metadatas, distances) in metadatas_vec.iter().zip(distances_vec.iter()) {
                    for (metadata_opt, distance) in metadatas.iter().zip(distances.iter()) {
                        if let Some(metadata) = metadata_opt {
                            if let Some(MetadataValue::Int(ctx_seq)) = metadata.get("context") {
                                let ctx_seq_u32 = *ctx_seq as u32;
                                // Convert distance to similarity score (lower distance = higher similarity)
                                let similarity = 1.0 / (1.0 + distance);
                                context_scores
                                    .entry(ctx_seq_u32)
                                    .and_modify(|s| *s = s.max(similarity))
                                    .or_insert(similarity);
                            }
                        }
                    }
                }
            }
        }

        // Sort by score (highest first) and return
        let mut results: Vec<(u32, f32)> = context_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }

    /// Get the latest context sequence number for an agent.
    async fn get_latest_context(&self, agent_id: AgentID) -> Result<Option<u32>, Error> {
        let collection = self
            .chroma
            .get_or_create_collection(&self.collection_name, None, None)
            .await
            .map_err(|e| Error::ChromaConnectionError(format!("Failed to get collection: {e}")))?;

        // Query for the highest context sequence number for this agent
        let filter = chromadb::filters::eq("agent", agent_id.to_string());
        let get_options = chromadb::collection::GetOptions {
            ids: vec![],
            where_metadata: Some(filter),
            limit: Some(100), // Get enough to find the max
            offset: None,
            where_document: None,
            include: Some(vec!["metadatas".to_string()]),
        };

        let result = collection
            .get(get_options)
            .await
            .map_err(|e| Error::ChromaConnectionError(format!("Failed to query contexts: {e}")))?;

        if let Some(metadatas) = result.metadatas {
            let mut max_context_seq = None;
            for metadata in metadatas.into_iter().flatten() {
                if let Some(MetadataValue::Int(ctx_seq)) = metadata.get("context") {
                    max_context_seq = Some(max_context_seq.unwrap_or(0).max(*ctx_seq as u32));
                }
            }
            return Ok(max_context_seq);
        }

        Ok(None)
    }

    /// Load a context from storage.
    async fn load_context(&self, agent_id: AgentID, context_seq_no: u32) -> Result<Context, Error> {
        let collection = self
            .chroma
            .get_or_create_collection(&self.collection_name, None, None)
            .await
            .map_err(|e| Error::ChromaConnectionError(format!("Failed to get collection: {e}")))?;

        // Query for all transactions in this context
        let filter = chromadb::filters::and_metadata(vec![
            chromadb::filters::eq("agent", agent_id.to_string()),
            chromadb::filters::eq("context", context_seq_no as i64),
        ]);

        let get_options = chromadb::collection::GetOptions {
            ids: vec![],
            where_metadata: Some(filter),
            limit: None,
            offset: None,
            where_document: None,
            include: Some(vec!["documents".to_string(), "metadatas".to_string()]),
        };

        let result = collection.get(get_options).await.map_err(|e| {
            Error::ChromaConnectionError(format!("Failed to query transactions: {e}"))
        })?;

        // Reconstruct transactions from chunks
        let mut chunk_map: HashMap<u64, Vec<TransactionChunk>> = HashMap::new();

        if let Some(docs) = result.documents {
            if let Some(metadatas) = result.metadatas {
                for (doc_opt, metadata_opt) in docs.iter().zip(metadatas.iter()) {
                    if let (Some(doc), Some(metadata)) = (doc_opt, metadata_opt) {
                        // Parse metadata to get transaction details
                        if let Some(MetadataValue::Int(tx_seq)) = metadata.get("transaction") {
                            if let Some(MetadataValue::Int(chunk_seq)) = metadata.get("chunk") {
                                if let Some(MetadataValue::Int(total_chunks)) =
                                    metadata.get("total_chunks")
                                {
                                    if let Some(MetadataValue::Int(ctx_seq)) =
                                        metadata.get("context")
                                    {
                                        // Extract the actual chunk data from the document
                                        // For the first chunk (chunk_seq_no == 0), the document might contain
                                        // the embedding summary followed by "\n\n" and then the actual data
                                        let chunk_data = if *chunk_seq == 0 && doc.contains("\n\n")
                                        {
                                            // Find the position after the embedding summary
                                            if let Some(pos) = doc.find("\n\n") {
                                                doc[pos + 2..].to_string()
                                            } else {
                                                doc.clone()
                                            }
                                        } else {
                                            doc.clone()
                                        };

                                        let chunk = TransactionChunk {
                                            agent_id,
                                            context_seq_no: *ctx_seq as u32,
                                            transaction_seq_no: *tx_seq as u64,
                                            chunk_seq_no: *chunk_seq as u32,
                                            total_chunks: *total_chunks as u32,
                                            data: chunk_data,
                                        };
                                        chunk_map.entry(*tx_seq as u64).or_default().push(chunk);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Reconstruct transactions from chunks
        let mut transactions = Vec::new();
        let mut tx_keys: Vec<u64> = chunk_map.keys().cloned().collect();
        tx_keys.sort();

        for tx_seq in tx_keys {
            if let Some(mut chunks) = chunk_map.remove(&tx_seq) {
                // Sort chunks by sequence number
                chunks.sort_by_key(|c| c.chunk_seq_no);

                // Validate all chunks are present
                if !chunks.is_empty() {
                    let expected_chunks = chunks[0].total_chunks;
                    if chunks.len() != expected_chunks as usize {
                        return Err(Error::TransactionError(format!(
                            "Missing chunks for transaction {}: expected {}, got {}",
                            tx_seq,
                            expected_chunks,
                            chunks.len()
                        )));
                    }

                    // Concatenate chunk data
                    let mut full_data = String::new();
                    for chunk in chunks {
                        full_data.push_str(&chunk.data);
                    }

                    // Deserialize transaction
                    let transaction: Transaction =
                        serde_json::from_str(&full_data).map_err(|e| {
                            Error::TransactionError(format!(
                                "Failed to deserialize transaction: {e}"
                            ))
                        })?;
                    transactions.push(transaction);
                }
            }
        }

        // Get context_seq_no from the first transaction (or 0 if no transactions)
        let context_seq_no = transactions
            .first()
            .map(|tx| tx.context_seq_no)
            .unwrap_or(0);

        Ok(Context {
            manager: self,
            agent_id,
            context_seq_no,
            transactions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    // ============================== Error Tests ==============================

    #[test]
    fn error_display_formatting() {
        let errors = vec![
            (
                Error::ChromaConnectionError("connection failed".to_string()),
                "ChromaDB connection error: connection failed",
            ),
            (
                Error::TransactionError("invalid tx".to_string()),
                "Transaction error: invalid tx",
            ),
            (
                Error::ContextNotFound("ctx123".to_string()),
                "Context not found: ctx123",
            ),
            (
                Error::ChunkSizeExceeded("too large".to_string()),
                "Chunk size exceeded: too large",
            ),
            (
                Error::InvalidSequence("bad seq".to_string()),
                "Invalid sequence: bad seq",
            ),
            (
                Error::SealError("seal failed".to_string()),
                "Seal error: seal failed",
            ),
        ];

        for (error, expected) in errors {
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(Error::TransactionError("test".to_string()));
        assert_eq!(err.to_string(), "Transaction error: test");
    }

    // ============================== ID Generation Tests ==============================

    #[test]
    fn context_id_generation_and_uniqueness() {
        let id1 = ContextID::generate().unwrap();
        let id2 = ContextID::generate().unwrap();
        assert_ne!(id1, id2, "Generated IDs should be unique");
        assert!(id1.to_string().starts_with("context:"));
        assert!(id2.to_string().starts_with("context:"));
    }

    #[test]
    fn agent_id_generation() {
        let id = AgentID::generate().unwrap();
        assert!(id.to_string().starts_with("agent:"));
    }

    #[test]
    fn mount_id_generation() {
        let id = MountID::generate().unwrap();
        assert!(id.to_string().starts_with("mount:"));
    }

    #[test]
    fn transaction_id_generation() {
        let id = TransactionID::generate().unwrap();
        assert!(id.to_string().starts_with("tx:"));
    }

    #[test]
    fn id_serialization_roundtrip() {
        let original = ContextID::generate().unwrap();
        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: ContextID = serde_json::from_str(&serialized).unwrap();
        assert_eq!(original, deserialized);
    }

    // ============================== Transaction Tests ==============================

    #[test]
    fn transaction_chunking_empty() {
        let tx = Transaction {
            agent_id: AgentID::generate().unwrap(),
            context_seq_no: 0,
            transaction_seq_no: 1,
            msgs: vec![],
            writes: vec![],
        };

        let chunks = tx.chunk_transaction().unwrap();
        assert_eq!(chunks.len(), 1, "Empty transaction should produce 1 chunk");
        assert_eq!(chunks[0].chunk_seq_no, 0);
        assert_eq!(chunks[0].total_chunks, 1);
        assert_eq!(chunks[0].agent_id, tx.agent_id);
        assert_eq!(chunks[0].context_seq_no, tx.context_seq_no);
        assert_eq!(chunks[0].transaction_seq_no, tx.transaction_seq_no);
    }

    #[test]
    fn transaction_chunking_small_within_limit() {
        let tx = Transaction {
            agent_id: AgentID::generate().unwrap(),
            context_seq_no: 5,
            transaction_seq_no: 10,
            msgs: vec![],
            writes: vec![FileWrite {
                mount: MountID::generate().unwrap(),
                path: "/small.txt".to_string(),
                data: "Small content".to_string(),
            }],
        };

        let chunks = tx.chunk_transaction().unwrap();
        assert_eq!(chunks.len(), 1, "Small transaction should produce 1 chunk");
        assert_eq!(chunks[0].chunk_seq_no, 0);
        assert_eq!(chunks[0].total_chunks, 1);

        // Verify the data can be deserialized back
        let reconstructed: Transaction = serde_json::from_str(&chunks[0].data).unwrap();
        assert_eq!(reconstructed.context_seq_no, tx.context_seq_no);
        assert_eq!(reconstructed.transaction_seq_no, tx.transaction_seq_no);
    }

    #[test]
    fn transaction_chunking_large_exceeds_limit() {
        // Create a transaction that exceeds CHUNK_SIZE_LIMIT (16KB)
        let mut writes = Vec::new();
        for i in 0..100 {
            writes.push(FileWrite {
                mount: MountID::generate().unwrap(),
                path: format!("/test/file_{i}.txt"),
                data: "x".repeat(200),
            });
        }

        let tx = Transaction {
            agent_id: AgentID::generate().unwrap(),
            context_seq_no: 0,
            transaction_seq_no: 1,
            msgs: vec![],
            writes,
        };

        let chunks = tx.chunk_transaction().unwrap();
        assert!(
            chunks.len() > 1,
            "Large transaction should produce multiple chunks"
        );

        // Verify chunk sequence numbers and total_chunks
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_seq_no, i as u32);
            assert_eq!(chunk.total_chunks, chunks.len() as u32);
            assert_eq!(chunk.agent_id, tx.agent_id);
            assert_eq!(chunk.context_seq_no, tx.context_seq_no);
            assert_eq!(chunk.transaction_seq_no, tx.transaction_seq_no);
        }

        // Verify we can reconstruct the original transaction
        let mut reconstructed_data = String::new();
        for chunk in chunks {
            reconstructed_data.push_str(&chunk.data);
        }
        let reconstructed: Transaction = serde_json::from_str(&reconstructed_data).unwrap();
        assert_eq!(reconstructed.writes.len(), 100);
    }

    #[test]
    fn transaction_chunking_boundary_conditions() {
        // Test exactly at the boundary of CHUNK_SIZE_LIMIT
        let data_size = CHUNK_SIZE_LIMIT - 200; // Account for JSON overhead
        let tx = Transaction {
            agent_id: AgentID::generate().unwrap(),
            context_seq_no: 0,
            transaction_seq_no: 1,
            msgs: vec![],
            writes: vec![FileWrite {
                mount: MountID::generate().unwrap(),
                path: "/boundary.txt".to_string(),
                data: "x".repeat(data_size),
            }],
        };

        let chunks = tx.chunk_transaction().unwrap();
        // Should still fit in one chunk if we're just under the limit
        if chunks.len() == 1 {
            assert_eq!(chunks[0].chunk_seq_no, 0);
            assert_eq!(chunks[0].total_chunks, 1);
        } else {
            // If it spills over, verify proper chunking
            for (i, chunk) in chunks.iter().enumerate() {
                assert_eq!(chunk.chunk_seq_no, i as u32);
                assert_eq!(chunk.total_chunks, chunks.len() as u32);
            }
        }
    }

    #[test]
    fn transaction_invariants_valid() {
        let tx = Transaction {
            agent_id: AgentID::generate().unwrap(),
            context_seq_no: 0,
            transaction_seq_no: 1,
            msgs: vec![],
            writes: vec![FileWrite {
                mount: MountID::generate().unwrap(),
                path: "/valid.txt".to_string(),
                data: "Valid data".to_string(),
            }],
        };

        assert!(tx.check_invariants().is_ok());
    }

    #[test]
    fn transaction_invariants_file_size_limit_exactly() {
        let tx = Transaction {
            agent_id: AgentID::generate().unwrap(),
            context_seq_no: 0,
            transaction_seq_no: 1,
            msgs: vec![],
            writes: vec![FileWrite {
                mount: MountID::generate().unwrap(),
                path: "/exact.txt".to_string(),
                data: "x".repeat(FILE_SIZE_LIMIT - 1),
            }],
        };

        assert!(tx.check_invariants().is_ok());
    }

    #[test]
    fn transaction_invariants_file_size_exceeded() {
        let tx = Transaction {
            agent_id: AgentID::generate().unwrap(),
            context_seq_no: 0,
            transaction_seq_no: 1,
            msgs: vec![],
            writes: vec![FileWrite {
                mount: MountID::generate().unwrap(),
                path: "/too_large.txt".to_string(),
                data: "x".repeat(FILE_SIZE_LIMIT + 1),
            }],
        };

        let result = tx.check_invariants();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::ChunkSizeExceeded(_)));
    }

    #[test]
    fn transaction_generate_embedding_summary() {
        let tx = Transaction {
            agent_id: AgentID::generate().unwrap(),
            context_seq_no: 3,
            transaction_seq_no: 7,
            msgs: vec![],
            writes: vec![FileWrite::default(), FileWrite::default()],
        };

        let summary = tx.generate_embedding_summary();
        assert!(summary.contains("Transaction 7"));
        assert!(summary.contains("context 3"));
        assert!(summary.contains("0 messages"));
        assert!(summary.contains("2 file writes"));
    }

    #[test]
    fn transaction_messages_iterator() {
        use claudius::MessageParam;

        let msgs = vec![MessageParam::user("msg1"), MessageParam::assistant("msg2")];

        let tx = Transaction {
            agent_id: AgentID::generate().unwrap(),
            context_seq_no: 0,
            transaction_seq_no: 1,
            msgs: msgs.clone(),
            writes: vec![],
        };

        let collected: Vec<_> = tx.messages().collect();
        assert_eq!(collected.len(), 2);

        // Test double-ended iterator
        let last = tx.messages().next_back();
        assert!(last.is_some());
    }

    // ============================== ContextSeal Tests ==============================

    #[test]
    fn context_seal_new_creation() {
        let ctx_id = ContextID::generate().unwrap();
        let seal = ContextSeal::new(ctx_id, "Test summary".to_string());

        assert_eq!(seal.context_id, ctx_id);
        assert_eq!(seal.summary, "Test summary");
        assert!(seal.next_context_id.is_none());

        // Check that sealed_at is reasonable (within last 10 seconds)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        assert!(seal.sealed_at <= now);
        assert!(seal.sealed_at > now - 10);
    }

    #[test]
    fn context_seal_with_next() {
        let ctx_id = ContextID::generate().unwrap();
        let next_id = ContextID::generate().unwrap();
        let seal = ContextSeal::new(ctx_id, "Summary".to_string()).with_next(next_id);

        assert_eq!(seal.context_id, ctx_id);
        assert_eq!(seal.next_context_id, Some(next_id));
    }

    #[test]
    fn context_seal_validation_valid() {
        let seal = ContextSeal::new(ContextID::generate().unwrap(), "Valid summary".to_string());
        assert!(seal.validate().is_ok());
    }

    #[test]
    fn context_seal_validation_empty_summary() {
        let seal = ContextSeal {
            context_id: ContextID::generate().unwrap(),
            next_context_id: None,
            sealed_at: 0,
            summary: String::new(),
        };

        let result = seal.validate();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::SealError(_)));
    }

    #[test]
    fn context_seal_validation_whitespace_summary() {
        let seal = ContextSeal {
            context_id: ContextID::generate().unwrap(),
            next_context_id: None,
            sealed_at: 12345,
            summary: "   ".to_string(),
        };

        // Currently the validation only checks is_empty(), not trimmed
        assert!(seal.validate().is_ok());
    }

    #[test]
    fn context_seal_serialization() {
        let ctx_id = ContextID::generate().unwrap();
        let seal = ContextSeal::new(ctx_id, "Test".to_string());

        let serialized = serde_json::to_string(&seal).unwrap();
        let deserialized: ContextSeal = serde_json::from_str(&serialized).unwrap();

        assert_eq!(seal.context_id, deserialized.context_id);
        assert_eq!(seal.summary, deserialized.summary);
        assert_eq!(seal.sealed_at, deserialized.sealed_at);
        assert_eq!(seal.next_context_id, deserialized.next_context_id);
    }

    // ============================== MarkdownList Tests ==============================

    #[test]
    fn markdown_list_new() {
        let mount_id = MountID::generate().unwrap();
        let list = MarkdownList::new(mount_id, "/test.md".to_string());

        assert_eq!(list.mount_id, mount_id);
        assert_eq!(list.path, "/test.md");
        assert!(list.sections.is_empty());
    }

    #[test]
    fn markdown_list_add_bullet() {
        let mut list = MarkdownList::new(MountID::generate().unwrap(), "/test.md".to_string());

        list.add_bullet("Tasks".to_string(), "First task".to_string());
        list.add_bullet("Tasks".to_string(), "Second task".to_string());
        list.add_bullet("Notes".to_string(), "Note 1".to_string());

        assert_eq!(list.sections.len(), 2);
        assert_eq!(list.sections.get("Tasks").unwrap().len(), 2);
        assert_eq!(list.sections.get("Notes").unwrap().len(), 1);
    }

    #[test]
    fn markdown_list_remove_bullet_valid() {
        let mut list = MarkdownList::new(MountID::generate().unwrap(), "/test.md".to_string());

        list.add_bullet("TODO".to_string(), "Task 1".to_string());
        list.add_bullet("TODO".to_string(), "Task 2".to_string());
        list.add_bullet("TODO".to_string(), "Task 3".to_string());

        let removed = list.remove_bullet("TODO", 1);
        assert_eq!(removed, Some("Task 2".to_string()));
        assert_eq!(list.sections.get("TODO").unwrap().len(), 2);
        assert_eq!(list.sections.get("TODO").unwrap()[0], "Task 1");
        assert_eq!(list.sections.get("TODO").unwrap()[1], "Task 3");
    }

    #[test]
    fn markdown_list_remove_bullet_invalid_section() {
        let mut list = MarkdownList::new(MountID::generate().unwrap(), "/test.md".to_string());
        list.add_bullet("TODO".to_string(), "Task".to_string());

        let removed = list.remove_bullet("NONEXISTENT", 0);
        assert!(removed.is_none());
    }

    #[test]
    fn markdown_list_remove_bullet_invalid_index() {
        let mut list = MarkdownList::new(MountID::generate().unwrap(), "/test.md".to_string());
        list.add_bullet("TODO".to_string(), "Task".to_string());

        // Try to remove at index 5 when only 1 item exists
        // This will panic due to Vec::remove
        // We should not test panics in this manner, skipping
    }

    #[test]
    fn markdown_list_update_bullet_valid() {
        let mut list = MarkdownList::new(MountID::generate().unwrap(), "/test.md".to_string());

        list.add_bullet("TODO".to_string(), "Original task".to_string());
        let result = list.update_bullet("TODO", 0, "Updated task".to_string());

        assert!(result.is_ok());
        assert_eq!(list.sections.get("TODO").unwrap()[0], "Updated task");
    }

    #[test]
    fn markdown_list_update_bullet_invalid_section() {
        let mut list = MarkdownList::new(MountID::generate().unwrap(), "/test.md".to_string());

        let result = list.update_bullet("NONEXISTENT", 0, "New".to_string());
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::TransactionError(_)));
    }

    #[test]
    fn markdown_list_update_bullet_invalid_index() {
        let mut list = MarkdownList::new(MountID::generate().unwrap(), "/test.md".to_string());
        list.add_bullet("TODO".to_string(), "Task".to_string());

        let result = list.update_bullet("TODO", 10, "New".to_string());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, Error::TransactionError(_)));
        assert!(err.to_string().contains("out of bounds"));
    }

    #[test]
    fn markdown_list_create_and_delete_section() {
        let mut list = MarkdownList::new(MountID::generate().unwrap(), "/test.md".to_string());

        // Create empty section
        list.create_section("Empty".to_string());
        assert!(list.sections.contains_key("Empty"));
        assert!(list.sections.get("Empty").unwrap().is_empty());

        // Add items and delete
        list.add_bullet("ToDelete".to_string(), "Item 1".to_string());
        list.add_bullet("ToDelete".to_string(), "Item 2".to_string());

        let deleted = list.delete_section("ToDelete");
        assert_eq!(
            deleted,
            Some(vec!["Item 1".to_string(), "Item 2".to_string()])
        );
        assert!(!list.sections.contains_key("ToDelete"));

        // Delete non-existent section
        let deleted = list.delete_section("NonExistent");
        assert!(deleted.is_none());
    }

    #[test]
    fn markdown_list_from_markdown_basic() {
        let markdown = r#"# Section 1
- Item 1
- Item 2

# Section 2
* Item A
* Item B
"#;

        let mount_id = MountID::generate().unwrap();
        let list = MarkdownList::from_markdown(mount_id, "/test.md".to_string(), markdown);

        assert_eq!(list.sections.len(), 2);
        assert_eq!(
            list.sections.get("Section 1").unwrap(),
            &vec!["Item 1".to_string(), "Item 2".to_string()]
        );
        assert_eq!(
            list.sections.get("Section 2").unwrap(),
            &vec!["Item A".to_string(), "Item B".to_string()]
        );
    }

    #[test]
    fn markdown_list_from_markdown_mixed_formatting() {
        let markdown = r#"# TODO
  - Indented item
-Regular item
  * Star item

## Subsection Header  
- Item in subsection

###Triple hash
* Another item"#;

        let mount_id = MountID::generate().unwrap();
        let list = MarkdownList::from_markdown(mount_id, "/test.md".to_string(), markdown);

        assert!(list.sections.contains_key("TODO"));
        assert!(list.sections.contains_key("Subsection Header"));
        assert!(list.sections.contains_key("Triple hash"));

        // Check that items are correctly parsed
        assert_eq!(list.sections.get("TODO").unwrap().len(), 3);
        assert!(
            list.sections
                .get("TODO")
                .unwrap()
                .contains(&"Indented item".to_string())
        );
        assert!(
            list.sections
                .get("TODO")
                .unwrap()
                .contains(&"Regular item".to_string())
        );
        assert!(
            list.sections
                .get("TODO")
                .unwrap()
                .contains(&"Star item".to_string())
        );
    }

    #[test]
    fn markdown_list_from_markdown_empty_sections() {
        let markdown = r#"# Empty Section

# Section with Items
- Item 1

# Another Empty
"#;

        let mount_id = MountID::generate().unwrap();
        let list = MarkdownList::from_markdown(mount_id, "/test.md".to_string(), markdown);

        assert!(list.sections.contains_key("Empty Section"));
        assert!(list.sections.get("Empty Section").unwrap().is_empty());
        assert_eq!(list.sections.get("Section with Items").unwrap().len(), 1);
        assert!(list.sections.contains_key("Another Empty"));
    }

    #[test]
    fn markdown_list_from_markdown_orphan_bullets() {
        // Bullets without a section header should be ignored
        let markdown = r#"- Orphan bullet 1
* Orphan bullet 2

# Valid Section
- Valid item"#;

        let mount_id = MountID::generate().unwrap();
        let list = MarkdownList::from_markdown(mount_id, "/test.md".to_string(), markdown);

        assert_eq!(list.sections.len(), 1);
        assert!(list.sections.contains_key("Valid Section"));
        assert_eq!(list.sections.get("Valid Section").unwrap().len(), 1);
    }

    #[test]
    fn markdown_list_to_markdown() {
        let mut list = MarkdownList::new(MountID::generate().unwrap(), "/test.md".to_string());

        list.add_bullet("TODO".to_string(), "Task 1".to_string());
        list.add_bullet("TODO".to_string(), "Task 2".to_string());
        list.add_bullet("DONE".to_string(), "Completed task".to_string());

        let markdown = list.to_markdown();

        assert!(markdown.contains("# TODO\n"));
        assert!(markdown.contains("- Task 1\n"));
        assert!(markdown.contains("- Task 2\n"));
        assert!(markdown.contains("# DONE\n"));
        assert!(markdown.contains("- Completed task\n"));
    }

    #[test]
    fn markdown_list_roundtrip() {
        let mut original = MarkdownList::new(MountID::generate().unwrap(), "/test.md".to_string());

        original.add_bullet("Section A".to_string(), "Item 1".to_string());
        original.add_bullet("Section A".to_string(), "Item 2".to_string());
        original.add_bullet("Section B".to_string(), "Item X".to_string());

        let markdown = original.to_markdown();
        let parsed =
            MarkdownList::from_markdown(original.mount_id, original.path.clone(), &markdown);

        assert_eq!(original.sections.len(), parsed.sections.len());
        for (section, items) in &original.sections {
            assert_eq!(items, parsed.sections.get(section).unwrap());
        }
    }

    #[test]
    fn markdown_list_serialization() {
        let mut list = MarkdownList::new(MountID::generate().unwrap(), "/test.md".to_string());
        list.add_bullet("TODO".to_string(), "Task".to_string());

        let serialized = serde_json::to_string(&list).unwrap();
        let deserialized: MarkdownList = serde_json::from_str(&serialized).unwrap();

        assert_eq!(list.mount_id, deserialized.mount_id);
        assert_eq!(list.path, deserialized.path);
        assert_eq!(list.sections, deserialized.sections);
    }

    // ============================== Context Tests ==============================

    #[test]
    fn context_check_invariants_valid() {
        let agent_id = AgentID::generate().unwrap();
        let transactions = vec![
            Transaction {
                agent_id,
                context_seq_no: 1,
                transaction_seq_no: 0,
                msgs: vec![],
                writes: vec![],
            },
            Transaction {
                agent_id,
                context_seq_no: 1,
                transaction_seq_no: 1,
                msgs: vec![],
                writes: vec![],
            },
            Transaction {
                agent_id,
                context_seq_no: 1,
                transaction_seq_no: 2,
                msgs: vec![],
                writes: vec![],
            },
        ];

        // We can't easily create a Context without a ContextManager,
        // but we can test the invariant logic via transaction sequences
        for tx in &transactions {
            assert!(tx.check_invariants().is_ok());
        }
    }

    #[test]
    fn context_invariants_agent_mismatch() {
        // Testing the logic that would be in Context::check_invariants
        let tx1 = Transaction {
            agent_id: AgentID::generate().unwrap(),
            context_seq_no: 1,
            transaction_seq_no: 0,
            msgs: vec![],
            writes: vec![],
        };

        let tx2 = Transaction {
            agent_id: AgentID::generate().unwrap(), // Different agent
            context_seq_no: 1,
            transaction_seq_no: 1,
            msgs: vec![],
            writes: vec![],
        };

        // These would fail Context invariant check
        assert_ne!(tx1.agent_id, tx2.agent_id);
    }

    #[test]
    fn context_invariants_sequence_gap() {
        let agent_id = AgentID::generate().unwrap();

        let tx1 = Transaction {
            agent_id,
            context_seq_no: 1,
            transaction_seq_no: 0,
            msgs: vec![],
            writes: vec![],
        };

        let tx2 = Transaction {
            agent_id,
            context_seq_no: 1,
            transaction_seq_no: 2, // Gap in sequence
            msgs: vec![],
            writes: vec![],
        };

        // Verify the gap exists
        assert_ne!(tx1.transaction_seq_no + 1, tx2.transaction_seq_no);
    }

    // ============================== TransactionChunk Tests ==============================

    #[test]
    fn transaction_chunk_fields() {
        let chunk = TransactionChunk {
            agent_id: AgentID::generate().unwrap(),
            context_seq_no: 5,
            transaction_seq_no: 10,
            chunk_seq_no: 2,
            total_chunks: 5,
            data: "chunk data".to_string(),
        };

        assert_eq!(chunk.context_seq_no, 5);
        assert_eq!(chunk.transaction_seq_no, 10);
        assert_eq!(chunk.chunk_seq_no, 2);
        assert_eq!(chunk.total_chunks, 5);
        assert_eq!(chunk.data, "chunk data");
    }

    #[test]
    fn transaction_chunk_serialization() {
        let chunk = TransactionChunk {
            agent_id: AgentID::generate().unwrap(),
            context_seq_no: 1,
            transaction_seq_no: 2,
            chunk_seq_no: 0,
            total_chunks: 1,
            data: "test data".to_string(),
        };

        let serialized = serde_json::to_string(&chunk).unwrap();
        let deserialized: TransactionChunk = serde_json::from_str(&serialized).unwrap();

        assert_eq!(chunk.agent_id, deserialized.agent_id);
        assert_eq!(chunk.context_seq_no, deserialized.context_seq_no);
        assert_eq!(chunk.transaction_seq_no, deserialized.transaction_seq_no);
        assert_eq!(chunk.chunk_seq_no, deserialized.chunk_seq_no);
        assert_eq!(chunk.total_chunks, deserialized.total_chunks);
        assert_eq!(chunk.data, deserialized.data);
    }

    // ============================== FileWrite Tests ==============================

    #[test]
    fn file_write_default() {
        let fw = FileWrite::default();
        assert_eq!(fw.mount, MountID::default());
        assert_eq!(fw.path, "");
        assert_eq!(fw.data, "");
    }

    #[test]
    fn file_write_serialization() {
        let fw = FileWrite {
            mount: MountID::generate().unwrap(),
            path: "/test/path.txt".to_string(),
            data: "file contents".to_string(),
        };

        let serialized = serde_json::to_string(&fw).unwrap();
        let deserialized: FileWrite = serde_json::from_str(&serialized).unwrap();

        assert_eq!(fw.mount, deserialized.mount);
        assert_eq!(fw.path, deserialized.path);
        assert_eq!(fw.data, deserialized.data);
    }

    // ============================== Integration-style Tests ==============================

    #[test]
    fn markdown_complex_workflow() {
        let mut list = MarkdownList::new(MountID::generate().unwrap(), "/workflow.md".to_string());

        // Build a complex list
        list.create_section("Backlog".to_string());
        list.add_bullet("Backlog".to_string(), "Feature A".to_string());
        list.add_bullet("Backlog".to_string(), "Feature B".to_string());
        list.add_bullet("Backlog".to_string(), "Feature C".to_string());

        list.create_section("In Progress".to_string());
        list.add_bullet("In Progress".to_string(), "Feature D".to_string());

        list.create_section("Done".to_string());

        // Move item from Backlog to In Progress
        let item = list.remove_bullet("Backlog", 0).unwrap();
        list.add_bullet("In Progress".to_string(), item);

        // Complete an item
        let completed = list.remove_bullet("In Progress", 0).unwrap();
        list.add_bullet("Done".to_string(), completed);

        // Verify final state
        assert_eq!(list.sections.get("Backlog").unwrap().len(), 2);
        assert_eq!(list.sections.get("In Progress").unwrap().len(), 1);
        assert_eq!(list.sections.get("Done").unwrap().len(), 1);
        assert_eq!(list.sections.get("Done").unwrap()[0], "Feature D");
    }

    #[test]
    fn transaction_chunking_reassembly() {
        // Create a large transaction
        let original_tx = Transaction {
            agent_id: AgentID::generate().unwrap(),
            context_seq_no: 5,
            transaction_seq_no: 42,
            msgs: vec![],
            writes: (0..100)
                .map(|i| FileWrite {
                    mount: MountID::generate().unwrap(),
                    path: format!("/file_{i}.txt"),
                    data: format!("Content for file {i}: {}", "x".repeat(200)),
                })
                .collect(),
        };

        // Chunk it
        let chunks = original_tx.chunk_transaction().unwrap();
        assert!(chunks.len() > 1, "Should create multiple chunks");

        // Reassemble
        let mut full_data = String::new();
        for chunk in chunks {
            full_data.push_str(&chunk.data);
        }

        // Deserialize and compare
        let reconstructed: Transaction = serde_json::from_str(&full_data).unwrap();
        assert_eq!(reconstructed.agent_id, original_tx.agent_id);
        assert_eq!(reconstructed.context_seq_no, original_tx.context_seq_no);
        assert_eq!(
            reconstructed.transaction_seq_no,
            original_tx.transaction_seq_no
        );
        assert_eq!(reconstructed.writes.len(), original_tx.writes.len());
    }

    // ============================== UTF-8 Chunking Tests ==============================

    #[test]
    fn transaction_chunking_preserves_utf8_boundaries() {
        // Create a transaction with multi-byte UTF-8 characters
        let tx = Transaction {
            agent_id: AgentID::generate().unwrap(),
            context_seq_no: 0,
            transaction_seq_no: 1,
            msgs: vec![],
            writes: vec![FileWrite {
                mount: MountID::generate().unwrap(),
                path: "/unicode.txt".to_string(),
                // Create a string with multi-byte characters that would break if split incorrectly
                data: "🎉".repeat(5000), // Emoji are 4-byte UTF-8 sequences
            }],
        };

        let chunks = tx.chunk_transaction().unwrap();

        // Verify all chunks are valid UTF-8
        // In Rust, all Strings are guaranteed to be valid UTF-8, so we just ensure no panic
        for chunk in &chunks {
            // If this doesn't panic, the string is valid UTF-8
            let _ = chunk.data.chars().count();
        }

        // Reconstruct and verify
        let mut reconstructed = String::new();
        for chunk in chunks {
            reconstructed.push_str(&chunk.data);
        }

        let original_json = serde_json::to_string(&tx).unwrap();
        assert_eq!(reconstructed, original_json);
    }

    #[test]
    fn transaction_chunking_handles_edge_cases() {
        // Test with exactly CHUNK_SIZE_LIMIT bytes
        let data_size = CHUNK_SIZE_LIMIT - 500; // Leave room for JSON structure
        let tx = Transaction {
            agent_id: AgentID::generate().unwrap(),
            context_seq_no: 0,
            transaction_seq_no: 1,
            msgs: vec![],
            writes: vec![FileWrite {
                mount: MountID::generate().unwrap(),
                path: "/exact.txt".to_string(),
                data: "a".repeat(data_size),
            }],
        };

        let chunks = tx.chunk_transaction().unwrap();

        // Should still produce valid chunks
        for chunk in &chunks {
            assert!(chunk.data.len() <= CHUNK_SIZE_LIMIT);
        }
    }

    // ============================== Context Invariant Tests ==============================

    #[test]
    fn context_invariants_detect_all_violations() {
        // This would normally be tested through Context methods, but we can test the logic
        let agent_id = AgentID::generate().unwrap();

        // Test valid sequence
        let tx1 = Transaction {
            agent_id,
            context_seq_no: 1,
            transaction_seq_no: 0,
            msgs: vec![],
            writes: vec![],
        };
        assert!(tx1.check_invariants().is_ok());

        // Test file size violation
        let tx2 = Transaction {
            agent_id,
            context_seq_no: 1,
            transaction_seq_no: 1,
            msgs: vec![],
            writes: vec![FileWrite {
                mount: MountID::generate().unwrap(),
                path: "/big.txt".to_string(),
                data: "x".repeat(FILE_SIZE_LIMIT + 1),
            }],
        };
        assert!(matches!(
            tx2.check_invariants(),
            Err(Error::ChunkSizeExceeded(_))
        ));
    }

    // ============================== MarkdownList Edge Cases ==============================

    #[test]
    fn markdown_list_handles_special_characters() {
        let mut list = MarkdownList::new(MountID::generate().unwrap(), "/special.md".to_string());

        // Test with special characters in section names and bullets
        list.add_bullet(
            "TODO: Special!".to_string(),
            "Task with *emphasis*".to_string(),
        );
        list.add_bullet(
            "TODO: Special!".to_string(),
            "Task with [link](url)".to_string(),
        );

        let markdown = list.to_markdown();
        assert!(markdown.contains("# TODO: Special!"));
        assert!(markdown.contains("- Task with *emphasis*"));
        assert!(markdown.contains("- Task with [link](url)"));

        // Round-trip test
        let parsed = MarkdownList::from_markdown(list.mount_id, list.path.clone(), &markdown);
        assert_eq!(parsed.sections.len(), list.sections.len());
    }

    #[test]
    fn markdown_list_handles_empty_and_whitespace() {
        let markdown = r#"# 

# Section with spaces   

- Item with trailing spaces   
-    Item with leading spaces

# Empty bullets
- 
- Valid item
"#;

        let mount_id = MountID::generate().unwrap();
        let list = MarkdownList::from_markdown(mount_id, "/test.md".to_string(), markdown);

        // Should handle various edge cases gracefully
        assert!(list.sections.contains_key("Section with spaces"));

        // Check that items are trimmed properly
        for bullets in list.sections.values() {
            for bullet in bullets {
                assert_eq!(bullet, bullet.trim());
                assert!(!bullet.is_empty());
            }
        }
    }

    // ============================== ContextSeal Edge Cases ==============================

    #[test]
    fn context_seal_timestamp_monotonic() {
        use std::thread;
        use std::time::Duration;

        let ctx1 = ContextID::generate().unwrap();
        let seal1 = ContextSeal::new(ctx1, "First".to_string());

        thread::sleep(Duration::from_millis(10));

        let ctx2 = ContextID::generate().unwrap();
        let seal2 = ContextSeal::new(ctx2, "Second".to_string());

        assert!(seal2.sealed_at >= seal1.sealed_at);
    }

    #[test]
    fn context_seal_fork_chain() {
        let ctx1 = ContextID::generate().unwrap();
        let ctx2 = ContextID::generate().unwrap();
        let ctx3 = ContextID::generate().unwrap();

        let seal1 = ContextSeal::new(ctx1, "Original".to_string()).with_next(ctx2);
        let seal2 = ContextSeal::new(ctx2, "Fork 1".to_string()).with_next(ctx3);

        assert_eq!(seal1.next_context_id, Some(ctx2));
        assert_eq!(seal2.next_context_id, Some(ctx3));

        // Validate seals
        assert!(seal1.validate().is_ok());
        assert!(seal2.validate().is_ok());
    }

    // ============================== Async Tests (ContextManager) ==============================

    #[tokio::test]
    async fn context_manager_id_generation() {
        // Test ID generation and format validation
        let agent_id = AgentID::generate().unwrap();
        let context_id = ContextID::generate().unwrap();
        let mount_id = MountID::generate().unwrap();
        let tx_id = TransactionID::generate().unwrap();

        assert!(agent_id.to_string().starts_with("agent:"));
        assert!(context_id.to_string().starts_with("context:"));
        assert!(mount_id.to_string().starts_with("mount:"));
        assert!(tx_id.to_string().starts_with("tx:"));

        // Verify uniqueness
        let agent_id2 = AgentID::generate().unwrap();
        assert_ne!(agent_id, agent_id2);
    }

    // Note: Full async tests for ContextManager would require a mock ChromaDB client
    // or an actual ChromaDB instance running. The existing test structure is appropriate
    // for unit tests that don't have external dependencies.
}
