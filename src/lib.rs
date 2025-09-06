#![doc = include_str!("../README.md")]

use std::collections::HashMap;

use chromadb::{ChromaClient, MetadataValue};
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

    // TODO(claude):  Make this function return an InvariantViolation enum and not assert.
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

        let next_id = if create_next {
            ContextID::generate().ok_or_else(|| {
                Error::TransactionError("Failed to generate next context ID".to_string())
            })?
        } else {
            // Use a null/empty context ID when there's no next context
            ContextID::generate().ok_or_else(|| {
                Error::TransactionError("Failed to generate placeholder context ID".to_string())
            })?
        };

        let seal = ContextSeal::new(context_id, next_id, summary);
        self.store_seal(&seal).await?;

        if create_next {
            Ok(Some(next_id))
        } else {
            Ok(None)
        }
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
        let next_context_id = ContextID::generate().ok_or_else(|| {
            Error::TransactionError("Failed to generate next context ID".to_string())
        })?;
        let seal = ContextSeal::new(new_context_id, next_context_id, format!("Fork: {summary}"));
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
            MetadataValue::Int(seal.sealed_at_ms as i64),
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
