use std::collections::HashMap;

use chromadb::{
    MetadataValue,
    collection::{ChromaCollection, CollectionEntries},
};

use crate::{AgentID, Transaction, TransactionID};

////////////////////////////////////////////// Errors //////////////////////////////////////////////

/// Error that can occur during transaction persistence operations.
#[derive(Debug)]
pub enum TransactionManagerError {
    /// Error connecting to or communicating with ChromaDB.
    ChromaError(String),
    /// Error chunking the transaction.
    ChunkingError(crate::TransactionSerializationError),
    /// GUID generation failed.
    GuidError,
}

impl std::fmt::Display for TransactionManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransactionManagerError::ChromaError(e) => write!(f, "ChromaDB error: {}", e),
            TransactionManagerError::ChunkingError(e) => {
                write!(f, "Transaction chunking error: {}", e)
            }
            TransactionManagerError::GuidError => write!(f, "Failed to generate GUID"),
        }
    }
}

impl std::error::Error for TransactionManagerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TransactionManagerError::ChromaError(_) => None,
            TransactionManagerError::ChunkingError(e) => Some(e),
            TransactionManagerError::GuidError => None,
        }
    }
}

impl From<crate::TransactionSerializationError> for TransactionManagerError {
    fn from(error: crate::TransactionSerializationError) -> Self {
        TransactionManagerError::ChunkingError(error)
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

//////////////////////////////////////// TransactionManager ////////////////////////////////////////

/// Manages transaction persistence using ChromaDB collections.
///
/// The TransactionManager follows the plan specified in PLAN.md:
/// 1. Uses rust-client to implement persistence
/// 2. Atomically adds all transaction chunks to a chroma collection with nonce metadata
/// 3. Can verify transaction persistence by checking if chunk[0] has the expected nonce
pub struct TransactionManager {
    collection: ChromaCollection,
}

impl TransactionManager {
    /// Create a new TransactionManager with a ChromaDB collection.
    pub fn new(collection: ChromaCollection) -> Self {
        TransactionManager { collection }
    }

    /// Persist a transaction by chunking it and atomically storing all chunks with a nonce.
    ///
    /// Returns the GUID that was used as the nonce if persistence succeeds.
    /// The nonce can later be used with `verify_persistence` to confirm the transaction was stored.
    pub async fn persist_transaction(
        &self,
        transaction: &Transaction,
    ) -> Result<String, TransactionManagerError> {
        // Generate a unique GUID for this persistence operation using TransactionID
        let nonce = TransactionID::generate()
            .ok_or(TransactionManagerError::GuidError)?
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

        // Create simple embeddings (zeros) for each chunk since we're using ChromaDB for metadata storage
        let embeddings: Vec<Vec<f32>> = chunks.iter().map(|_| vec![0.0; 384]).collect();

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
            .map_err(|e| TransactionManagerError::ChromaError(e.to_string()))?;

        Ok(nonce)
    }

    /// Verify that a transaction was successfully persisted by checking chunk[0] for the expected nonce.
    ///
    /// Returns `true` if chunk[0] exists and has the specified nonce, `false` otherwise.
    pub async fn verify_persistence(
        &self,
        transaction: &Transaction,
        expected_nonce: &str,
    ) -> Result<bool, TransactionManagerError> {
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
            .map_err(|e| TransactionManagerError::ChromaError(e.to_string()))?;

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
    async fn transaction_manager_creation() {
        let client = create_test_client().await;
        let collection = client
            .get_or_create_collection("test_transactions", None, None)
            .await
            .expect("Failed to create ChromaDB collection");
        let _manager = TransactionManager::new(collection);
        // Test passes if we can create the manager without panicking
    }

    #[tokio::test]
    async fn persist_and_verify_transaction() {
        let client = create_test_client().await;
        let collection = client
            .get_or_create_collection("test_transactions", None, None)
            .await
            .expect("Failed to create ChromaDB collection");
        let manager = TransactionManager::new(collection);
        let transaction = create_test_transaction();

        // Persist the transaction
        let nonce_result = manager.persist_transaction(&transaction).await;
        assert!(nonce_result.is_ok());

        let nonce = nonce_result.unwrap();
        assert!(!nonce.is_empty());

        // Verify the persistence
        let verification_result = manager.verify_persistence(&transaction, &nonce).await;
        assert!(verification_result.is_ok());
        assert!(verification_result.unwrap());

        // Verify with wrong nonce should return false
        let wrong_nonce = TransactionID::generate().unwrap().to_string();
        let wrong_verification = manager.verify_persistence(&transaction, &wrong_nonce).await;
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
        let manager = TransactionManager::new(collection);

        // Create a large transaction that will be chunked
        let large_content = "x".repeat(crate::CHUNK_SIZE_LIMIT * 2);
        let mut transaction = create_test_transaction();
        transaction.msgs.push(MessageParam {
            role: MessageRole::Assistant,
            content: large_content.into(),
        });

        let nonce = manager.persist_transaction(&transaction).await.unwrap();
        let verification = manager
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
        let manager = TransactionManager::new(collection);
        let transaction = create_test_transaction();
        let fake_nonce = TransactionID::generate().unwrap().to_string();

        let verification = manager
            .verify_persistence(&transaction, &fake_nonce)
            .await
            .unwrap();
        assert!(!verification);
    }
}
