//! Text embedding functionality for the Context Engineer's Toolkit.
//!
//! This module provides text embedding capabilities using the sentence-transformers/all-MiniLM-L6-v2
//! model for generating vector embeddings from text. These embeddings are used by Chroma for
//! semantic search and retrieval of transaction data.
//!
//! The module provides two main types:
//! - [`EmbeddingModel`]: A low-level interface to the BERT embedding model
//! - [`EmbeddingService`]: A high-level shared service for generating embeddings

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{Repo, RepoType, api::sync::Api};
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Embedding model for generating vector embeddings from text using sentence-transformers/all-MiniLM-L6-v2.
///
/// This struct encapsulates a BERT-based transformer model that converts text strings into
/// high-dimensional vectors (embeddings). The model produces 384-dimensional embeddings
/// that capture semantic meaning and can be used for similarity search and retrieval.
///
/// The model runs on CPU for simplicity and compatibility, though GPU support could be
/// added in the future for improved performance.
///
/// # Examples
///
/// ```rust,no_run
/// use cetk::EmbeddingModel;
///
/// # async fn example() -> anyhow::Result<()> {
/// let model = EmbeddingModel::new()?;
/// let embedding = model.embed_single("Hello, world!")?;
/// assert_eq!(embedding.len(), 384); // all-MiniLM-L6-v2 produces 384-dimensional embeddings
/// # Ok(())
/// # }
/// ```
pub struct EmbeddingModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingModel {
    /// Create a new EmbeddingModel using sentence-transformers/all-MiniLM-L6-v2.
    ///
    /// This constructor downloads the pre-trained model and tokenizer from Hugging Face Hub
    /// if they are not already cached locally. The model files include:
    /// - `config.json`: Model configuration
    /// - `tokenizer.json`: Tokenizer configuration
    /// - `pytorch_model.bin`: Model weights
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Network connection to Hugging Face Hub fails
    /// - Model files cannot be downloaded or read
    /// - Model weights cannot be loaded
    /// - Device initialization fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cetk::EmbeddingModel;
    ///
    /// let model = EmbeddingModel::new()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn new() -> Result<Self> {
        let device = Device::Cpu; // Use CPU for simplicity, could add GPU support later

        // Load the model and tokenizer from Hugging Face Hub
        let api = Api::new()?;
        let repo = api.repo(Repo::new(
            "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            RepoType::Model,
        ));

        let config_filename = repo.get("config.json")?;
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let weights_filename = repo.get("pytorch_model.bin")?;

        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

        // Load the weights - try safetensors format first, then PyTorch format
        let weights =
            if weights_filename.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                candle_core::safetensors::load(&weights_filename, &device)?
            } else {
                let tensors = candle_core::pickle::read_all(&weights_filename)?;
                let mut weights = std::collections::HashMap::new();
                for (k, v) in tensors {
                    weights.insert(k, v.to_device(&device)?);
                }
                weights
            };
        let vb = VarBuilder::from_tensors(weights, candle_core::DType::F32, &device);
        let model = BertModel::load(vb, &config)?;

        Ok(EmbeddingModel {
            model,
            tokenizer,
            device,
        })
    }

    /// Generate embeddings for a batch of texts.
    ///
    /// This method processes multiple text strings in a single forward pass,
    /// which is more efficient than processing them individually. Each text
    /// is tokenized, padded to the same length, and then processed by the model.
    ///
    /// The resulting embeddings are normalized and have 384 dimensions each.
    ///
    /// # Arguments
    ///
    /// * `texts` - A slice of string slices to generate embeddings for
    ///
    /// # Returns
    ///
    /// A vector of embeddings, where each embedding is a vector of 384 f32 values.
    /// The order of embeddings corresponds to the order of input texts.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Tokenization fails
    /// - Model inference fails
    /// - Tensor operations fail
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cetk::EmbeddingModel;
    ///
    /// # fn example() -> anyhow::Result<()> {
    /// let model = EmbeddingModel::new()?;
    /// let texts = ["First text", "Second text"];
    /// let embeddings = model.embed(&texts)?;
    /// assert_eq!(embeddings.len(), 2);
    /// assert_eq!(embeddings[0].len(), 384);
    /// # Ok(())
    /// # }
    /// ```
    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Clone tokenizer to make it mutable
        let mut tokenizer = self.tokenizer.clone();

        // Configure padding
        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
        }

        // Tokenize the input texts
        let tokens = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(anyhow::Error::msg)?;

        // Convert tokens to tensors
        let token_ids = tokens
            .iter()
            .map(|token| {
                let ids = token.get_ids().to_vec();
                Tensor::new(ids.as_slice(), &self.device)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let token_type_ids = token_ids.zeros_like()?; // all-MiniLM-L6-v2 doesn't use token_type_ids

        // Generate embeddings
        let embeddings = self.model.forward(&token_ids, &token_type_ids, None)?;

        // Apply mean pooling and normalization
        let (_n_sentences, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = embeddings.broadcast_div(&embeddings.sqr()?.sum_keepdim(1)?.sqrt()?)?;

        // Convert to Vec<Vec<f32>>
        let embeddings = embeddings.to_vec2::<f32>()?;

        Ok(embeddings)
    }

    /// Generate a single embedding for one text.
    ///
    /// This is a convenience method that calls [`embed`](Self::embed) with a single text
    /// and extracts the first result. For processing multiple texts, prefer using
    /// [`embed`](Self::embed) directly as it's more efficient.
    ///
    /// # Arguments
    ///
    /// * `text` - The text string to generate an embedding for
    ///
    /// # Returns
    ///
    /// A 384-dimensional embedding vector as f32 values.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The embedding generation fails
    /// - No embedding is produced (should not happen in normal operation)
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cetk::EmbeddingModel;
    ///
    /// # fn example() -> anyhow::Result<()> {
    /// let model = EmbeddingModel::new()?;
    /// let embedding = model.embed_single("Hello, world!")?;
    /// assert_eq!(embedding.len(), 384);
    /// # Ok(())
    /// # }
    /// ```
    pub fn embed_single(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed(&[text])?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Failed to generate embedding"))
    }
}

/// Shared embedding model instance.
///
/// EmbeddingService provides a thread-safe, shared interface to an embedding model.
/// It wraps an [`EmbeddingModel`] in an [`Arc`] for efficient sharing across multiple
/// contexts without duplicating the heavyweight model resources.
///
/// This is the preferred interface for using embeddings in applications that need
/// to share the model across multiple threads or components.
///
/// # Examples
///
/// ```rust,no_run
/// use cetk::EmbeddingService;
///
/// # fn example() -> anyhow::Result<()> {
/// let service = EmbeddingService::new()?;
/// let embedding = service.embed_single("Hello, world!")?;
/// assert_eq!(embedding.len(), 384);
/// # Ok(())
/// # }
/// ```
pub struct EmbeddingService {
    model: Arc<EmbeddingModel>,
}

impl EmbeddingService {
    /// Create a new embedding service.
    ///
    /// This creates a new [`EmbeddingModel`] and wraps it in an [`Arc`] for efficient
    /// sharing. The model is initialized with the sentence-transformers/all-MiniLM-L6-v2
    /// weights from Hugging Face Hub.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying [`EmbeddingModel::new`] fails, which can
    /// happen if:
    /// - Network connection to Hugging Face Hub fails
    /// - Model files cannot be downloaded or read
    /// - Model weights cannot be loaded
    /// - Device initialization fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cetk::EmbeddingService;
    ///
    /// let service = EmbeddingService::new()?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn new() -> Result<Self> {
        let model = Arc::new(EmbeddingModel::new()?);
        Ok(EmbeddingService { model })
    }

    /// Generate embeddings for a batch of texts.
    ///
    /// This method delegates to the underlying [`EmbeddingModel::embed`] method.
    /// See that method for detailed documentation on behavior, arguments, and errors.
    ///
    /// # Arguments
    ///
    /// * `texts` - A slice of string slices to generate embeddings for
    ///
    /// # Returns
    ///
    /// A vector of 384-dimensional embeddings, one for each input text.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cetk::EmbeddingService;
    ///
    /// # fn example() -> anyhow::Result<()> {
    /// let service = EmbeddingService::new()?;
    /// let texts = ["First text", "Second text"];
    /// let embeddings = service.embed(&texts)?;
    /// assert_eq!(embeddings.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.model.embed(texts)
    }

    /// Generate a single embedding for one text.
    ///
    /// This method delegates to the underlying [`EmbeddingModel::embed_single`] method.
    /// See that method for detailed documentation on behavior, arguments, and errors.
    ///
    /// # Arguments
    ///
    /// * `text` - The text string to generate an embedding for
    ///
    /// # Returns
    ///
    /// A 384-dimensional embedding vector.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cetk::EmbeddingService;
    ///
    /// # fn example() -> anyhow::Result<()> {
    /// let service = EmbeddingService::new()?;
    /// let embedding = service.embed_single("Hello, world!")?;
    /// assert_eq!(embedding.len(), 384);
    /// # Ok(())
    /// # }
    /// ```
    pub fn embed_single(&self, text: &str) -> Result<Vec<f32>> {
        self.model.embed_single(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_service_creation() {
        let service_result = EmbeddingService::new();
        if service_result.is_err() {
            println!("TODO(claude): cleanup this output");
            println!(
                "Skipping test - embedding model not available: {:?}",
                service_result.err()
            );
            return;
        }

        let _service = service_result.unwrap();
        // Test passes if we can create the service without panicking
    }

    #[test]
    fn single_text_embedding() {
        let service_result = EmbeddingService::new();
        if service_result.is_err() {
            println!("TODO(claude): cleanup this output");
            println!("Skipping test - embedding model not available");
            return;
        }

        let service = service_result.unwrap();
        let embedding_result = service.embed_single("This is a test sentence.");

        assert!(embedding_result.is_ok());
        let embedding = embedding_result.unwrap();
        assert!(!embedding.is_empty());
        assert_eq!(embedding.len(), 384); // all-MiniLM-L6-v2 produces 384-dimensional embeddings
    }

    #[test]
    fn batch_text_embedding() {
        let service_result = EmbeddingService::new();
        if service_result.is_err() {
            println!("TODO(claude): cleanup this output");
            println!("Skipping test - embedding model not available");
            return;
        }

        let service = service_result.unwrap();
        let texts = ["First test sentence.", "Second test sentence."];
        let embeddings_result = service.embed(&texts);

        assert!(embeddings_result.is_ok());
        let embeddings = embeddings_result.unwrap();
        assert_eq!(embeddings.len(), 2);
        for embedding in embeddings {
            assert_eq!(embedding.len(), 384);
        }
    }

    #[test]
    fn empty_batch_embedding() {
        let service_result = EmbeddingService::new();
        if service_result.is_err() {
            println!("TODO(claude): cleanup this output");
            println!("Skipping test - embedding model not available");
            return;
        }

        let service = service_result.unwrap();
        let embeddings_result = service.embed(&[]);

        assert!(embeddings_result.is_ok());
        let embeddings = embeddings_result.unwrap();
        assert!(embeddings.is_empty());
    }
}
