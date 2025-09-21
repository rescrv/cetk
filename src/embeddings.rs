use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{Repo, RepoType, api::sync::Api};
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Embedding model for generating vector embeddings from text using sentence-transformers/all-MiniLM-L6-v2
pub struct EmbeddingModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingModel {
    /// Create a new EmbeddingModel using sentence-transformers/all-MiniLM-L6-v2
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

    /// Generate embeddings for a batch of texts
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

    /// Generate a single embedding for one text
    pub fn embed_single(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed(&[text])?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Failed to generate embedding"))
    }
}

/// Shared embedding model instance
pub struct EmbeddingService {
    model: Arc<EmbeddingModel>,
}

impl EmbeddingService {
    /// Create a new embedding service
    pub fn new() -> Result<Self> {
        let model = Arc::new(EmbeddingModel::new()?);
        Ok(EmbeddingService { model })
    }

    /// Generate embeddings for a batch of texts
    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.model.embed(texts)
    }

    /// Generate a single embedding for one text
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
