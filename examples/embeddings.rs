//! Embeddings example
//!
//! This example demonstrates:
//! - Creating an embedding service
//! - Generating embeddings for single and multiple texts
//! - Basic similarity comparison

use cetk::EmbeddingService;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create embedding service
    let service = match EmbeddingService::new() {
        Ok(s) => s,
        Err(_) => {
            println!("Embedding model not available - skipping example");
            return Ok(());
        }
    };

    // Test single embedding
    let text = "The Context Engineer's Toolkit provides transaction management";
    let embedding = service.embed_single(text)?;
    assert_eq!(embedding.len(), 384); // all-MiniLM-L6-v2 produces 384-dimensional embeddings

    // Test batch embeddings
    let texts = [
        "Transaction storage and retrieval",
        "Agent context management",
        "File system operations",
        "Semantic search capabilities",
    ];

    let text_refs: Vec<&str> = texts.to_vec();
    let embeddings = service.embed(&text_refs)?;

    assert_eq!(embeddings.len(), 4);
    for embedding in &embeddings {
        assert_eq!(embedding.len(), 384);
    }

    // Calculate similarity between first two embeddings using cosine similarity
    let similarity = cosine_similarity(&embeddings[0], &embeddings[1]);
    assert!((0.0..=1.0).contains(&similarity));

    // Find most similar text to query
    let query = "Managing agent transactions";
    let query_embedding = service.embed_single(query)?;

    let mut best_similarity = -1.0;
    let mut best_index = 0;

    for (i, embedding) in embeddings.iter().enumerate() {
        let sim = cosine_similarity(&query_embedding, embedding);
        if sim > best_similarity {
            best_similarity = sim;
            best_index = i;
        }
    }

    // The query should be most similar to "Transaction storage and retrieval"
    assert_eq!(best_index, 0);
    assert!(best_similarity > 0.5); // Should have reasonable similarity

    // Test empty batch
    let empty_embeddings = service.embed(&[])?;
    assert!(empty_embeddings.is_empty());

    println!("✓ Embeddings example completed successfully");
    println!("  - Single embedding: {} dimensions", embedding.len());
    println!("  - Batch embeddings: {} texts processed", embeddings.len());
    println!(
        "  - Best match for '{}': '{}' (similarity: {:.3})",
        query, texts[best_index], best_similarity
    );

    Ok(())
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}
