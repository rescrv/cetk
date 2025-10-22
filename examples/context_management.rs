//! Context management example
//!
//! This example demonstrates:
//! - Managing multiple contexts for an agent
//! - File operations across contexts
//! - String replacement and content searching

use cetk::{AgentID, ContextManager, MountID};
use chroma::ChromaHttpClient;
use claudius::{MessageParam, MessageRole};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup Chroma connection
    let client = ChromaHttpClient::cloud()?;
    let collection = client
        .get_or_create_collection("context_example", None, None)
        .await?;
    let context_manager = ContextManager::new(collection)?;

    let agent_id = AgentID::generate().unwrap();
    let mount_id = MountID::generate().unwrap();

    // Load existing agent data (empty initially)
    let mut agent_data = context_manager.load_agent(agent_id).await?;
    assert!(agent_data.contexts.is_empty());

    // Context 1: Initial setup
    agent_data
        .next_transaction(&context_manager)
        .message(MessageParam {
            role: MessageRole::User,
            content: "Set up initial project files".into(),
        })
        .write_file(
            mount_id,
            "/main.rs",
            "fn main() {\n    println!(\"Hello\");\n}",
        )?
        .write_file(
            mount_id,
            "/Cargo.toml",
            "[package]\nname = \"example\"\nversion = \"0.1.0\"",
        )?
        .save()
        .await?;

    // Context 1: Make some changes
    agent_data
        .next_transaction(&context_manager)
        .str_replace_file(mount_id, "/main.rs", "Hello", "Hello, world")?
        .save()
        .await?;

    // Verify file content was updated
    let main_content = agent_data.get_file_content(mount_id, "/main.rs")?;
    assert_eq!(
        main_content,
        Some("fn main() {\n    println!(\"Hello, world\");\n}".to_string())
    );

    // Context 2: New feature development
    agent_data
        .new_context(&context_manager)
        .message(MessageParam {
            role: MessageRole::User,
            content: "Add a new feature with tests".into(),
        })
        .write_file(
            mount_id,
            "/lib.rs",
            "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}",
        )?
        .write_file(
            mount_id,
            "/tests.rs",
            "#[test]\nfn test_add() {\n    assert_eq!(add(2, 3), 5);\n}",
        )?
        .save()
        .await?;

    // Update Cargo.toml to include the library
    agent_data
        .next_transaction(&context_manager)
        .str_replace_file(
            mount_id,
            "/Cargo.toml",
            "version = \"0.1.0\"",
            "version = \"0.1.0\"\n\n[lib]\nname = \"example\"",
        )?
        .save()
        .await?;

    // Verify we have multiple contexts
    assert_eq!(agent_data.contexts.len(), 2);
    assert_eq!(agent_data.contexts[0].context_seq_no, 1);
    assert_eq!(agent_data.contexts[0].transactions.len(), 2);
    assert_eq!(agent_data.contexts[1].context_seq_no, 2);
    assert_eq!(agent_data.contexts[1].transactions.len(), 2);

    // Test file listing and searching
    let files = agent_data.list_files(mount_id);
    assert_eq!(files.len(), 4);
    assert!(files.contains(&"/main.rs".to_string()));
    assert!(files.contains(&"/Cargo.toml".to_string()));
    assert!(files.contains(&"/lib.rs".to_string()));
    assert!(files.contains(&"/tests.rs".to_string()));

    // Search for function definitions
    let matches = agent_data.search_file_contents(mount_id, "fn ")?;
    assert_eq!(matches.len(), 3); // main.rs, lib.rs, and tests.rs all have functions

    // Test transaction builder file operations
    let builder = agent_data.next_transaction(&context_manager);

    // View current files
    let main_rs = builder.view_file(mount_id, "/main.rs");
    assert!(main_rs.is_some());
    assert!(main_rs.unwrap().contains("Hello, world"));

    let lib_rs = builder.view_file(mount_id, "/lib.rs");
    assert!(lib_rs.is_some());
    assert!(lib_rs.unwrap().contains("pub fn add"));

    // Test search in builder
    let builder_matches = builder.search_files(mount_id, "assert");
    assert_eq!(builder_matches.len(), 1); // Only tests.rs should match
    assert_eq!(builder_matches[0].0, "/tests.rs");

    // Add a documentation update
    let nonce = builder
        .write_file(
            mount_id,
            "/README.md",
            "# Example Project\n\nA simple Rust project with tests.",
        )?
        .save()
        .await?;

    assert!(!nonce.is_empty());

    // Reload from storage to verify persistence
    let reloaded = context_manager.load_agent(agent_id).await?;
    assert_eq!(reloaded.contexts.len(), 2);

    let readme_content = reloaded.get_file_content(mount_id, "/README.md")?;
    assert_eq!(
        readme_content,
        Some("# Example Project\n\nA simple Rust project with tests.".to_string())
    );

    // Verify helper methods
    let latest_context = reloaded.latest_context();
    assert!(latest_context.is_some());
    assert_eq!(latest_context.unwrap().context_seq_no, 2);

    let specific_context = reloaded.get_context(1);
    assert!(specific_context.is_some());
    assert_eq!(specific_context.unwrap().transactions.len(), 2);

    let all_transactions = reloaded.all_transactions();
    assert_eq!(all_transactions.len(), 5); // 2 in context 1, 3 in context 2

    println!("✓ Context management example completed successfully");
    Ok(())
}
