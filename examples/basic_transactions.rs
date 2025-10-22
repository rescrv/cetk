//! Basic transaction storage example
//!
//! This example shows how to:
//! - Create and persist transactions to Chroma
//! - Load agent data from storage
//! - Build new transactions using the fluent API

use cetk::{AgentData, AgentID, ContextManager, MountID};
use chroma::ChromaHttpClient;
use claudius::{MessageParam, MessageRole};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to Chroma
    let client = ChromaHttpClient::cloud()?;
    let collection = client
        .get_or_create_collection("basic_example", None, None)
        .await?;
    let context_manager = ContextManager::new(collection)?;

    let agent_id = AgentID::generate().unwrap();
    let mount_id = MountID::generate().unwrap();

    // Start with empty agent data
    let mut agent_data = AgentData {
        agent_id,
        contexts: Vec::new(),
    };

    // Create first transaction using fluent API
    let nonce1 = agent_data
        .next_transaction(&context_manager)
        .message(MessageParam {
            role: MessageRole::User,
            content: "Create a hello world file".into(),
        })
        .message(MessageParam {
            role: MessageRole::Assistant,
            content: "I'll create that file for you".into(),
        })
        .write_file(mount_id, "/hello.txt", "Hello, world!")?
        .save()
        .await?;

    // Verify transaction was persisted
    assert!(!nonce1.is_empty());
    assert_eq!(agent_data.contexts.len(), 1);
    assert_eq!(agent_data.contexts[0].transactions.len(), 1);

    // Add another transaction to the same context
    let nonce2 = agent_data
        .next_transaction(&context_manager)
        .message(MessageParam {
            role: MessageRole::User,
            content: "Now create a config file".into(),
        })
        .write_file(mount_id, "/config.json", r#"{"app": "cetk-example"}"#)?
        .save()
        .await?;

    assert!(!nonce2.is_empty());
    assert_eq!(agent_data.contexts[0].transactions.len(), 2);

    // Start a new context
    let nonce3 = agent_data
        .new_context(&context_manager)
        .message(MessageParam {
            role: MessageRole::User,
            content: "Let's start fresh with a new context".into(),
        })
        .write_file(
            mount_id,
            "/readme.md",
            "# My Project\n\nThis is a new context.",
        )?
        .save()
        .await?;

    assert!(!nonce3.is_empty());
    assert_eq!(agent_data.contexts.len(), 2);
    assert_eq!(agent_data.contexts[1].transactions.len(), 1);

    // Load agent data from storage (simulating app restart)
    let loaded_data = context_manager.load_agent(agent_id).await?;

    // Verify loaded data matches what we stored
    assert_eq!(loaded_data.agent_id, agent_id);
    assert_eq!(loaded_data.contexts.len(), 2);
    assert_eq!(loaded_data.contexts[0].transactions.len(), 2);
    assert_eq!(loaded_data.contexts[1].transactions.len(), 1);

    // Verify file content is available
    let hello_content = loaded_data.get_file_content(mount_id, "/hello.txt")?;
    assert_eq!(hello_content, Some("Hello, world!".to_string()));

    let config_content = loaded_data.get_file_content(mount_id, "/config.json")?;
    assert_eq!(
        config_content,
        Some(r#"{"app": "cetk-example"}"#.to_string())
    );

    // List all files
    let files = loaded_data.list_files(mount_id);
    assert_eq!(files.len(), 3);
    assert!(files.contains(&"/hello.txt".to_string()));
    assert!(files.contains(&"/config.json".to_string()));
    assert!(files.contains(&"/readme.md".to_string()));

    println!("✓ Basic transaction storage example completed successfully");
    Ok(())
}
