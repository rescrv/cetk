//! Comprehensive agent workflow example
//!
//! This example demonstrates a complete workflow:
//! - Agent initialization and persistence
//! - Multi-turn conversations with file operations
//! - Context switching and history management
//! - Error handling and recovery

use cetk::{AgentID, ContextManager, MountID};
use chroma::ChromaHttpClient;
use claudius::{MessageParam, MessageRole};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup Chroma connection
    let client = ChromaHttpClient::cloud()?;
    let collection = client
        .get_or_create_collection("workflow_example", None, None)
        .await?;
    let context_manager = ContextManager::new(collection)?;

    let agent_id = AgentID::generate().unwrap();
    let code_mount = MountID::generate().unwrap();
    let docs_mount = MountID::generate().unwrap();

    // === Phase 1: Initial Project Setup ===

    let mut agent = context_manager.load_agent(agent_id).await?;

    // User requests project setup
    agent
        .next_transaction(&context_manager)
        .message(MessageParam {
            role: MessageRole::User,
            content: "I want to create a new Rust web service project called 'api-server'".into(),
        })
        .message(MessageParam {
            role: MessageRole::Assistant,
            content: "I'll create the initial project structure for your Rust web service.".into(),
        })
        .write_file(code_mount, "/Cargo.toml",
            "[package]\nname = \"api-server\"\nversion = \"0.1.0\"\n\n[dependencies]\ntokio = { version = \"1\", features = [\"full\"] }\naxum = \"0.7\"")?
        .write_file(code_mount, "/src/main.rs",
            "use axum::{routing::get, Router};\n\n#[tokio::main]\nasync fn main() {\n    let app = Router::new()\n        .route(\"/\", get(hello));\n    \n    let listener = tokio::net::TcpListener::bind(\"0.0.0.0:3000\").await.unwrap();\n    axum::serve(listener, app).await.unwrap();\n}\n\nasync fn hello() -> &'static str {\n    \"Hello, World!\"\n}")?
        .write_file(docs_mount, "/README.md",
            "# API Server\n\nA simple Rust web service using Axum.\n\n## Usage\n\n```bash\ncargo run\n```")?
        .save()
        .await?;

    assert_eq!(agent.contexts.len(), 1);
    assert!(agent.get_file_content(code_mount, "/Cargo.toml")?.is_some());

    // === Phase 2: Add Features ===

    // User requests database integration
    agent
        .next_transaction(&context_manager)
        .message(MessageParam {
            role: MessageRole::User,
            content: "Add database support with SQLx and PostgreSQL".into(),
        })
        .message(MessageParam {
            role: MessageRole::Assistant,
            content: "I'll add SQLx and database connection handling.".into(),
        })
        .str_replace_file(code_mount, "/Cargo.toml",
            "axum = \"0.7\"",
            "axum = \"0.7\"\nsqlx = { version = \"0.7\", features = [\"runtime-tokio-rustls\", \"postgres\"] }\nserde = { version = \"1.0\", features = [\"derive\"] }")?
        .write_file(code_mount, "/src/database.rs",
            "use sqlx::{Pool, Postgres};\n\npub type DbPool = Pool<Postgres>;\n\npub async fn create_pool(database_url: &str) -> Result<DbPool, sqlx::Error> {\n    sqlx::postgres::PgPoolOptions::new()\n        .max_connections(20)\n        .connect(database_url)\n        .await\n}")?
        .str_replace_file(code_mount, "/src/main.rs",
            "use axum::{routing::get, Router};",
            "mod database;\n\nuse axum::{routing::get, Router, extract::State};\nuse database::DbPool;")?
        .save()
        .await?;

    // User requests API endpoints
    agent
        .next_transaction(&context_manager)
        .message(MessageParam {
            role: MessageRole::User,
            content: "Add REST endpoints for user management".into(),
        })
        .message(MessageParam {
            role: MessageRole::Assistant,
            content: "I'll create user endpoints with proper routing and handlers.".into(),
        })
        .write_file(code_mount, "/src/handlers/mod.rs",
            "pub mod users;\n\npub use users::*;")?
        .write_file(code_mount, "/src/handlers/users.rs",
            "use axum::{Json, extract::{State, Path}};\nuse serde::{Deserialize, Serialize};\nuse crate::database::DbPool;\n\n#[derive(Serialize, Deserialize)]\npub struct User {\n    pub id: i32,\n    pub name: String,\n    pub email: String,\n}\n\npub async fn get_user(State(_pool): State<DbPool>, Path(id): Path<i32>) -> Json<User> {\n    // TODO: Implement database query\n    Json(User {\n        id,\n        name: \"Test User\".to_string(),\n        email: \"test@example.com\".to_string(),\n    })\n}\n\npub async fn create_user(State(_pool): State<DbPool>, Json(user): Json<User>) -> Json<User> {\n    // TODO: Implement database insertion\n    Json(user)\n}")?
        .str_replace_file(code_mount, "/src/main.rs",
            "mod database;",
            "mod database;\nmod handlers;")?
        .save()
        .await?;

    // Verify we have the expected files
    let code_files = agent.list_files(code_mount);
    assert!(code_files.contains(&"/Cargo.toml".to_string()));
    assert!(code_files.contains(&"/src/main.rs".to_string()));
    assert!(code_files.contains(&"/src/database.rs".to_string()));
    assert!(code_files.contains(&"/src/handlers/users.rs".to_string()));

    let docs_files = agent.list_files(docs_mount);
    assert!(docs_files.contains(&"/README.md".to_string()));

    // === Phase 3: New Context for Refactoring ===

    // Simulate context switch (major refactoring session)
    agent
        .new_context(&context_manager)
        .message(MessageParam {
            role: MessageRole::User,
            content: "Let's refactor the code structure and add proper error handling".into(),
        })
        .message(MessageParam {
            role: MessageRole::Assistant,
            content: "I'll reorganize the code with better error handling and structure.".into(),
        })
        .write_file(code_mount, "/src/error.rs",
            "use axum::response::{Response, IntoResponse};\nuse axum::http::StatusCode;\n\n#[derive(Debug)]\npub enum ApiError {\n    Database(sqlx::Error),\n    NotFound,\n    BadRequest(String),\n}\n\nimpl IntoResponse for ApiError {\n    fn into_response(self) -> Response {\n        let (status, message) = match self {\n            ApiError::Database(_) => (StatusCode::INTERNAL_SERVER_ERROR, \"Database error\"),\n            ApiError::NotFound => (StatusCode::NOT_FOUND, \"Resource not found\"),\n            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.as_str()),\n        };\n        (status, message).into_response()\n    }\n}")?
        .str_replace_file(code_mount, "/src/handlers/users.rs",
            "use axum::{Json, extract::{State, Path}};",
            "use axum::{Json, extract::{State, Path}};\nuse crate::error::ApiError;")?
        .str_replace_file(code_mount, "/src/main.rs",
            "mod database;\nmod handlers;",
            "mod database;\nmod handlers;\nmod error;")?
        .save()
        .await?;

    // Update documentation
    agent
        .next_transaction(&context_manager)
        .str_replace_file(docs_mount, "/README.md",
            "A simple Rust web service using Axum.",
            "A robust Rust web service using Axum with PostgreSQL support.")?
        .str_replace_file(docs_mount, "/README.md",
            "```bash\ncargo run\n```",
            "```bash\n# Set database URL\nexport DATABASE_URL=postgres://user:pass@localhost/dbname\n\n# Run the server\ncargo run\n```\n\n## API Endpoints\n\n- `GET /users/{id}` - Get user by ID\n- `POST /users` - Create new user")?
        .save()
        .await?;

    // === Verification ===

    // Check we have two contexts
    assert_eq!(agent.contexts.len(), 2);
    assert_eq!(agent.contexts[0].context_seq_no, 1);
    assert_eq!(agent.contexts[0].transactions.len(), 3);
    assert_eq!(agent.contexts[1].context_seq_no, 2);
    assert_eq!(agent.contexts[1].transactions.len(), 2);

    // Search for error handling code
    let error_matches = agent.search_file_contents(code_mount, "ApiError")?;
    assert!(!error_matches.is_empty());

    // Verify latest context
    let latest = agent.latest_context().unwrap();
    assert_eq!(latest.context_seq_no, 2);

    // Test transaction builder capabilities
    let builder = agent.next_transaction(&context_manager);

    // Check current state of main.rs
    let main_content = builder.view_file(code_mount, "/src/main.rs").unwrap();
    assert!(main_content.contains("mod error"));

    // Search for TODO comments
    let todos = builder.search_files(code_mount, "TODO");
    assert!(!todos.is_empty());

    // === Persistence Test ===

    // Reload agent to verify everything was persisted
    let reloaded_agent = context_manager.load_agent(agent_id).await?;
    assert_eq!(reloaded_agent.contexts.len(), 2);
    assert_eq!(reloaded_agent.all_transactions().len(), 5);

    // Verify file content survived reload
    let main_after_reload = reloaded_agent.get_file_content(code_mount, "/src/main.rs")?;
    assert!(main_after_reload.unwrap().contains("mod error"));

    let readme_after_reload = reloaded_agent.get_file_content(docs_mount, "/README.md")?;
    assert!(readme_after_reload.unwrap().contains("PostgreSQL support"));

    println!("✓ Complete agent workflow example completed successfully");
    println!("  - Contexts: {}", reloaded_agent.contexts.len());
    println!(
        "  - Total transactions: {}",
        reloaded_agent.all_transactions().len()
    );
    println!(
        "  - Code files: {}",
        reloaded_agent.list_files(code_mount).len()
    );
    println!(
        "  - Documentation files: {}",
        reloaded_agent.list_files(docs_mount).len()
    );

    Ok(())
}
