# CETK Chroma Format Implementation Specification

## Overview
This document specifies the implementation plan for the CETK (Context Engineer's Toolkit) Chroma storage format as described in the README. The format defines how agent contexts, transactions, and virtual filesystems are persisted in Chroma collections.

## Phase 1: Core Data Structures & Types

### Context ID Type
[ ] Add `ContextID` type using `generate_id!` macro with prefix `"context:"`
[ ] Implement serde support with `generate_id_serde!` macro
[ ] Add conversion methods (to_string, from_human_readable)

### Transaction Structure Enhancements
[ ] Add chunking support for documents exceeding 16KB limit
[ ] Implement `chunk_transaction()` method to split large transactions
[ ] Add `generate_embedding_summary()` method for transaction summaries
[ ] Define `TransactionChunk` struct with chunk sequencing
[ ] Add validation for chunk size limits in `check_invariants()`

### Context Seal Structure
[ ] Define `ContextSeal` struct with fields:
    [ ] `context_id`: Current context UUID
    [ ] `next_context_id`: Next context in lineage
    [ ] `sealed_at`: Timestamp
    [ ] `summary`: Context summary for embedding
[ ] Add serde derive macros for serialization
[ ] Implement methods for seal creation and validation

### Markdown List Structure
[ ] Define `MarkdownList` struct with:
    [ ] `mount_id`: Virtual filesystem mount UUID
    [ ] `path`: File path in virtual filesystem
    [ ] `sections`: HashMap of section headers to bullet points
[ ] Implement `add_bullet()` method
[ ] Implement `remove_bullet()` method
[ ] Implement `update_bullet()` method
[ ] Add section management (create/delete sections)
[ ] Add parsing from markdown text
[ ] Add serialization to markdown format

## Phase 2: Chroma Integration

### ContextManager Constructor Update
[ ] Replace `chroma: ()` parameter with `chroma: ChromaClient`
[ ] Add `collection_name` parameter for Chroma collection
[ ] Initialize ChromaDB connection in `new()`
[ ] Create or get collection on initialization
[ ] Add connection validation and error handling

### Storage Methods in ContextManager
[ ] Implement `store_transaction()` method:
    [ ] Accept Transaction and chunk if needed
    [ ] Generate embeddings from summary
    [ ] Store with proper metadata structure
    [ ] Return storage confirmation
[ ] Implement `store_seal()` method:
    [ ] Store context seal document
    [ ] Add searchable embedding
    [ ] Update lineage tracking
[ ] Implement `store_markdown_list()` method:
    [ ] Parse markdown into bullet points
    [ ] Store each bullet as separate document
    [ ] Maintain section organization

### Retrieval Methods
[ ] Implement `load_context()` method:
    [ ] Query all transactions for context UUID
    [ ] Reconstruct from chunks
    [ ] Validate transaction sequence
    [ ] Return complete Context
[ ] Implement `get_latest_context()` method:
    [ ] Find most recent seal for agent
    [ ] Follow seal chain to latest
    [ ] Load associated context
[ ] Implement `search_contexts()` method:
    [ ] Accept query embedding or text
    [ ] Search across context summaries
    [ ] Return ranked context IDs

## Phase 3: Transaction Storage Implementation

### Transaction Storage Logic
[ ] Implement chunking algorithm:
    [ ] Split transactions > 16KB into chunks
    [ ] Maintain chunk ordering with sequence numbers
    [ ] Preserve transaction integrity across chunks
[ ] Generate metadata structure:
    [ ] `agent`: Agent UUID
    [ ] `context`: Context UUID
    [ ] `transaction`: Transaction sequence number
    [ ] `chunk`: Chunk sequence number
[ ] Create document keys:
    [ ] Format: `context=UUID;transaction=seq_no;chunk=seq_no`
    [ ] Ensure uniqueness and ordering
[ ] Store embeddings:
    [ ] Generate from transaction summary
    [ ] Associate with first chunk
    [ ] Link chunks via metadata

### Transaction Validation
[ ] Verify transaction sequence ordering
[ ] Check chunk size doesn't exceed 16KB
[ ] Validate all required metadata fields present
[ ] Ensure agent_id consistency
[ ] Verify context_seq_no matches
[ ] Check transaction_seq_no continuity

## Phase 4: Context Lifecycle

### Context Opening Implementation
[ ] Query for latest seal by agent_id
[ ] Load all transactions for context
[ ] Reconstruct transaction history:
    [ ] Order by transaction sequence
    [ ] Merge chunks into complete transactions
    [ ] Validate continuity
[ ] Handle missing context (create new)
[ ] Initialize Context struct with loaded data
[ ] Set up transaction counter from history

### Context Sealing
[ ] Create seal document with:
    [ ] Current context UUID
    [ ] Next context UUID (when forking/compacting)
    [ ] Timestamp
    [ ] Summary for search
[ ] Store with key: `context=UUID;seal`
[ ] Generate and store search embedding
[ ] Update agent's latest context pointer
[ ] Validate seal uniqueness

### Context Forking Support
[ ] Allow multiple child contexts from parent
[ ] Track fork relationships in seal metadata
[ ] Implement `fork_context()` method:
    [ ] Create new context UUID
    [ ] Copy current state
    [ ] Create fork seal
    [ ] Maintain parent reference
[ ] Add lineage traversal methods
[ ] Support merge detection (future)

## Phase 5: Virtual Filesystem Support

### Chunked File Storage (Future)
[ ] Design chunking strategy for large files
[ ] Define metadata schema for file chunks
[ ] Plan reconstruction algorithm
[ ] Document in separate spec when implementing

### Markdown List Storage Implementation
[ ] Parse markdown files into structure:
    [ ] Extract section headers
    [ ] Parse bullet points per section
    [ ] Handle nested bullets
[ ] Store each bullet as document:
    [ ] One document per bullet point
    [ ] Maintain section association
    [ ] Preserve ordering
[ ] Use metadata schema:
    [ ] `mount`: Mount UUID
    [ ] `path`: File path
    [ ] `section`: Section header
[ ] Create keys:
    [ ] Format: `mount=UUID;path=/path/to/file.md;section=header/bullet`
    [ ] Ensure uniqueness
[ ] Implement retrieval:
    [ ] Query by mount and path
    [ ] Reconstruct sections
    [ ] Preserve bullet ordering

## Phase 6: Testing & Error Handling

### Error Type Definitions
[ ] Define Error enum variants:
    [ ] `ChromaConnectionError`: Connection failures
    [ ] `TransactionError`: Transaction validation failures
    [ ] `ContextNotFound`: Missing context
    [ ] `ChunkSizeExceeded`: Document too large
    [ ] `InvalidSequence`: Sequence number issues
    [ ] `SealError`: Seal creation/validation failures
[ ] Add error context and debugging info
[ ] Implement Display trait for errors
[ ] Add error conversion from ChromaDB errors

### Unit Tests
[ ] Test transaction chunking:
    [ ] Small transactions (no chunking)
    [ ] Large transactions (multiple chunks)
    [ ] Exact 16KB boundary cases
    [ ] Chunk reconstruction
[ ] Test context operations:
    [ ] Context creation
    [ ] Context loading
    [ ] Context sealing
    [ ] Missing context handling
[ ] Test markdown lists:
    [ ] Bullet addition/removal
    [ ] Section management
    [ ] Markdown parsing
    [ ] Storage and retrieval
[ ] Test seal operations:
    [ ] Seal creation
    [ ] Lineage tracking
    [ ] Fork handling

### Integration Tests
[ ] Full transaction lifecycle:
    [ ] Create context
    [ ] Add transactions
    [ ] Seal context
    [ ] Load sealed context
[ ] Compaction scenarios:
    [ ] Create long history
    [ ] Compact to new context
    [ ] Verify data preservation
[ ] Concurrent access:
    [ ] Multiple readers
    [ ] Single writer
    [ ] Lock contention handling
[ ] Error recovery:
    [ ] Partial write failures
    [ ] Connection interruptions
    [ ] Invalid data handling

## Implementation Priority Order

### Phase 1: Foundation (Required First)
[ ] ContextID type
[ ] Enhanced Transaction structure
[ ] Basic Error types

### Phase 2: Core Storage (Required Second)
[ ] Update ContextManager constructor with ChromaDB
[ ] Implement basic `transact()` method
[ ] Implement basic `open()` method

### Phase 3: Context Management
[ ] Implement context sealing
[ ] Add seal storage and retrieval
[ ] Implement latest context discovery

### Phase 4: Advanced Features
[ ] Add transaction chunking
[ ] Implement markdown list support
[ ] Add context forking

### Phase 5: Testing & Hardening
[ ] Unit tests for each component
[ ] Integration tests for workflows
[ ] Error handling improvements

## Key Implementation Constraints

### Chroma Document Structure
[ ] All documents scoped with `agent=UUID` metadata
[ ] Document size limit: 16KB per chunk
[ ] Keys must follow specified format for each type
[ ] Embeddings required for searchability

### Transaction Invariants
[ ] Transactions append-only within context
[ ] Sequence numbers must be continuous
[ ] Agent ID consistent across transaction
[ ] Context sequence number immutable

### Concurrency Model
[ ] Pessimistic locking for transactions (existing)
[ ] Single writer per context
[ ] Multiple readers allowed
[ ] Atomic context sealing

### Storage Guarantees
[ ] Every context permanently stored
[ ] No data loss during compaction
[ ] Complete audit trail maintained
[ ] Backward navigation via seals

## Future Enhancements

### Planned Features
[ ] Chunked file support for large files
[ ] Context merging capabilities
[ ] Advanced search with metadata filters
[ ] Batch transaction operations
[ ] Context snapshots/checkpoints

### Performance Optimizations
[ ] Embedding caching
[ ] Batch chunk retrieval
[ ] Lazy transaction loading
[ ] Index optimization strategies
[ ] Connection pooling

## Success Criteria

### Functional Requirements Met
[ ] Contexts persist across restarts
[ ] Transactions stored durably
[ ] Markdown lists fully functional
[ ] Context sealing works correctly
[ ] Agent state recoverable

### Performance Targets
[ ] Transaction storage < 100ms
[ ] Context loading < 500ms
[ ] Search results < 200ms
[ ] Chunking overhead < 10%

### Quality Standards
[ ] All tests passing
[ ] No data corruption
[ ] Graceful error handling
[ ] Complete documentation
[ ] Clean API design