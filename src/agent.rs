use crate::AgentID;
use sst::log::WriteBatch;
use sst::{Cursor, Error as SstError, KeyValueRef};

/////////////////////////////////////////////// Errors //////////////////////////////////////////////

/// Error that occurs during Agent batch operations.
#[derive(Debug)]
pub enum AgentBatchError {
    /// Failed to serialize agent data to JSON.
    SerializationFailed(serde_json::Error),
    /// Underlying SST error during batch operation.
    SstError(SstError),
}

impl std::fmt::Display for AgentBatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SerializationFailed(err) => write!(f, "Agent serialization failed: {}", err),
            Self::SstError(err) => write!(f, "SST batch operation failed: {}", err),
        }
    }
}

impl std::error::Error for AgentBatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SerializationFailed(err) => Some(err),
            Self::SstError(_err) => None,
        }
    }
}

impl From<serde_json::Error> for AgentBatchError {
    fn from(err: serde_json::Error) -> Self {
        Self::SerializationFailed(err)
    }
}

impl From<SstError> for AgentBatchError {
    fn from(err: SstError) -> Self {
        Self::SstError(err)
    }
}

/// Error that occurs during Agent cursor operations.
#[derive(Debug)]
pub enum AgentCursorError {
    /// Failed to deserialize agent data from JSON.
    DeserializationFailed(serde_json::Error),
    /// Failed to parse agent key from bytes.
    KeyParsingFailed(&'static str),
    /// Underlying SST error during cursor operation.
    SstError(SstError),
}

impl std::fmt::Display for AgentCursorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeserializationFailed(err) => write!(f, "Agent deserialization failed: {}", err),
            Self::KeyParsingFailed(msg) => write!(f, "Agent key parsing failed: {}", msg),
            Self::SstError(err) => write!(f, "SST cursor operation failed: {}", err),
        }
    }
}

impl std::error::Error for AgentCursorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DeserializationFailed(err) => Some(err),
            Self::KeyParsingFailed(_) => None,
            Self::SstError(_err) => None,
        }
    }
}

impl From<serde_json::Error> for AgentCursorError {
    fn from(err: serde_json::Error) -> Self {
        Self::DeserializationFailed(err)
    }
}

impl From<SstError> for AgentCursorError {
    fn from(err: SstError) -> Self {
        Self::SstError(err)
    }
}

///////////////////////////////////////////// AgentKeys ////////////////////////////////////////////

/// TableSetID for Agent table - corresponds to 1_A in EXAMPLE.md
const AGENT_TABLE_ID: u32 = 1;

/// Field IDs for Agent fields
const AGENT_CREATED_AT_FIELD: u32 = 2; // corresponds to 2_A in EXAMPLE.md
const AGENT_UPDATED_AT_FIELD: u32 = 3; // corresponds to 3_A in EXAMPLE.md

/// Key for Agent's created_at field: (TableSetID, 1_A, 1_B, agent_id, 2_A)
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct AgentCreatedAtKey {
    pub agent_id: AgentID,
}

/// Key for Agent's updated_at field: (TableSetID, 1_A, 1_B, agent_id, 3_A)
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct AgentUpdatedAtKey {
    pub agent_id: AgentID,
}

impl AgentCreatedAtKey {
    pub fn new(agent_id: AgentID) -> Self {
        Self { agent_id }
    }

    /// Pack key as binary tuple: (TableSetID, 1_A, 1_B, agent_id, 2_A)
    pub fn pack(&self) -> Vec<u8> {
        let mut key = Vec::new();
        // TODO(claude): Use proper TableSetID from configuration
        key.extend_from_slice(&1u32.to_be_bytes()); // TableSetID
        key.extend_from_slice(&AGENT_TABLE_ID.to_be_bytes()); // 1_A
        key.extend_from_slice(&1u32.to_be_bytes()); // 1_B (agent_id field ID)
        key.extend_from_slice(&self.agent_id.id); // agent_id
        key.extend_from_slice(&AGENT_CREATED_AT_FIELD.to_be_bytes()); // 2_A
        key
    }

    /// Unpack key from binary tuple format
    pub fn unpack(key: &[u8]) -> Result<Self, &'static str> {
        if key.len() < 20 {
            // 4 + 4 + 4 + 16 (min AgentID) + 4
            return Err("key too short");
        }

        let mut offset = 0;

        // Skip TableSetID (4 bytes)
        offset += 4;

        // Check 1_A (AGENT_TABLE_ID)
        let table_id = u32::from_be_bytes([
            key[offset],
            key[offset + 1],
            key[offset + 2],
            key[offset + 3],
        ]);
        if table_id != AGENT_TABLE_ID {
            return Err("invalid table ID");
        }
        offset += 4;

        // Skip 1_B (4 bytes)
        offset += 4;

        // Extract agent_id (16 bytes for UUID)
        if key.len() < offset + 16 + 4 {
            return Err("key too short for agent_id");
        }

        let agent_id_bytes = &key[offset..offset + 16];
        let agent_id_array: [u8; 16] = agent_id_bytes
            .try_into()
            .map_err(|_| "invalid agent_id length")?;
        let agent_id = AgentID::new(agent_id_array);
        offset += 16;

        // Check field ID (AGENT_CREATED_AT_FIELD)
        let field_id = u32::from_be_bytes([
            key[offset],
            key[offset + 1],
            key[offset + 2],
            key[offset + 3],
        ]);
        if field_id != AGENT_CREATED_AT_FIELD {
            return Err("invalid field ID");
        }

        Ok(Self { agent_id })
    }
}

impl AgentUpdatedAtKey {
    pub fn new(agent_id: AgentID) -> Self {
        Self { agent_id }
    }

    /// Pack key as binary tuple: (TableSetID, 1_A, 1_B, agent_id, 3_A)
    pub fn pack(&self) -> Vec<u8> {
        let mut key = Vec::new();
        // TODO(claude): Use proper TableSetID from configuration
        key.extend_from_slice(&1u32.to_be_bytes()); // TableSetID
        key.extend_from_slice(&AGENT_TABLE_ID.to_be_bytes()); // 1_A
        key.extend_from_slice(&1u32.to_be_bytes()); // 1_B (agent_id field ID)
        key.extend_from_slice(&self.agent_id.id); // agent_id
        key.extend_from_slice(&AGENT_UPDATED_AT_FIELD.to_be_bytes()); // 3_A
        key
    }

    /// Unpack key from binary tuple format
    pub fn unpack(key: &[u8]) -> Result<Self, &'static str> {
        if key.len() < 20 {
            // 4 + 4 + 4 + 16 (min AgentID) + 4
            return Err("key too short");
        }

        let mut offset = 0;

        // Skip TableSetID (4 bytes)
        offset += 4;

        // Check 1_A (AGENT_TABLE_ID)
        let table_id = u32::from_be_bytes([
            key[offset],
            key[offset + 1],
            key[offset + 2],
            key[offset + 3],
        ]);
        if table_id != AGENT_TABLE_ID {
            return Err("invalid table ID");
        }
        offset += 4;

        // Skip 1_B (4 bytes)
        offset += 4;

        // Extract agent_id (16 bytes for UUID)
        if key.len() < offset + 16 + 4 {
            return Err("key too short for agent_id");
        }

        let agent_id_bytes = &key[offset..offset + 16];
        let agent_id_array: [u8; 16] = agent_id_bytes
            .try_into()
            .map_err(|_| "invalid agent_id length")?;
        let agent_id = AgentID::new(agent_id_array);
        offset += 16;

        // Check field ID (AGENT_UPDATED_AT_FIELD)
        let field_id = u32::from_be_bytes([
            key[offset],
            key[offset + 1],
            key[offset + 2],
            key[offset + 3],
        ]);
        if field_id != AGENT_UPDATED_AT_FIELD {
            return Err("invalid field ID");
        }

        Ok(Self { agent_id })
    }
}

/////////////////////////////////////////////// Agent //////////////////////////////////////////////

/// An agent entity stored in the database.
///
/// This struct represents an agent with a unique ID and timestamps
/// for creation and last update. It provides ORM-like operations
/// for interacting with the SST storage layer.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Agent {
    /// Unique identifier for this agent.
    pub agent_id: AgentID,
    /// When this agent was first created.
    pub created_at: std::time::SystemTime,
    /// When this agent was last updated.
    pub updated_at: std::time::SystemTime,
}

impl Agent {
    /// Create a new agent with the given ID.
    /// Both created_at and updated_at are set to the current time.
    pub fn new(agent_id: AgentID) -> Self {
        let now = std::time::SystemTime::now();
        Self {
            agent_id,
            created_at: now,
            updated_at: now,
        }
    }

    /// Update the agent's updated_at timestamp to the current time.
    pub fn touch(&mut self) {
        self.updated_at = std::time::SystemTime::now();
    }

    /// Save this agent to the provided WriteBatch.
    /// This stores the agent as multiple key-value pairs following EXAMPLE.md pattern.
    pub fn save_to_batch(&self, batch: &mut WriteBatch) -> Result<(), AgentBatchError> {
        // Store created_at field
        let created_at_key = AgentCreatedAtKey::new(self.agent_id);
        let created_at_key_bytes = created_at_key.pack();
        let created_at_timestamp_us = self
            .created_at
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| {
                AgentBatchError::SerializationFailed(serde_json::Error::io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid created_at timestamp",
                )))
            })?
            .as_micros() as u64;
        let created_at_bytes = created_at_timestamp_us.to_be_bytes();

        let created_at_kvr = KeyValueRef {
            key: &created_at_key_bytes,
            timestamp: 0,
            value: Some(&created_at_bytes),
        };

        batch.insert(created_at_kvr)?;

        // Store updated_at field
        let updated_at_key = AgentUpdatedAtKey::new(self.agent_id);
        let updated_at_key_bytes = updated_at_key.pack();
        let updated_at_timestamp_us = self
            .updated_at
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| {
                AgentBatchError::SerializationFailed(serde_json::Error::io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid updated_at timestamp",
                )))
            })?
            .as_micros() as u64;
        let updated_at_bytes = updated_at_timestamp_us.to_be_bytes();

        let updated_at_kvr = KeyValueRef {
            key: &updated_at_key_bytes,
            timestamp: 0,
            value: Some(&updated_at_bytes),
        };

        Ok(batch.insert(updated_at_kvr)?)
    }

    /// Delete this agent from the provided WriteBatch.
    /// This creates tombstone records for all agent fields.
    pub fn delete_from_batch(&self, batch: &mut WriteBatch) -> Result<(), AgentBatchError> {
        // Delete created_at field
        let created_at_key = AgentCreatedAtKey::new(self.agent_id);
        let created_at_key_bytes = created_at_key.pack();

        let created_at_kvr = KeyValueRef {
            key: &created_at_key_bytes,
            timestamp: 0,
            value: None,
        };

        batch.insert(created_at_kvr)?;

        // Delete updated_at field
        let updated_at_key = AgentUpdatedAtKey::new(self.agent_id);
        let updated_at_key_bytes = updated_at_key.pack();

        let updated_at_kvr = KeyValueRef {
            key: &updated_at_key_bytes,
            timestamp: 0,
            value: None,
        };

        Ok(batch.insert(updated_at_kvr)?)
    }

    /// Load an agent by gathering data from multiple keys using the cursor.
    /// This method seeks to find both created_at and updated_at fields for the agent.
    pub fn load_by_id<C: Cursor>(
        cursor: &mut C,
        agent_id: AgentID,
    ) -> Result<Option<Self>, AgentCursorError> {
        // Try to load created_at field
        let created_at_key = AgentCreatedAtKey::new(agent_id);
        let created_at_key_bytes = created_at_key.pack();

        cursor.seek(&created_at_key_bytes)?;

        let created_at = if let Some(key) = cursor.key() {
            if key.key == created_at_key_bytes {
                if let Some(value) = cursor.value() {
                    if value.len() == 8 {
                        let timestamp_array: [u8; 8] = value.try_into().map_err(|_| {
                            AgentCursorError::DeserializationFailed(serde_json::Error::io(
                                std::io::Error::new(
                                    std::io::ErrorKind::InvalidData,
                                    "invalid created_at bytes",
                                ),
                            ))
                        })?;
                        let timestamp_us = u64::from_be_bytes(timestamp_array);
                        std::time::UNIX_EPOCH + std::time::Duration::from_micros(timestamp_us)
                    } else {
                        return Err(AgentCursorError::DeserializationFailed(
                            serde_json::Error::io(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                "invalid created_at length",
                            )),
                        ));
                    }
                } else {
                    // Tombstone - agent was deleted
                    return Ok(None);
                }
            } else {
                // Key not found
                return Ok(None);
            }
        } else {
            // No key found
            return Ok(None);
        };

        // Try to load updated_at field
        let updated_at_key = AgentUpdatedAtKey::new(agent_id);
        let updated_at_key_bytes = updated_at_key.pack();

        cursor.seek(&updated_at_key_bytes)?;

        let updated_at = if let Some(key) = cursor.key() {
            if key.key == updated_at_key_bytes {
                if let Some(value) = cursor.value() {
                    if value.len() == 8 {
                        let timestamp_array: [u8; 8] = value.try_into().map_err(|_| {
                            AgentCursorError::DeserializationFailed(serde_json::Error::io(
                                std::io::Error::new(
                                    std::io::ErrorKind::InvalidData,
                                    "invalid updated_at bytes",
                                ),
                            ))
                        })?;
                        let timestamp_us = u64::from_be_bytes(timestamp_array);
                        std::time::UNIX_EPOCH + std::time::Duration::from_micros(timestamp_us)
                    } else {
                        return Err(AgentCursorError::DeserializationFailed(
                            serde_json::Error::io(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                "invalid updated_at length",
                            )),
                        ));
                    }
                } else {
                    // Tombstone - agent was deleted
                    return Ok(None);
                }
            } else {
                // Key not found
                return Ok(None);
            }
        } else {
            // No key found
            return Ok(None);
        };

        Ok(Some(Agent {
            agent_id,
            created_at,
            updated_at,
        }))
    }

    /// Find an agent by its ID using the provided cursor.
    /// This is an alias for load_by_id for API compatibility.
    pub fn find_by_id<C: Cursor>(
        cursor: &mut C,
        agent_id: AgentID,
    ) -> Result<Option<Self>, AgentCursorError> {
        Self::load_by_id(cursor, agent_id)
    }

    /// Iterate over all agents using the provided cursor, calling the closure for each.
    /// The closure should return Ok(true) to continue iteration, Ok(false) to stop.
    ///
    /// This implementation scans for created_at keys to identify unique agent IDs,
    /// then loads each complete agent record.
    pub fn iterate_all<C: Cursor, F>(cursor: &mut C, mut f: F) -> Result<(), AgentCursorError>
    where
        F: FnMut(Self) -> Result<bool, AgentCursorError>,
    {
        // Start at the beginning of the agent table key space
        let table_prefix = {
            let mut prefix = Vec::new();
            prefix.extend_from_slice(&1u32.to_be_bytes()); // TableSetID
            prefix.extend_from_slice(&AGENT_TABLE_ID.to_be_bytes()); // 1_A
            prefix.extend_from_slice(&1u32.to_be_bytes()); // 1_B (agent_id field ID)
            prefix
        };

        cursor.seek(&table_prefix)?;

        let mut seen_agents = std::collections::HashSet::new();

        while let Some(key_ref) = cursor.key() {
            // Check if this key belongs to our table
            if key_ref.key.len() < table_prefix.len() {
                break;
            }

            if key_ref.key[..table_prefix.len()] != *table_prefix {
                break;
            }

            // Try to parse the key to extract agent_id
            if let Ok(created_at_key) = AgentCreatedAtKey::unpack(key_ref.key) {
                // Only process each agent once
                if !seen_agents.contains(&created_at_key.agent_id) {
                    seen_agents.insert(created_at_key.agent_id);

                    // Load the complete agent
                    if let Some(agent) = Self::load_by_id(cursor, created_at_key.agent_id)?
                        && !f(agent)?
                    {
                        return Ok(());
                    }
                }
            }

            // Move to next key
            cursor.next()?;
        }

        Ok(())
    }

    /// Count the total number of agents using the provided cursor.
    pub fn count<C: Cursor>(cursor: &mut C) -> Result<usize, AgentCursorError> {
        let mut count = 0;
        Self::iterate_all(cursor, |_| {
            count += 1;
            Ok(true)
        })?;
        Ok(count)
    }

    /// Check if an agent with the given ID exists using the provided cursor.
    pub fn exists<C: Cursor>(cursor: &mut C, agent_id: AgentID) -> Result<bool, AgentCursorError> {
        Self::find_by_id(cursor, agent_id).map(|opt| opt.is_some())
    }
}

/////////////////////////////////////////////// tests //////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    fn create_test_agent() -> Agent {
        let agent_id =
            AgentID::from_human_readable("agent:12345678-1234-1234-1234-123456789012").unwrap();
        Agent::new(agent_id)
    }

    #[test]
    fn agent_creation_sets_timestamps() {
        let before = SystemTime::now();
        let agent = create_test_agent();
        let after = SystemTime::now();

        assert!(agent.created_at >= before && agent.created_at <= after);
        assert!(agent.updated_at >= before && agent.updated_at <= after);
        assert_eq!(agent.created_at, agent.updated_at);
    }

    #[test]
    fn agent_touch_updates_timestamp() {
        let mut agent = create_test_agent();
        let original_updated_at = agent.updated_at;
        let original_created_at = agent.created_at;

        // Sleep a small amount to ensure timestamp difference
        std::thread::sleep(std::time::Duration::from_millis(1));
        agent.touch();

        assert_eq!(agent.created_at, original_created_at);
        assert!(agent.updated_at > original_updated_at);
    }

    #[test]
    fn agent_key_creation() {
        let agent_id =
            AgentID::from_human_readable("agent:87654321-4321-4321-4321-210987654321").unwrap();
        let created_at_key = AgentCreatedAtKey::new(agent_id);
        assert_eq!(created_at_key.agent_id, agent_id);
    }

    #[test]
    fn agent_key_pack_unpack_roundtrip() {
        let agent_id =
            AgentID::from_human_readable("agent:87654321-4321-4321-4321-210987654321").unwrap();

        let created_at_key = AgentCreatedAtKey::new(agent_id);
        let created_at_packed = created_at_key.pack();
        assert!(!created_at_packed.is_empty());

        let updated_at_key = AgentUpdatedAtKey::new(agent_id);
        let updated_at_packed = updated_at_key.pack();
        assert!(!updated_at_packed.is_empty());

        // Keys for different fields should be different
        assert_ne!(created_at_packed, updated_at_packed);
    }

    #[test]
    fn agent_key_unpack_roundtrip() {
        let agent_id =
            AgentID::from_human_readable("agent:87654321-4321-4321-4321-210987654321").unwrap();

        let created_at_key = AgentCreatedAtKey::new(agent_id);
        let created_at_packed = created_at_key.pack();
        let unpacked_key = AgentCreatedAtKey::unpack(&created_at_packed).unwrap();
        assert_eq!(created_at_key.agent_id, unpacked_key.agent_id);

        let updated_at_key = AgentUpdatedAtKey::new(agent_id);
        let updated_at_packed = updated_at_key.pack();
        let unpacked_key = AgentUpdatedAtKey::unpack(&updated_at_packed).unwrap();
        assert_eq!(updated_at_key.agent_id, unpacked_key.agent_id);
    }

    #[test]
    fn agent_serialization_roundtrip() {
        let agent = create_test_agent();

        let json = serde_json::to_vec(&agent).unwrap();
        let deserialized: Agent = serde_json::from_slice(&json).unwrap();

        assert_eq!(agent.agent_id, deserialized.agent_id);
        assert_eq!(agent.created_at, deserialized.created_at);
        assert_eq!(agent.updated_at, deserialized.updated_at);
    }

    #[test]
    fn agent_batch_error_display() {
        let json_err = serde_json::Error::io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "test error",
        ));
        let batch_err = AgentBatchError::SerializationFailed(json_err);

        let display = format!("{}", batch_err);
        assert!(display.contains("Agent serialization failed"));
    }

    #[test]
    fn agent_cursor_error_display() {
        let json_err = serde_json::Error::io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "test error",
        ));
        let cursor_err = AgentCursorError::DeserializationFailed(json_err);

        let display = format!("{}", cursor_err);
        assert!(display.contains("Agent deserialization failed"));

        let key_err = AgentCursorError::KeyParsingFailed("invalid format");
        let display = format!("{}", key_err);
        assert!(display.contains("Agent key parsing failed: invalid format"));
    }

    #[test]
    fn agent_batch_error_from_serde() {
        let json_err =
            serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::InvalidData, "test"));
        let batch_err: AgentBatchError = json_err.into();

        match batch_err {
            AgentBatchError::SerializationFailed(_) => {}
            _ => panic!("Expected SerializationFailed variant"),
        }
    }

    #[test]
    fn agent_cursor_error_from_serde() {
        let json_err =
            serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::InvalidData, "test"));
        let cursor_err: AgentCursorError = json_err.into();

        match cursor_err {
            AgentCursorError::DeserializationFailed(_) => {}
            _ => panic!("Expected DeserializationFailed variant"),
        }
    }

    // Mock cursor implementation for testing
    struct MockCursor {
        // Store multiple key-value pairs to simulate agent storage
        data: Vec<(Vec<u8>, Vec<u8>)>,
        position: usize,
        seek_called: bool,
        next_called: bool,
    }

    impl MockCursor {
        fn new() -> Self {
            Self {
                data: Vec::new(),
                position: 0,
                seek_called: false,
                next_called: false,
            }
        }

        fn with_agent(agent: &Agent) -> Self {
            let mut data = Vec::new();

            // Add created_at key-value pair
            let created_at_key = AgentCreatedAtKey::new(agent.agent_id);
            let created_at_timestamp_us = agent
                .created_at
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64;
            data.push((
                created_at_key.pack(),
                created_at_timestamp_us.to_be_bytes().to_vec(),
            ));

            // Add updated_at key-value pair
            let updated_at_key = AgentUpdatedAtKey::new(agent.agent_id);
            let updated_at_timestamp_us = agent
                .updated_at
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64;
            data.push((
                updated_at_key.pack(),
                updated_at_timestamp_us.to_be_bytes().to_vec(),
            ));

            Self {
                data,
                position: 0,
                seek_called: false,
                next_called: false,
            }
        }

        fn with_invalid_value() -> Self {
            let agent_id =
                AgentID::from_human_readable("agent:12345678-1234-1234-1234-123456789012").unwrap();
            let created_at_key = AgentCreatedAtKey::new(agent_id);
            let data = vec![(created_at_key.pack(), b"invalid".to_vec())];

            Self {
                data,
                position: 0,
                seek_called: false,
                next_called: false,
            }
        }
    }

    impl Cursor for MockCursor {
        fn seek(&mut self, key: &[u8]) -> Result<(), SstError> {
            self.seek_called = true;

            // Find the first key >= the seek key
            self.position = 0;
            for (i, (stored_key, _)) in self.data.iter().enumerate() {
                if stored_key.as_slice() >= key {
                    self.position = i;
                    return Ok(());
                }
            }

            // No key found >= seek key, position at end
            self.position = self.data.len();
            Ok(())
        }

        fn seek_to_first(&mut self) -> Result<(), SstError> {
            self.position = 0;
            Ok(())
        }

        fn seek_to_last(&mut self) -> Result<(), SstError> {
            self.position = if self.data.is_empty() {
                0
            } else {
                self.data.len() - 1
            };
            Ok(())
        }

        fn prev(&mut self) -> Result<(), SstError> {
            if self.position > 0 {
                self.position -= 1;
            }
            Ok(())
        }

        fn next(&mut self) -> Result<(), SstError> {
            self.next_called = true;
            if self.position < self.data.len() {
                self.position += 1;
            }
            Ok(())
        }

        fn key(&self) -> Option<sst::KeyRef<'_>> {
            if self.position < self.data.len() {
                Some(sst::KeyRef {
                    key: &self.data[self.position].0,
                    timestamp: 0,
                })
            } else {
                None
            }
        }

        fn value(&self) -> Option<&[u8]> {
            if self.position < self.data.len() {
                Some(&self.data[self.position].1)
            } else {
                None
            }
        }
    }

    #[test]
    fn agent_load_by_id_success() {
        let agent = create_test_agent();
        let mut cursor = MockCursor::with_agent(&agent);

        let result = Agent::load_by_id(&mut cursor, agent.agent_id);
        assert!(result.is_ok());
        let loaded_agent = result.unwrap();
        assert!(loaded_agent.is_some());
        let loaded_agent = loaded_agent.unwrap();
        assert_eq!(loaded_agent.agent_id, agent.agent_id);
        // Note: timestamps may have slight differences due to truncation, so check they're close
        let time_diff = agent
            .created_at
            .duration_since(loaded_agent.created_at)
            .unwrap_or_else(|_| {
                loaded_agent
                    .created_at
                    .duration_since(agent.created_at)
                    .unwrap()
            });
        assert!(time_diff < std::time::Duration::from_millis(1));
    }

    #[test]
    fn agent_load_by_id_no_key() {
        let mut cursor = MockCursor::new();
        let agent_id =
            AgentID::from_human_readable("agent:00000000-0000-0000-0000-000000000000").unwrap();

        let result = Agent::load_by_id(&mut cursor, agent_id);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn agent_load_by_id_invalid_value() {
        let mut cursor = MockCursor::with_invalid_value();
        let agent_id =
            AgentID::from_human_readable("agent:12345678-1234-1234-1234-123456789012").unwrap();

        let result = Agent::load_by_id(&mut cursor, agent_id);
        // Should fail due to invalid timestamp value length
        assert!(result.is_err());
    }

    #[test]
    fn agent_find_by_id_success() {
        let agent = create_test_agent();
        let mut cursor = MockCursor::with_agent(&agent);

        let result = Agent::find_by_id(&mut cursor, agent.agent_id).unwrap();
        assert!(result.is_some());

        let found_agent = result.unwrap();
        assert_eq!(agent.agent_id, found_agent.agent_id);
        assert!(cursor.seek_called);
    }

    #[test]
    fn agent_find_by_id_not_found() {
        let mut cursor = MockCursor::new();
        let agent_id =
            AgentID::from_human_readable("agent:00000000-0000-0000-0000-000000000000").unwrap();

        let result = Agent::find_by_id(&mut cursor, agent_id).unwrap();
        assert!(result.is_none());
        assert!(cursor.seek_called);
    }

    #[test]
    fn agent_iterate_all_success() {
        let agent = create_test_agent();
        let mut cursor = MockCursor::with_agent(&agent);

        let mut count = 0;
        let result = Agent::iterate_all(&mut cursor, |found_agent| {
            count += 1;
            assert_eq!(agent.agent_id, found_agent.agent_id);
            Ok(true) // Continue iteration
        });

        assert!(result.is_ok());
        assert_eq!(count, 1);
        // next_called might be true due to iteration
        assert!(cursor.seek_called);
    }

    #[test]
    fn agent_iterate_all_early_termination() {
        let agent = create_test_agent();
        let mut cursor = MockCursor::with_agent(&agent);

        let mut count = 0;
        let result = Agent::iterate_all(&mut cursor, |_| {
            count += 1;
            Ok(false) // Stop iteration
        });

        assert!(result.is_ok());
        assert_eq!(count, 1);
        assert!(cursor.seek_called);
    }

    #[test]
    fn agent_count() {
        let agent = create_test_agent();
        let mut cursor = MockCursor::with_agent(&agent);

        let count = Agent::count(&mut cursor).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn agent_count_empty() {
        let mut cursor = MockCursor::new();
        let count = Agent::count(&mut cursor).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn agent_exists_true() {
        let agent = create_test_agent();
        let mut cursor = MockCursor::with_agent(&agent);

        let exists = Agent::exists(&mut cursor, agent.agent_id).unwrap();
        assert!(exists);
    }

    #[test]
    fn agent_exists_false() {
        let mut cursor = MockCursor::new();
        let agent_id =
            AgentID::from_human_readable("agent:00000000-0000-0000-0000-000000000000").unwrap();

        let exists = Agent::exists(&mut cursor, agent_id).unwrap();
        assert!(!exists);
    }
}
