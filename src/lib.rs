#![doc = include_str!("../README.md")]
//!
//! ## Context Engineer's Toolkit (CETK)
//!
//! This crate provides core types and utilities for the Context Engineer's Toolkit:

use one_two_eight::generate_id;

mod embeddings;
mod transaction;
mod transaction_manager;

pub use embeddings::{EmbeddingModel, EmbeddingService};
pub use transaction::{
    ChunkSizeExceededError, FileWrite, FromChunksError, InvariantViolation, Transaction,
    TransactionChunk, TransactionSerializationError,
};
pub use transaction_manager::{TransactionManager, TransactionManagerError};

///////////////////////////////////////////// Constants ////////////////////////////////////////////

pub const CHUNK_SIZE_LIMIT: usize = 8192;

///////////////////////////////////////// generate_id_serde ////////////////////////////////////////

/// Generate the serde Deserialize/Serialize routines for a one_two_eight ID.
macro_rules! generate_id_crate {
    ($name:ident, $visitor:ident) => {
        impl tuple_key::Element for $name {
            const DATA_TYPE: tuple_key::KeyDataType = tuple_key::KeyDataType::string;

            fn append_to(&self, key: &mut tuple_key::TupleKey) {
                self.prefix_free_readable().append_to(key);
            }

            fn parse_from(bytes: &[u8]) -> Result<Self, &'static str> {
                let uuid_str = String::parse_from(bytes)?;
                let id_bytes = one_two_eight::decode(&uuid_str).ok_or("invalid UUID format")?;
                Ok(Self::new(id_bytes))
            }
        }

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
generate_id_crate!(AgentID, AgentIDVisitor);

/////////////////////////////////////////// TransactionID //////////////////////////////////////////

generate_id!(TransactionID, "tx:");
generate_id_crate!(TransactionID, TransactionIDVisitor);

////////////////////////////////////////////// MountID /////////////////////////////////////////////

generate_id!(MountID, "mount:");
generate_id_crate!(MountID, MountIDVisitor);

///////////////////////////////////////////// ContextID ////////////////////////////////////////////

generate_id!(ContextID, "context:");
generate_id_crate!(ContextID, ContextIDVisitor);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_size_limit_is_set_correctly() {
        assert_eq!(CHUNK_SIZE_LIMIT, 8192);
    }

    #[test]
    fn agent_id_creates_from_human_readable() {
        let id_str = "agent:00000000-0000-0000-0000-000000000001";
        let agent_id = AgentID::from_human_readable(id_str);
        assert!(agent_id.is_some());
        assert_eq!(agent_id.unwrap().to_string(), id_str);
    }

    #[test]
    fn agent_id_rejects_invalid_prefix() {
        let invalid_id = "invalid:00000000-0000-0000-0000-000000000001";
        let agent_id = AgentID::from_human_readable(invalid_id);
        assert!(agent_id.is_none());
    }

    #[test]
    fn agent_id_rejects_malformed_uuid() {
        let invalid_id = "agent:not-a-uuid";
        let agent_id = AgentID::from_human_readable(invalid_id);
        assert!(agent_id.is_none());
    }

    #[test]
    fn agent_id_generates_unique_ids() {
        let id1 = AgentID::generate().unwrap();
        let id2 = AgentID::generate().unwrap();
        assert_ne!(id1, id2);
        assert!(id1.to_string().starts_with("agent:"));
        assert!(id2.to_string().starts_with("agent:"));
    }

    #[test]
    fn agent_id_roundtrip_string_conversion() {
        let original = AgentID::generate().unwrap();
        let string_repr = original.to_string();
        let restored = AgentID::from_human_readable(&string_repr).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn agent_id_serializes_to_json() {
        let agent_id =
            AgentID::from_human_readable("agent:12345678-1234-1234-1234-123456789012").unwrap();
        let json = serde_json::to_string(&agent_id).unwrap();
        assert_eq!(json, "\"agent:12345678-1234-1234-1234-123456789012\"");
    }

    #[test]
    fn agent_id_deserializes_from_json() {
        let json = "\"agent:12345678-1234-1234-1234-123456789012\"";
        let agent_id: AgentID = serde_json::from_str(json).unwrap();
        assert_eq!(
            agent_id.to_string(),
            "agent:12345678-1234-1234-1234-123456789012"
        );
    }

    #[test]
    fn agent_id_deserialization_rejects_invalid_format() {
        let invalid_json = "\"invalid:12345678-1234-1234-1234-123456789012\"";
        let result: Result<AgentID, _> = serde_json::from_str(invalid_json);
        assert!(result.is_err());
    }

    #[test]
    fn agent_id_deserialization_rejects_malformed_uuid() {
        let invalid_json = "\"agent:not-a-valid-uuid\"";
        let result: Result<AgentID, _> = serde_json::from_str(invalid_json);
        assert!(result.is_err());
    }

    #[test]
    fn agent_id_serde_roundtrip() {
        let original = AgentID::generate().unwrap();
        let json = serde_json::to_string(&original).unwrap();
        let restored: AgentID = serde_json::from_str(&json).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn transaction_id_creates_from_human_readable() {
        let id_str = "tx:00000000-0000-0000-0000-000000000001";
        let tx_id = TransactionID::from_human_readable(id_str);
        assert!(tx_id.is_some());
        assert_eq!(tx_id.unwrap().to_string(), id_str);
    }

    #[test]
    fn transaction_id_rejects_invalid_prefix() {
        let invalid_id = "transaction:00000000-0000-0000-0000-000000000001";
        let tx_id = TransactionID::from_human_readable(invalid_id);
        assert!(tx_id.is_none());
    }

    #[test]
    fn transaction_id_generates_unique_ids() {
        let id1 = TransactionID::generate().unwrap();
        let id2 = TransactionID::generate().unwrap();
        assert_ne!(id1, id2);
        assert!(id1.to_string().starts_with("tx:"));
        assert!(id2.to_string().starts_with("tx:"));
    }

    #[test]
    fn transaction_id_serializes_to_json() {
        let tx_id =
            TransactionID::from_human_readable("tx:12345678-1234-1234-1234-123456789012").unwrap();
        let json = serde_json::to_string(&tx_id).unwrap();
        assert_eq!(json, "\"tx:12345678-1234-1234-1234-123456789012\"");
    }

    #[test]
    fn transaction_id_deserializes_from_json() {
        let json = "\"tx:12345678-1234-1234-1234-123456789012\"";
        let tx_id: TransactionID = serde_json::from_str(json).unwrap();
        assert_eq!(tx_id.to_string(), "tx:12345678-1234-1234-1234-123456789012");
    }

    #[test]
    fn transaction_id_serde_roundtrip() {
        let original = TransactionID::generate().unwrap();
        let json = serde_json::to_string(&original).unwrap();
        let restored: TransactionID = serde_json::from_str(&json).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn mount_id_creates_from_human_readable() {
        let id_str = "mount:00000000-0000-0000-0000-000000000001";
        let mount_id = MountID::from_human_readable(id_str);
        assert!(mount_id.is_some());
        assert_eq!(mount_id.unwrap().to_string(), id_str);
    }

    #[test]
    fn mount_id_rejects_invalid_prefix() {
        let invalid_id = "filesystem:00000000-0000-0000-0000-000000000001";
        let mount_id = MountID::from_human_readable(invalid_id);
        assert!(mount_id.is_none());
    }

    #[test]
    fn mount_id_generates_unique_ids() {
        let id1 = MountID::generate().unwrap();
        let id2 = MountID::generate().unwrap();
        assert_ne!(id1, id2);
        assert!(id1.to_string().starts_with("mount:"));
        assert!(id2.to_string().starts_with("mount:"));
    }

    #[test]
    fn mount_id_serializes_to_json() {
        let mount_id =
            MountID::from_human_readable("mount:12345678-1234-1234-1234-123456789012").unwrap();
        let json = serde_json::to_string(&mount_id).unwrap();
        assert_eq!(json, "\"mount:12345678-1234-1234-1234-123456789012\"");
    }

    #[test]
    fn mount_id_deserializes_from_json() {
        let json = "\"mount:12345678-1234-1234-1234-123456789012\"";
        let mount_id: MountID = serde_json::from_str(json).unwrap();
        assert_eq!(
            mount_id.to_string(),
            "mount:12345678-1234-1234-1234-123456789012"
        );
    }

    #[test]
    fn mount_id_serde_roundtrip() {
        let original = MountID::generate().unwrap();
        let json = serde_json::to_string(&original).unwrap();
        let restored: MountID = serde_json::from_str(&json).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn context_id_creates_from_human_readable() {
        let id_str = "context:00000000-0000-0000-0000-000000000001";
        let context_id = ContextID::from_human_readable(id_str);
        assert!(context_id.is_some());
        assert_eq!(context_id.unwrap().to_string(), id_str);
    }

    #[test]
    fn context_id_rejects_invalid_prefix() {
        let invalid_id = "ctx:00000000-0000-0000-0000-000000000001";
        let context_id = ContextID::from_human_readable(invalid_id);
        assert!(context_id.is_none());
    }

    #[test]
    fn context_id_generates_unique_ids() {
        let id1 = ContextID::generate().unwrap();
        let id2 = ContextID::generate().unwrap();
        assert_ne!(id1, id2);
        assert!(id1.to_string().starts_with("context:"));
        assert!(id2.to_string().starts_with("context:"));
    }

    #[test]
    fn context_id_serializes_to_json() {
        let context_id =
            ContextID::from_human_readable("context:12345678-1234-1234-1234-123456789012").unwrap();
        let json = serde_json::to_string(&context_id).unwrap();
        assert_eq!(json, "\"context:12345678-1234-1234-1234-123456789012\"");
    }

    #[test]
    fn context_id_deserializes_from_json() {
        let json = "\"context:12345678-1234-1234-1234-123456789012\"";
        let context_id: ContextID = serde_json::from_str(json).unwrap();
        assert_eq!(
            context_id.to_string(),
            "context:12345678-1234-1234-1234-123456789012"
        );
    }

    #[test]
    fn context_id_serde_roundtrip() {
        let original = ContextID::generate().unwrap();
        let json = serde_json::to_string(&original).unwrap();
        let restored: ContextID = serde_json::from_str(&json).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn different_id_types_are_distinct() {
        // Generate IDs with similar UUIDs but different prefixes
        let uuid_str = "12345678-1234-1234-1234-123456789012";

        let agent_id = AgentID::from_human_readable(&format!("agent:{}", uuid_str)).unwrap();
        let tx_id = TransactionID::from_human_readable(&format!("tx:{}", uuid_str)).unwrap();
        let mount_id = MountID::from_human_readable(&format!("mount:{}", uuid_str)).unwrap();
        let context_id = ContextID::from_human_readable(&format!("context:{}", uuid_str)).unwrap();

        // Verify they have different string representations
        assert_ne!(agent_id.to_string(), tx_id.to_string());
        assert_ne!(agent_id.to_string(), mount_id.to_string());
        assert_ne!(agent_id.to_string(), context_id.to_string());
        assert_ne!(tx_id.to_string(), mount_id.to_string());
        assert_ne!(tx_id.to_string(), context_id.to_string());
        assert_ne!(mount_id.to_string(), context_id.to_string());
    }

    #[test]
    fn id_equality_works() {
        let id1 =
            AgentID::from_human_readable("agent:12345678-1234-1234-1234-123456789012").unwrap();
        let id2 =
            AgentID::from_human_readable("agent:12345678-1234-1234-1234-123456789012").unwrap();
        let id3 =
            AgentID::from_human_readable("agent:87654321-4321-4321-4321-210987654321").unwrap();

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert_ne!(id2, id3);
    }

    #[test]
    fn id_handles_empty_string() {
        assert!(AgentID::from_human_readable("").is_none());
        assert!(TransactionID::from_human_readable("").is_none());
        assert!(MountID::from_human_readable("").is_none());
        assert!(ContextID::from_human_readable("").is_none());
    }

    #[test]
    fn id_handles_only_prefix() {
        assert!(AgentID::from_human_readable("agent:").is_none());
        assert!(TransactionID::from_human_readable("tx:").is_none());
        assert!(MountID::from_human_readable("mount:").is_none());
        assert!(ContextID::from_human_readable("context:").is_none());
    }

    #[test]
    fn id_handles_missing_colon() {
        assert!(
            AgentID::from_human_readable("agent00000000-0000-0000-0000-000000000001").is_none()
        );
        assert!(
            TransactionID::from_human_readable("tx00000000-0000-0000-0000-000000000001").is_none()
        );
        assert!(
            MountID::from_human_readable("mount00000000-0000-0000-0000-000000000001").is_none()
        );
        assert!(
            ContextID::from_human_readable("context00000000-0000-0000-0000-000000000001").is_none()
        );
    }

    #[test]
    fn id_handles_case_sensitive_prefix() {
        // Prefixes should be case sensitive
        assert!(
            AgentID::from_human_readable("AGENT:00000000-0000-0000-0000-000000000001").is_none()
        );
        assert!(
            TransactionID::from_human_readable("TX:00000000-0000-0000-0000-000000000001").is_none()
        );
        assert!(
            MountID::from_human_readable("MOUNT:00000000-0000-0000-0000-000000000001").is_none()
        );
        assert!(
            ContextID::from_human_readable("CONTEXT:00000000-0000-0000-0000-000000000001")
                .is_none()
        );
    }

    #[test]
    fn id_handles_uuid_with_wrong_format() {
        // Wrong number of dashes
        assert!(AgentID::from_human_readable("agent:00000000000000000000000000000001").is_none());
        // Wrong length segments
        assert!(AgentID::from_human_readable("agent:000-00-00-00-000").is_none());
        // Non-hex characters
        assert!(
            AgentID::from_human_readable("agent:gggggggg-gggg-gggg-gggg-gggggggggggg").is_none()
        );
    }

    #[test]
    fn serde_error_handling_non_string() {
        // Test deserialization from non-string JSON values
        let invalid_json = "123";
        let result: Result<AgentID, _> = serde_json::from_str(invalid_json);
        assert!(result.is_err());
    }

    #[test]
    fn serde_error_handling_null() {
        let null_json = "null";
        let result: Result<AgentID, _> = serde_json::from_str(null_json);
        assert!(result.is_err());
    }

    #[test]
    fn serde_error_handling_array() {
        let array_json = "[\"agent:00000000-0000-0000-0000-000000000001\"]";
        let result: Result<AgentID, _> = serde_json::from_str(array_json);
        assert!(result.is_err());
    }

    #[test]
    fn id_with_minimum_uuid() {
        let min_uuid = "00000000-0000-0000-0000-000000000000";

        let agent_id = AgentID::from_human_readable(&format!("agent:{}", min_uuid));
        let tx_id = TransactionID::from_human_readable(&format!("tx:{}", min_uuid));
        let mount_id = MountID::from_human_readable(&format!("mount:{}", min_uuid));
        let context_id = ContextID::from_human_readable(&format!("context:{}", min_uuid));

        assert!(agent_id.is_some());
        assert!(tx_id.is_some());
        assert!(mount_id.is_some());
        assert!(context_id.is_some());
    }

    #[test]
    fn id_with_maximum_uuid() {
        let max_uuid = "ffffffff-ffff-ffff-ffff-ffffffffffff";

        let agent_id = AgentID::from_human_readable(&format!("agent:{}", max_uuid));
        let tx_id = TransactionID::from_human_readable(&format!("tx:{}", max_uuid));
        let mount_id = MountID::from_human_readable(&format!("mount:{}", max_uuid));
        let context_id = ContextID::from_human_readable(&format!("context:{}", max_uuid));

        assert!(agent_id.is_some());
        assert!(tx_id.is_some());
        assert!(mount_id.is_some());
        assert!(context_id.is_some());
    }

    #[test]
    fn generated_ids_have_correct_format() {
        let agent_id = AgentID::generate().unwrap();
        let tx_id = TransactionID::generate().unwrap();
        let mount_id = MountID::generate().unwrap();
        let context_id = ContextID::generate().unwrap();

        // Verify format: prefix:xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        let agent_str = agent_id.to_string();
        let tx_str = tx_id.to_string();
        let mount_str = mount_id.to_string();
        let context_str = context_id.to_string();

        assert!(agent_str.starts_with("agent:"));
        assert_eq!(agent_str.len(), "agent:".len() + 36); // UUID is 36 chars
        assert_eq!(agent_str.matches('-').count(), 4); // UUID has 4 dashes

        assert!(tx_str.starts_with("tx:"));
        assert_eq!(tx_str.len(), "tx:".len() + 36);
        assert_eq!(tx_str.matches('-').count(), 4);

        assert!(mount_str.starts_with("mount:"));
        assert_eq!(mount_str.len(), "mount:".len() + 36);
        assert_eq!(mount_str.matches('-').count(), 4);

        assert!(context_str.starts_with("context:"));
        assert_eq!(context_str.len(), "context:".len() + 36);
        assert_eq!(context_str.matches('-').count(), 4);
    }
}
