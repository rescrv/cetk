use crate::AgentID;

///////////////////////////////////////////// AgentKey /////////////////////////////////////////////

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize, tuple_key_derive::TypedTupleKey)]
pub struct AgentKey {
    #[tuple_key(1)]
    pub agent_id: AgentID,
}

/////////////////////////////////////////////// Agent //////////////////////////////////////////////

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Agent {
    pub agent_id: AgentID,
    pub created_at: std::time::SystemTime,
    pub updated_at: std::time::SystemTime,
}

impl Agent {
    pub fn new(agent_id: AgentID) -> Self {
        todo!();
    }

    pub fn touch(&mut self) {
        self.updated_at = std::time::SystemTime::now();
    }

    // TODO(claude):  Somehow save the agent once updated.
}
