<macguffin>
We're building an agents framework called cetk.  It must store a
map[agent_id]map[context_seq_no]map[transaction_seq_no] struct.  Below is the complete data
structure we need for our motivating example.

NOTE: Field numbers are reused for efficiency (e.g., multiple fields can use number 1).
The letter suffixes (_A, _B, _C, etc.) in this example are only for distinguishing
different instances of the same field number - they are NOT part of the actual field numbering.

<table_definition>
table Agent(string agent_id = 1_B) @ 1_A {
    timestamp created_at = 2_A;
    timestamp updated_at = 3_A;
    column Context(uint64 context_seq_no = 1_C) @ 4 {
        column Transaction(uint32 transaction_seq_no = 1_D) @ 1_E {
            repeated MessageParam messages = 2_B;
            repeated FileWrite writes = 3_B;
            }
        }
    }
}
</table_definition>
<key-value-pairs>
(TableSetID, 1_A, 1_B, agent_id, 2_A) -> created_at
(TableSetID, 1_A, 1_B, agent_id, 3_A) -> updated_at
(TableSetID, 1_A, 1_B, agent_id, 4, 1_C, context_seq_no, 1_E, 1_D, transaction_seq_no, 2_B, <index>) -> messages[index]
(TableSetID, 1_A, 1_B, agent_id, 4, 1_C, context_seq_no, 1_E, 1_D, transaction_seq_no, 3_B, <index>) -> writes[index]
</key-value-pairs>
</macguffin>
