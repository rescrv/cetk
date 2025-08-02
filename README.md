# cetk

cetk is the context-engineer's toolkit (for building agents).

It solves the following problems:
- **durability**:  Persist agent state, including conversation messages, file system contents, and
  application state.
- **virtual filesystems**:  Persist changes to a virtual filesystem for use in offloading of state.
- **compaction**:  Compaction is a first-class primitive.  Access every context going back to the
  beginnning of time so that state is not lost; compact so that everything fits in the context
  window of your favorite Anthropic model.
- **memory**:  For an LLM, memory is the act of knowing that which is not (yet) in context.  This
  takes several shapes in cetk:
  - **map(message history)**:  It is possible, but increasingly expensive (and quadratic) to walk
    the entire maintained history of contexts, going back through time.  This allows for
    fact-finding agents to extract high-fidelity facts from prior history.  Literally perfect
    memory, limited only by the quality of the fact-finder.
  - **index(message history)**:  Instead, the conversation gets indexed in some fasion (Chroma!) and
    the top-k contexts going back through time can be used for fact-finding until a satisfactory
    answer is had.
- **agents**:  Integrates with the `claudius::Agent` trait to provide these features
  nearly-out-of-the-box as a batteries-included agents framework.

## Assumptions

- You are building long-lived agents.
- Think continual processes.
- I'm using synchronous Rust because async is only needed when this moves on from being a prototype.
- In order to build up enough state to make this expensive, you will have incurred enough LLM costs
  to ask me my vision for the project.

## Durability

Unless you plan to write perfect code and never again deploy code, you need to persist your agent's
state somewhere so that it can survive a new code push.

cetk aims to be a one-size-fits-most solution to this.  It provides a transactional way to update
the context, compact the context, or read the context at a point-in-time.  And it's possible to
combine these primitives to do a point-in-time restore.

The core abstraction is a sequence of contexts.  Each context is a sequence of transactions that
extend the message list, write the filesystem, and record application state.  From this we can
derive that contexts are append-only and correspond to an ever-growing LLM conversation.  Compaction
turns one context into another.  The most basic compaction would be to summarize in an initial
message, but that's not the only strategy permitted by the cetk.

cetk provides transactional access to the latest context; the transactions are pessimsitic locking
so that it is possible to put an LLM call under the transaction without races or expensive spurious
aborts.

## Virtual Filesystem

cetk provides a virtual filesystem abstraction so that it is possible to give Claude access to
"files" for note keeping and organization.

A virtual filesystem is identified by a UUID; it can be written within the context of a transaction,
and can be read at any time.

The advantage of using a virtual filesystem is that there's no need to create another filesystem for
scratch space.  The claudius crate natively supports UNIX-like mounts for filesystems, so it is
possible to mount a virtual filesystem and real filesystem simultaneously.

## Compaction

Compaction is the act of modifying the context in a way that does not simply extend it.  The
durability abstraction cetk provides saves every context and uses compaction as the sole vehicle for
creating a new context for an agent.

Because compaction is intimately tied to the nature of one's application, cetk provides a way to do
arbitrary transformations while it handles the concurrency that excludes concurrent compactions.

## Memory

Memory is the act of knowing that which is not (yet) in context.  Remembering is the act of
transferring something from elsewhere to in-context.  cetk provides two paths for remembering:
Perfect memory and cheap memory.

## Perfect Memory

Given that every context ever instantiated is kept, there is never lost information; one need only
pay the cost to re-evaluate every context with a question to be able to find the answer one seeks.

Perfect memory, therefore, is about providing a way to go backwards in time until a context is found
that has an answer to the prompt.  It is as simple as a for loop, and as expensive as context window
full of tokens.

## Cheap, Fast Memory

The cost barriers to perfect memory become cost prohibitive.  Instead, we can use information
retrieval to get cheap, fast memory at the loss of some ability to recall things about the past.

One such way to do this would be to store each message in the exchange as its own object.  Searching
for past messages becomes easy in this scenario.  Unfortunately, it's never quite that easy:  simply
retrieving related messages can miss contextual clues, and fail to capture details that span
multiple messages.

A step-function improvement would be to use the messages as an index over the contexts and somehow
score them using the number or quality of messages retrieved.  The top-k contexts going back through
time can be used for fact-finding until a satisfactory answer is had.  For short-lived tasks this
degrades to near-perfect memory.  For agents that last weeks, months, or years (the goal), this will
be cheaper and faster to consult because it will be O(1) in the agent's history.

## The Format

cetk is currently a prototype and uses a single JSONL file per context to store the transactions,
one per line.  This is because cetk currently isn't used in multi-tenant applications.  Eventually,
a key-value store backend will support multi-tenant applications.  For now, it's intended to
prototype and explore the best ways to build individual agents.  The best patterns will be ported to
the key-value backend.

## TODO

- Integrate with Chroma.
- Point-in-time reads off the filesystem.
