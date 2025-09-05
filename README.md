# cetk

cetk is the context-engineer's toolkit (for building agents).

It solves the following problems:
- **durability**:  Persist agent state, including conversation messages, file system contents, and
  application state in Chroma.
- **virtual filesystems**:  Persist changes to one of many virtual filesystems for use in offloading
  of state.
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

## Durability

Unless you plan to write perfect code and never again deploy code, you need to persist your agent's
state somewhere so that it can survive a new code push.

cetk aims to be a one-size-fits-most solution to this.  It provides a transactional way to update
the context, compact the context, or read the context at a point-in-time.  And it's possible to
combine these primitives to do a point-in-time restore.

The core abstraction is a sequence of contexts.  Each context is a sequence of transactions that
extend the message list and write the filesystem.  From this we can derive that contexts are
append-only and correspond to an ever-growing LLM conversation.  Compaction turns one context into
another.  The most basic compaction would be to summarize in an initial message, but that's not the
only strategy permitted by the cetk.

cetk provides transactional access to the latest context; the transactions are enforced via
pessimsitic locking so that it is possible to put an LLM call under the transaction without races or
expensive, spurious races and aborts.

## Virtual Filesystem

cetk provides a virtual filesystem abstraction so that it is possible to give Claude access to
"files" for note keeping and organization.

A virtual filesystem is identified by a UUID; it can be written within the context of a transaction,
and can be read at any time.

The advantage of using a virtual filesystem is that there's no need to create another filesystem for
scratch space.  The claudius crate natively supports UNIX-like mounts for filesystems, so it is
possible to mount a virtual filesystem and real filesystem simultaneously.

The practical application of this is to implement two different types of file systems:
- General, chunked files.
- Special subsets of markdown restricted to sections containing bulleted lists.

The special subsets of markdown are easy to manipulate programatically e.g.
`context.section("Foo").insert("A new bullet point is made.")` and easy to index with
per-bullet-point high fidelity in Chroma.

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

Given that every context ever instantiated is maintained in Chroma, there is never lost information;
one need only pay the cost to re-evaluate every context with a question to be able to find the
answer one seeks.

Perfect memory, therefore, is about providing a way to go backwards in time until a context is found
that has an answer to the open question.  It is as simple as a for loop, and as expensive as context
window full of tokens many iterations over.

The key to perfect memory is to roleplay the following scenario to an LLM at compaction time:  You
are about to retire.  You have not completed your task.  Write a complete description of the
hand-off of work done thus far and remaining work to be done.  Assume that they have unlimited
bandwidth for follow-up questions at a cost of $x per question.

This sets up a recursive---in the computer science sense---memory structure where each context
relies upon the one that came before it to answer questions about the task at hand, falling back to
the user.  A way of peeling back layers on the onion.  Construct a similar prompt to the above and
have each agent ask questions of the previous context with perfect fidelity at a known, articulable
cost.

## Cheap, Fast Memory

The cost barriers to perfect memory become cost prohibitive in a long-running agent.  Instead, we
can use information retrieval to get cheap, fast memory at the loss of some ability to recall things
about the past.

One such way to do this would be to store each message in the exchange as its own object.  Searching
for past messages becomes easy in this scenario.  Unfortunately, it's never quite that easy:  simply
retrieving related messages can miss contextual clues, and fail to capture details that span
multiple messages.

A step-function improvement would be to use the messages as an index over the contexts and somehow
score them using the number or quality of messages retrieved.  The top-k contexts going back through
time can be used for fact-finding until a satisfactory answer is had.  For short-lived tasks this
degrades to near-perfect memory.  For agents that last weeks, months, or years (the goal), this will
be cheaper and faster to consult because it will be O(1) in the agent's history.

The key to making cheap, fast memory work is to think of it as a log of contemporaneous notes.  If
you can neatly categorize your application as a log, then you can tell the LLM to consider the
temporal nature of its question.  If it is a stable property in the Chandy-Lamport sense, then it
should be acceptable to answer the question with cheap, fast memory.

## The Format

It's important to understand that the format of cetk on top of Chroma is as much a part of the
public API as the public API in this crate.  For that reason (and so Claude can navigate) I am fully
documenting the structure here.

Provision Chroma collections at the level you would a database or schema in a traditional RDBMS.
There's a delicate balancing act to be struck.  Chroma is certainly capable of supporting an
application with one collection per long-lived (e.g. 1:1 with a human) agent.  The problem comes in
that this becomes extremely inefficient to query for problems.  When building an agent you want to
be able to query across different instances to see underlying trends.

To that end, a Chroma collection is populated by the following objects, all scoped with a metadata
`agent=UUID` annotation and with keys that share the `agent=UUID/` prefix:
- **context transactions**:  A transaction extends the context.
    - metadata: context=UUID transaction=seq_no chunk=seq_no
    - key: `context=UUID;transaction=seq_no;chunk=seq_no`.
    - document:  A 16kB chunk of text.
    - embedding:  embed(summary of transaction).
- **context seal**:  A context seal says the next natural context to follow a context.  It is
  possible for many contexts to follow (forked conversations), but only only one will be chosen by
  the parent to continue the lineage.
    - metadata: context=UUID
    - key: `context=UUID;seal`
    - document:  A single UUID
    - embedding:  embed(search_document:  A summary of this context.).
- **chunked file**:  Not currently supported due to complexity.
- **markdown lists**:  A set of sets, presented as a possibly-section-delimited list of bullet
  points in Markdown.
    - metadata: mount=UUID path=/path/to/file.md section="section header".
    - key: `mount=UUID;path=/path/to/file.md;section=section header/bullet`.
    - document:  A single < 16kB bullet.
    - embedding:  embed(document).
- **lock**:  A lock file, one per agent.
    - metadata: No additional metadata.
    - key: `LOCK`
    - document:  A textual description of the caller.
    - embedding:  [0; DIM].
