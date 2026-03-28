# Design: Importing PDFs

Linear issue: ZOT-128  
GitHub issue: [#166](https://github.com/zotero-rag/zotero-rag/issues/166)

Author: Rahul Yedida

## Background

A feature of interest is "study sessions", which offers users a NotebookLM-style experience. In particular, users should be able to add PDFs to the current session that are not part of their Zotero libraries. This has several use cases:

* The user may not want a PDF to be part of their Zotero library.
* The user may want to import a PDF that is not a paper, but lab notes, proof-of-concepts, etc.

This design document focuses on importing PDFs to the current session.

## Design

The core part of this design is to store data about the imported PDF(s) in state. This state will be stored in the `State` struct in `zqa::common`. We use the `Chunker` from `zqa_pdftools::chunk`, and use `ChunkingStrategy::SectionBased` to obtain a `Vec<DocumentChunk>`, which we store in the state. The chunker guarantees that a section begins right after the previous section ends (which the unit tests cover). Concretely, the state stores a `Vec` of the following structure:

```rust
struct UserDocument {
    filename: String,
    chunks: Vec<DocumentChunk>,
    summary: String
    // Leave room for other metadata
}
```

Under the assumption that the first section in a paper is titled "Introduction"; if we do not find it within some threshold, we instead use the first section that is larger than some threshold (under the assumption that the preface is typically relatively small, and the introduction is larger).

The documents imported by the user are made available to the models via tools that spawn sub-agents. Sub-agents are responsible for extracting relevant content from the imported documents; we do this to preserve the token limit of the "root" agent. Moreover, this also prevents polluting the main agent's context if many documents are imported. With this in place, the root agent can be prompted in the system message to use this tool to access the documents if needed. In general, users who import documents will typically refer to them in their query (e.g., "What are the main contributions of the document I added?"). Documents that are imported remain in state with the same lifetime as the session, i.e., until the user uses `/new` or exits and reopens the CLI.

The tool provided to the root agent is responsible for all activities related to extracting content via sub-agents. That there even are sub-agents involved is made opaque. This is because the exact mechanism of extracting relevant content may change over time, and is an implementation detail.

## Rejected alternatives

### Pseudo-DB

This feature could also be implemented by providing a facade that abstracts over all data sources, including the user documents, the LanceDB, and future sources such as the arXiv API (which is also a planned feature). In this design, the "enabled" sources are configured via flags, and the `zqa` crate is responsible for providing this facade (since `zqa_rag` is only responsible for working with LanceDB).

This design was rejected because the required machinery is overkill for this problem. Instead, tools provide a natural way for the root agent to query any sources it deems necessary. Further, the flags could still be implemented in the proposed design, which are implemented by simply not providing the disabled tools to the agent.

### Structure in state

An alternative structure in the state is a `HashMap<String, Document>`, mapping filenames to some metadata. This was rejected because if the filename is not descriptive (as is often the case with PDFs downloaded from arXiv, OpenReview, etc.), the key is useless.
