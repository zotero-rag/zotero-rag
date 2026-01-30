//! Chunking utilities for documents.
//!
//! In general, the workflow is that you use the utilities here to get chunks for your documents,
//! convert the metadata included along with your domain-specific metadata into `RecordBatch`es,
//! and pass those back to `zqa-rag` to work with LanceDB.

use crate::parse::ExtractedContent;

/// A single chunk in a document. It's unlikely you will want to use this directly; instead, you
/// probably want to use the utilities here to get these chunks for you.
pub struct DocumentChunk {
    /// The (1-indexed) serial number of the chunk in the document.
    pub chunk_id: usize,
    /// The number of chunks in the document.
    pub chunk_count: usize,
    /// The content in the chunk
    pub content: String,
    /// The (0-indexed) byte range of this chunk in the content. Bounds are [inclusive, exclusive).
    pub byte_range: (usize, usize),
    /// The (1-indexed) page range this chunk spans. Bounds are [inclusive, inclusive].
    pub page_range: (usize, usize),
    /// The chunking strategy used
    pub strategy: ChunkingStrategy,
}

/// A set of chunking strategies you can use. Although you can use any strategy with any provider,
/// the `zqa-rag` crate has some recommendations (that you are free to ignore!) that you can get from
/// `EmbeddingProvider::recommended_chunking_strategy`.
#[derive(Clone, Copy, Debug)]
pub enum ChunkingStrategy {
    /// Store the entire document in one chunk. This is the simplest strategy and works quite well
    /// with providers such as Voyage AI and Cohere that perform truncation for you.
    WholeDocument,
    /// Chunk by sections, merging sections where possible and splitting sections where forced to.
    /// Takes a `usize` describing the max token count per chunk.
    SectionBased(usize),
}

/// A struct that is responsible for chunking documents based on a strategy.
pub struct Chunker {
    /// The raw unchunked content
    content: ExtractedContent,
    /// The chunking strategy
    strategy: ChunkingStrategy,
}

impl Chunker {
    /// Create a new chunker for given content, using a specified chunking strategy.
    #[must_use]
    pub fn new(content: ExtractedContent, strategy: ChunkingStrategy) -> Self {
        Self { content, strategy }
    }

    /// Chunk the document using the strategy selected in [`Chunker::new`].
    ///
    /// # Returns
    ///
    /// A list of [`DocumentChunk`] objects, each of which contains the contents and related
    /// metadata for that chunk.
    #[must_use]
    pub fn chunk(&self) -> Vec<DocumentChunk> {
        match &self.strategy {
            ChunkingStrategy::WholeDocument => {
                let text_len = self.content.text_content.len();
                vec![DocumentChunk {
                    chunk_id: 1,
                    chunk_count: 1,
                    content: self.content.text_content.clone(),
                    byte_range: (0, text_len),
                    page_range: (1, self.content.page_count),
                    strategy: ChunkingStrategy::WholeDocument,
                }]
            }
            ChunkingStrategy::SectionBased(_max_tok) => {
                let text_len = self.content.text_content.len();
                let sections = &self.content.sections;

                if sections.is_empty() {
                    // If no sections, treat as whole document
                    return vec![DocumentChunk {
                        chunk_id: 1,
                        chunk_count: 1,
                        content: self.content.text_content.clone(),
                        byte_range: (0, text_len),
                        page_range: (1, self.content.page_count),
                        strategy: self.strategy,
                    }];
                }

                sections
                    .iter()
                    .enumerate()
                    .map(|(i, sec)| {
                        // Calculate the end byte: either the start of the next section, or end of text
                        let byte_end = sections
                            .get(i + 1)
                            .map_or(text_len, |next_sec| next_sec.byte_index);

                        // Calculate the end page: either the page of the next section, or last page
                        let page_end = sections
                            .get(i + 1)
                            .map_or(self.content.page_count, |next_sec| next_sec.page_number);

                        DocumentChunk {
                            chunk_id: i + 1,
                            chunk_count: sections.len(),
                            content: self.content.text_content[sec.byte_index..byte_end]
                                .to_string(),
                            byte_range: (sec.byte_index, byte_end),
                            page_range: (sec.page_number, page_end),
                            strategy: self.strategy,
                        }
                    })
                    .collect()
            }
        }
    }
}
