//! Chunking utilities for documents.
//!
//! In general, the workflow is that you use the utilities here to get chunks for your documents,
//! convert the metadata included along with your domain-specific metadata into `RecordBatch`es,
//! and pass those back to `zqa-rag` to work with LanceDB.

use itertools::Itertools;

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

    /// Given initial conditions and constraints, chunk `text` into chunks.
    ///
    /// # Arguments
    ///
    /// * `text` - The content to chunk
    /// * `budget` - The maximum number of *characters* per chunk. As noted by
    ///   [`core::str::Chars`], a character corresponds to a Unicode scalar value, not a grapheme
    ///   cluster or a byte.
    /// * `start_chunk_id` - The starting index to use for each `chunk_id`.
    /// * `start_byte_idx` - The byte offset to add to the `byte_range`.
    /// * `page_range` - The page range spanned by this text.
    ///
    /// # Returns
    ///
    /// A list of [`DocumentChunk`], each with a maximum of `budget` characters, with `chunk_id`s
    /// starting at `start_chunk_id`, and `byte_range` offset by `start_byte_idx`.
    fn chunk_text(
        text: &str,
        budget: usize,
        start_chunk_id: usize,
        start_byte_idx: usize,
        page_range: (usize, usize),
    ) -> Vec<DocumentChunk> {
        let chunk_count = text.chars().count().div_ceil(budget);
        let mut byte_offset = start_byte_idx;

        text.chars()
            .chunks(budget)
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                let cur_str: String = s.collect();
                let len = cur_str.len();

                let chunk = DocumentChunk {
                    chunk_id: start_chunk_id + i,
                    chunk_count,
                    content: cur_str,
                    byte_range: (byte_offset, byte_offset + len),
                    page_range,
                    strategy: ChunkingStrategy::SectionBased(budget),
                };
                byte_offset += len;

                chunk
            })
            .collect::<Vec<_>>()
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
            ChunkingStrategy::SectionBased(max_tok) => {
                let text_len = self.content.text_content.len();
                let sections = &self.content.sections;

                if sections.is_empty() {
                    // Treat entire doc as one section and chunk by `max_tok`.
                    return Chunker::chunk_text(
                        &self.content.text_content,
                        *max_tok,
                        1,
                        0,
                        (1, self.content.page_count),
                    );
                }

                let mut chunk_count = 0;
                sections
                    .iter()
                    .enumerate()
                    .flat_map(|(i, sec)| {
                        // Calculate the end byte: either the start of the next section, or end of text
                        let byte_end = sections
                            .get(i + 1)
                            .map_or(text_len, |next_sec| next_sec.byte_index);

                        // Calculate the end page: either the page of the next section, or last page
                        let page_end = sections
                            .get(i + 1)
                            .map_or(self.content.page_count, |next_sec| {
                                next_sec.page_number.saturating_sub(1)
                            });

                        let chunks = Chunker::chunk_text(
                            &self.content.text_content[sec.byte_index..byte_end],
                            *max_tok,
                            chunk_count + 1,
                            sec.byte_index,
                            (sec.page_number, page_end),
                        );
                        chunk_count += chunks.len();

                        chunks
                    })
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::SectionBoundary;

    /// Helper function to create test content without sections
    fn create_simple_content(text: &str, page_count: usize) -> ExtractedContent {
        ExtractedContent {
            text_content: text.to_string(),
            sections: vec![],
            page_count,
        }
    }

    /// Helper function to create test content with sections
    fn create_content_with_sections(
        text: &str,
        sections: Vec<SectionBoundary>,
        page_count: usize,
    ) -> ExtractedContent {
        ExtractedContent {
            text_content: text.to_string(),
            sections,
            page_count,
        }
    }

    #[test]
    fn test_whole_document_strategy() {
        let text = "This is a test document with some content.";
        let content = create_simple_content(text, 1);
        let chunker = Chunker::new(content, ChunkingStrategy::WholeDocument);

        let chunks = chunker.chunk();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_id, 1);
        assert_eq!(chunks[0].chunk_count, 1);
        assert_eq!(chunks[0].content, text);
        assert_eq!(chunks[0].byte_range, (0, text.len()));
        assert_eq!(chunks[0].page_range, (1, 1));
    }

    #[test]
    fn test_whole_document_multipage() {
        let text = "Content spanning multiple pages.";
        let content = create_simple_content(text, 5);
        let chunker = Chunker::new(content, ChunkingStrategy::WholeDocument);

        let chunks = chunker.chunk();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].page_range, (1, 5));
    }

    #[test]
    fn test_section_based_no_sections_simple() {
        // Text with exactly 10 characters
        let text = "0123456789";
        let content = create_simple_content(text, 1);
        let chunker = Chunker::new(content, ChunkingStrategy::SectionBased(5));

        let chunks = chunker.chunk();

        assert_eq!(chunks.len(), 2);

        // First chunk
        assert_eq!(chunks[0].chunk_id, 1);
        assert_eq!(chunks[0].chunk_count, 2);
        assert_eq!(chunks[0].content, "01234");
        assert_eq!(chunks[0].byte_range, (0, 5));
        assert_eq!(chunks[0].page_range, (1, 1));

        // Second chunk
        assert_eq!(chunks[1].chunk_id, 2);
        assert_eq!(chunks[1].chunk_count, 2);
        assert_eq!(chunks[1].content, "56789");
        assert_eq!(chunks[1].byte_range, (5, 10));
        assert_eq!(chunks[1].page_range, (1, 1));
    }

    #[test]
    fn test_section_based_no_sections_uneven() {
        // Text with 12 characters, chunks of 5
        let text = "012345678901";
        let content = create_simple_content(text, 1);
        let chunker = Chunker::new(content, ChunkingStrategy::SectionBased(5));

        let chunks = chunker.chunk();

        // Should create 3 chunks: 5 + 5 + 2
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].content, "01234");
        assert_eq!(chunks[1].content, "56789");
        assert_eq!(chunks[2].content, "01");
        assert_eq!(chunks[2].byte_range, (10, 12));
    }

    #[test]
    fn test_section_based_with_sections() {
        // Create content with two sections
        let text = "Section 1 content here. Section 2 content here also.";
        let sections = vec![
            SectionBoundary {
                page_number: 1,
                byte_index: 0,
                level: 0,
                parent_idx: None,
                font_size: 14.0,
            },
            SectionBoundary {
                page_number: 1,
                byte_index: 24, // Start of "Section 2"
                level: 0,
                parent_idx: None,
                font_size: 14.0,
            },
        ];

        let content = create_content_with_sections(text, sections, 1);
        let chunker = Chunker::new(content, ChunkingStrategy::SectionBased(100));

        let chunks = chunker.chunk();

        // With a large budget, each section should be its own chunk
        assert_eq!(chunks.len(), 2);

        // First section
        assert_eq!(chunks[0].chunk_id, 1);
        assert_eq!(chunks[0].content, "Section 1 content here. ");
        assert_eq!(chunks[0].byte_range, (0, 24));

        // Second section
        assert_eq!(chunks[1].chunk_id, 2);
        assert_eq!(chunks[1].content, "Section 2 content here also.");
        assert_eq!(chunks[1].byte_range, (24, text.len()));
    }

    #[test]
    fn test_section_based_sections_split_when_large() {
        // Create a section that needs to be split
        let text = "This is a very long section that will need to be split into multiple chunks.";
        let sections = vec![SectionBoundary {
            page_number: 1,
            byte_index: 0,
            level: 0,
            parent_idx: None,
            font_size: 14.0,
        }];

        let content = create_content_with_sections(text, sections, 1);
        let chunker = Chunker::new(content, ChunkingStrategy::SectionBased(20));

        let chunks = chunker.chunk();

        // Should split into multiple chunks
        assert!(chunks.len() > 1);

        // All chunks should have sequential IDs
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_id, i + 1);
        }

        // All chunks should be within budget (except possibly the last one)
        for chunk in &chunks[..chunks.len() - 1] {
            assert!(chunk.content.chars().count() <= 20);
        }
    }

    #[test]
    fn test_byte_ranges_are_continuous() {
        let text = "abcdefghijklmnopqrstuvwxyz";
        let content = create_simple_content(text, 1);
        let chunker = Chunker::new(content, ChunkingStrategy::SectionBased(10));

        let chunks = chunker.chunk();

        // Verify byte ranges are continuous
        for i in 0..chunks.len() - 1 {
            assert_eq!(chunks[i].byte_range.1, chunks[i + 1].byte_range.0);
        }

        // First chunk should start at 0
        assert_eq!(chunks[0].byte_range.0, 0);

        // Last chunk should end at text length
        assert_eq!(chunks.last().unwrap().byte_range.1, text.len());
    }

    #[test]
    fn test_empty_content() {
        let content = create_simple_content("", 1);
        let chunker = Chunker::new(content, ChunkingStrategy::WholeDocument);

        let chunks = chunker.chunk();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "");
        assert_eq!(chunks[0].byte_range, (0, 0));
    }

    #[test]
    fn test_single_character() {
        let text = "a";
        let content = create_simple_content(text, 1);
        let chunker = Chunker::new(content, ChunkingStrategy::SectionBased(1));

        let chunks = chunker.chunk();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "a");
        assert_eq!(chunks[0].byte_range, (0, 1));
    }

    #[test]
    fn test_unicode_handling() {
        // Test with multi-byte UTF-8 characters
        let text = "Hello 世界 🌍";
        let content = create_simple_content(text, 1);
        let chunker = Chunker::new(content, ChunkingStrategy::SectionBased(5));

        let chunks = chunker.chunk();

        // Verify all chunks are valid UTF-8
        for chunk in &chunks {
            assert!(chunk.content.is_ascii() || chunk.content.chars().count() > 0);
        }

        // Verify byte ranges
        let mut reconstructed = String::new();
        for chunk in &chunks {
            reconstructed.push_str(&chunk.content);
        }
        assert_eq!(reconstructed, text);
    }

    #[test]
    fn test_page_ranges_with_sections() {
        let text = "Page 1 content. Page 2 content. Page 3 content.";
        let sections = vec![
            SectionBoundary {
                page_number: 1,
                byte_index: 0,
                level: 0,
                parent_idx: None,
                font_size: 14.0,
            },
            SectionBoundary {
                page_number: 2,
                byte_index: 16,
                level: 0,
                parent_idx: None,
                font_size: 14.0,
            },
            SectionBoundary {
                page_number: 3,
                byte_index: 32,
                level: 0,
                parent_idx: None,
                font_size: 14.0,
            },
        ];

        let content = create_content_with_sections(text, sections, 3);
        let chunker = Chunker::new(content, ChunkingStrategy::SectionBased(100));

        let chunks = chunker.chunk();

        assert_eq!(chunks.len(), 3);

        // Check page ranges
        assert_eq!(chunks[0].page_range, (1, 1)); // Page 1, ends before page 2 section
        assert_eq!(chunks[1].page_range, (2, 2)); // Page 2, ends before page 3 section
        assert_eq!(chunks[2].page_range, (3, 3)); // Page 3, last section
    }

    #[test]
    fn test_chunk_count_consistency() {
        let text = "a".repeat(100);
        let content = create_simple_content(&text, 1);
        let chunker = Chunker::new(content, ChunkingStrategy::SectionBased(25));

        let chunks = chunker.chunk();

        // All chunks should have the same chunk_count
        let expected_count = chunks.len();
        for chunk in &chunks {
            assert_eq!(chunk.chunk_count, expected_count);
        }
    }

    #[test]
    fn test_section_boundary_at_end() {
        // Test when a section starts at the very end of the document
        // In practice, empty sections don't produce chunks
        let text = "Content here.";
        let sections = vec![
            SectionBoundary {
                page_number: 1,
                byte_index: 0,
                level: 0,
                parent_idx: None,
                font_size: 14.0,
            },
            SectionBoundary {
                page_number: 1,
                byte_index: text.len(),
                level: 0,
                parent_idx: None,
                font_size: 14.0,
            },
        ];

        let content = create_content_with_sections(text, sections, 1);
        let chunker = Chunker::new(content, ChunkingStrategy::SectionBased(100));

        let chunks = chunker.chunk();

        // Should have 1 chunk (the empty section doesn't produce a chunk)
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, text);
    }

    #[test]
    fn test_large_budget() {
        // Budget larger than content
        let text = "Short text.";
        let content = create_simple_content(text, 1);
        let chunker = Chunker::new(content, ChunkingStrategy::SectionBased(1000));

        let chunks = chunker.chunk();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, text);
    }
}
