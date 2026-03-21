# zqa-pdftools

A work-in-progress PDF parser tailored to academic papers. Extracts text, detects document structure, converts math font symbols, and splits documents into embedding-ready chunks. Ignores tables and images.

## Features

- **Text extraction**: Parses PDF content streams, skipping images and tables
- **Section detection**: Infers hierarchy from font size analysis
- **Math symbol conversion**: Translates Computer Modern font encodings (CMEX, CMMI, CMSY, MSBM) to LaTeX representations
- **Document chunking**: Section-based or whole-document strategies
- **Configurable thresholds**: Tune word-spacing and table-detection heuristics

## Usage

### Extract text and sections

```rust
use zqa_pdftools::parse::{ExtractedContent, extract_text};

let content: ExtractedContent = extract_text("paper.pdf")?;

println!("Pages: {}", content.page_count);
for section in &content.sections {
    println!("[L{}] {}", section.level, &content.text_content[section.byte_index..]);
}
```

### Chunk a document

```rust
use zqa_pdftools::chunk::{Chunker, ChunkingStrategy};

let chunker = Chunker::new(content);
let chunks = chunker.chunk(ChunkingStrategy::SectionBased(512));

for chunk in chunks {
    println!("Pages {:?}, {} bytes", chunk.page_range, chunk.content.len());
}
```

## Key Types

### `ExtractedContent`

Result of a successful parse:

| Field | Type | Description |
|-------|------|-------------|
| `text_content` | `String` | Full extracted text |
| `sections` | `Vec<SectionBoundary>` | Detected section boundaries |
| `page_count` | `usize` | Total page count |

### `SectionBoundary`

```rust
pub struct SectionBoundary {
    pub page_number: usize,
    pub byte_index:  usize,   // offset into ExtractedContent::text_content
    pub level:       u8,      // 0 = title, 1 = section, 2 = subsection, …
    pub parent_idx:  Option<usize>,
    pub font_size:   f32,
}
```

### `ChunkingStrategy`

```rust
pub enum ChunkingStrategy {
    WholeDocument,
    SectionBased(usize),  // max tokens per chunk
}
```

### `DocumentChunk`

```rust
pub struct DocumentChunk {
    pub chunk_id:    usize,
    pub chunk_count: usize,
    pub content:     String,
    pub byte_range:  std::ops::Range<usize>,
    pub page_range:  std::ops::Range<usize>,
    pub strategy:    ChunkingStrategy,
}
```

## Tuning

`PdfParserThresholds` exposes three parameters:

| Constant | Default | Effect |
|----------|---------|--------|
| `same_word_threshold` | `60.0` | Max horizontal gap (pts) treated as same word |
| `table_euclidean_threshold` | `40.0` | Distance threshold for table cell grouping |
| `tbl_td_threshold` | `5` | Minimum cells to classify a region as a table |

## Limitations

- Still a work in progress; complex multi-column layouts may not parse correctly
- Table and figure content is intentionally skipped
- Math extraction is limited to Computer Modern font families

## MSRV

Rust **1.91** (edition 2024).
