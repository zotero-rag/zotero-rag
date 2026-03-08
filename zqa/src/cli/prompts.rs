use crate::tools::retrieval::RETRIEVAL_TOOL_NAME;
use crate::tools::summarization::SUMMARIZATION_TOOL_NAME;
use crate::utils::library::ZoteroItemMetadata;

/// Get the "extraction" prompt, which extracts the relevant parts of a retrieved paper. This is
/// the first step of the question-answering process.
///
/// # Arguments
///
/// * `query` - The user query to answer
/// * `pdf_text` - The full text of the research paper
///
/// # Returns
///
/// * `prompt` - The prompt for extracting the relevant parts of the query.
#[must_use]
pub fn get_extraction_prompt(query: &str, pdf_text: &str, metadata: &ZoteroItemMetadata) -> String {
    let authors = metadata.authors.as_ref().map_or_else(
        || "No author list provided; please infer from the text.".into(),
        |list| list.join("; "),
    );
    let title = &metadata.title;

    format!("Given a question from a user and the full text from a research paper, extract the relevant
parts that are suitable for answering the question. Wrap each relevant excerpt from the paper in
<excerpt></excerpt> tags. Here are some guidelines:

1. It is best to quote excerpts verbatim. You may correct spelling errors, but do not modify other
parts of the text.
2. Some parts of the text, especially equations, may appear as hard to understand text. This is a
currently-known limitation: for now, repeat these verbatim. You may add a best-guess of what the
equation was supposed to be, in LaTeX form and wrapped in double-dollar signs ($$...$$), inside
parentheses after such malformed text. Inside the parentheses, indicate that this is a \"possible fix\".
Aside from this, do not modify the source text.
3. You are encouraged to also cite excerpts from the paper that cite other papers, *if these excerpts
are relevant*. In such cases, if the citation uses numbers, change the numbering to be an author-year
format, preferably in the APA style. Write all your references at the end of your response, after a
\"References:\" header. If the user specifically requests a different citation style such as
MLA, you should use that instead.
4. Some text from the paper's Appendix or Supplementary Material may be in this text. Do not use
excerpts from this material.

Here is the user query: <user_query>{query}</user_query>.

Below is some information about the paper:
<metadata>
    <title>{title}</title>
    <authors>{authors}</authors>
</metadata>

Below is the full text of the paper:

<pdf_text>
{pdf_text}
</pdf_text>

Format your response in exactly the below format:

<title>Paper title here</title>
<authors>Author list</authors>
<reference>An APA-style citation to the current paper</reference>
<excerpt>An excerpt here</excerpt>
<excerpt>Another excerpt here</excerpt>

Note that if the user requests references to be in a different format (e.g., MLA, Chicago), you should use that
reference format instead of APA.")
}

/// Get the "title" prompt, which generates a short title for a conversation based on the user
/// query.
///
/// # Arguments
///
/// * `query` - The user query to generate a title for
///
/// # Returns
///
/// * `prompt` - The prompt for generating a title
#[must_use]
pub fn get_title_prompt(query: &str) -> String {
    format!(
        "You are given a user query to an AI assistant that grounds its response in the user's
Zotero library. Generate a short, descriptive title for this conversation in 10 words or fewer. \
The title should capture the main topic. Respond with only the title text, no quotes or punctuation.

<user_query>
{query}
</user_query>"
    )
}

/// Get the "summarization" prompt, which takes the excerpts from each search result and then asks
/// the LLM to generate a researched answer to the user query.
///
/// # Arguments
///
/// * `query` - The user query to answer
///
/// # Returns
///
/// * `prompt` - The prompt for answering the user query
#[must_use]
pub fn get_summarize_prompt(query: &str) -> String {
    format!(
        "You are given a user question and excerpts from papers that are relevant in answering the question.
Each paper that was used as a reference may have multiple excerpts that potentially answer the user's
question. Use these search results to draft an answer to the user query. 

You have access to tools to help retrieve results and summarize results from the user's Zotero library. Your
answer MUST be grounded in these results.
* The `{RETRIEVAL_TOOL_NAME}` will retrieve metadata of papers that seem relevant, but not their contents.
* The `{SUMMARIZATION_TOOL_NAME}` takes a list of paper IDs (which you can get from the `{RETRIEVAL_TOOL_NAME}`),
and for each paper, produces a list of passages that are most relevant to the user's query.

Here are some guidelines when replying:

1. Your answer must maintain a scholarly, formal tone. Write your answer as though it were part of a research
paper discussing relevant work. However, if the user asks for a different tone or format, you should follow
those instructions instead.
2. In the (rare) event that the user query asks an unsolved problem, summarize the relevant work from
the search results, but preface your answer by politely explaining that the problem is known to be
unsolved. Unsolved problems do *not* include problems currently under active study, and only include known-unsolved
problems (e.g., solving the Halting problem).
3. It is important that your answer ends with a \"References:\" section. You should list all cited references
here in APA format.
4. When using the excerpts to write your answer, ignore numbering in citations; use APA-style citations
throughout. You should change numbered citations to be in APA format instead.
5. If the user requests references to be in a different format (e.g., MLA, Chicago, etc.), use that format instead.
This includes numbered formats: if the user prefers a numbered citation style, use that instead.
6. In certain places, the excerpts may have an equation marked with \"possible fix\". This indicates that a different
agent found a malformed equation, and attempted to fix it. If this fixed version appears correct, use that instead.
7. Format the answer depending on the user's request. For example, if the user requests a summary of prior work on a
problem, it is appropriate to use Markdown-formatted sections. In other cases, a user may ask a question with a
straightforward answer (based on the search results). In this case, it is usually better to answer the question
directly.
8. Not all search results and excerpts may be relevant. You do *not* need to use *all* the excerpts and search results.
Instead, use excerpts and search results that have relevant information to the user's query. If you are unsure, err
on the side of *inclusion*. It is better to erroneously include a paper than to falsely ignore one.

Here is the user query: <user_query>{query}</user_query>.")
}
