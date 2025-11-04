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
pub fn get_extraction_prompt(query: &str, pdf_text: &str) -> String {
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
\"References:\" header.
4. Some text from the paper's Appendix or Supplementary Material may be in this text. Do not use
excerpts from this material.

Here is the user query: <user_query>{query}</user_query>.
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

/// Get the "summarization" prompt, which takes the excerpts from each search result and then asks
/// the LLM to generate a researched answer to the user query.
///
/// # Arguments
///
/// * `query` - The user query to answer
/// * `excerpts` - The excerpts from each search result
///
/// # Returns
///
/// * `prompt` - The prompt for answering the user query
pub fn get_summarize_prompt(query: &str, excerpts: Vec<&String>) -> String {
    let search_results = excerpts
        .iter()
        .map(|res| format!("<search_result>{res}</search_result>"))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "You are given a user question and excerpts from papers that are relevant in answering the question.
Each paper that was used as a reference may have multiple excerpts that potentially answer the user's
question. Use these search results to draft an answer to the user query. Here are some guidelines:

1. Your answer must maintain a scholarly, formal tone. Write your answer as though it were part of a research
paper discussing relevant work.
2. In the (rare) event that the user query asks an unsolved problem, summarize the relevant work from
the search results, but preface your answer by politely explaining that the problem is known to be
unsolved. Unsolved problems do *not* include problems currently under active study, and only include known-unsolved
problems (e.g., solving the Halting problem).
3. Each set of excerpts from a paper is wrapped in <search_result>...</search_result>. Each such search
result will include an APA-style citation to the paper from which relevant excerpts are taken. Within
each <search_result>, you will find the paper title, authors, and excerpts, possibly including references to other 
papers. The end of each <search_result> will have references to papers that have been cited.
4. It is important that your answer ends with a \"References:\" section. You should list all cited references
here in APA format.
5. When using the excerpts to write your answer, ignore numbering in citations; use APA-style citations 
throughout. You should change numbered citations to be in APA format instead.
6. If the user requests references to be in a different format (e.g., MLA, Chicago, etc.), use that format instead.
7. In certain places, the excerpts may have an equation marked with \"possible fix\". This indicates that a different
agent found a malformed equation, and attempted to fix it. If this fixed version appears correct, use that instead.
8. Format the answer depending on the user's request. For example, if the user requests a summary of prior work on a 
problem, it is appropriate to use Markdown-formatted sections. In other cases, a user may ask a question with a 
straightforward answer (based on the search results). In this case, it is usually better to answer the question 
directly.
9. Not all search results and excerpts may be relevant. You do *not* need to use *all* the excerpts and search results.
Instead, use excerpts and search results that have relevant information to the user's query. If you are unsure, err
on the side of *inclusion*. It is better to erroneously include a paper than to falsely ignore one.

Here is the user query: <user_query>{query}</user_query>.
Here are the search results: 

<search_results>
{search_results}
</search_results>")
}
