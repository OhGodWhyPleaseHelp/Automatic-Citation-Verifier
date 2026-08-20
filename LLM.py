from openai import OpenAI, AsyncOpenAI
import re
import json
import asyncio
import threading
from utils import log_usage
import os
from dotenv import load_dotenv

load_dotenv()
MODEL_URL = os.environ.get("MODEL1_URL")
MODEL_NAME = os.environ.get("MODEL1_NAME")
MODEL_API_KEY = os.environ.get("MODEL1_API_KEY")

def sync_client():
    client = OpenAI(
        api_key=MODEL_API_KEY,
        base_url=MODEL_URL
    )
    MODEL = MODEL_NAME
    return client, MODEL


def async_client():
    client = AsyncOpenAI(
        api_key=MODEL_API_KEY,
        base_url=MODEL_URL
    )
    MODEL = MODEL_NAME
    return client, MODEL


def split_reply(model_reply: str, identifier: str = 'json') -> str:
    """
    The model reply usually comes with a <think> attached.
    This splits it and removes the identifier, like "```json".
    :param model_reply: output from the LLM.
    :param identifier: LLM identifier to show what type of output it is (e.g. python, json)
    :return: Only the output of the LLM
    """
    model_reply = model_reply.replace("Trying again...", '')
    split = model_reply.split(f"```{identifier}")
    if len(split) == 1 or '</think>' in split[-1]:
        split = model_reply.split(f"</think>")
        if len(split) == 1:
            raise Exception(f"Splitting the model reply has gone wrong!\n{model_reply}")
    output = split[-1]
    output = output.replace("```", "")
    output = output.replace("```", "")
    return output


def findUniqueIdentifier(text: str):
    client, MODEL = sync_client()
    reply = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """"You are a strict JSON-only extraction engine.
Your task is to analyze the provided text and determine whether it contains a reference list or bibliography section.

If a reference list exists, you must extract the exact boundary substring that separates the main body from the reference list.

Rules:
- You must always return valid JSON.
- Do not include any text outside of the JSON object.
- The identifier MUST be a contiguous, verbatim substring of the provided text.
- The identifier must include ALL characters (including whitespace, newlines, page numbers, separators, or symbols) between the end of the last non-reference content and the reference list section title.
- Do not normalize, trim, or modify whitespace.
- The identifier must match exactly with the provided text.
- The identifier must include the title of the reference list section (e.g., references, bibliography).
- You may use regex (e.g., Python regex) only if the boundary contains variable elements such as dates or page numbers. If regex is used, you must state that you used it.
- If a reference list exists, return:
  - "status": "exists"
  - "identifier": a string containing the identifier to split on.
  - "regex": boolean if the identifier uses regex.
- If no reference list exists, return:
  - "status": "error"
  - "message": a brief explanation describing why there there is no previous sentence.
                  """
            },
            {
                "role": "user",
                "content": """Analyze the following text and determine whether it contains a reference list. If it does, extract the exact boundary substring that separates the main body from the reference list.
Return only JSON.

TEXT:
######
""" + text + """
######

Example 1:
TEXT:
######
we want to thank the sutd-zju idea visiting professor grant (sutd-zju (vp) 202103 and the sutd-zju thematic research grant (sutd-zju (tr) 202204) for supporting this work. this work is supported under deeps-ai (design,education, engineering, psychology, and science with ai).\nreferences\n[1] t. h. teo, “a practical approach to mitigate the dependencies of generative ai in engineering education,” 2024, available at ssrn: https://ssrn.com/abstract=4855880, doi: 10.2139/ssrn.4855880.
######
Output:
{
"status": "exists"
"identifier": "this work is supported under deeps-ai (design,education, engineering, psychology, and science with ai).\nreferences"
"regex": false
}

Example 2:
TEXT:
######
data availability
the data and the code supporting this study’s findings are available at
https://github.com/ai-chem/nanominer.git.
received: 6 december 2024; accepted: 23 may 2025;      14
bibliography
1. foppiano, l., lambard, g., amagasa, t. & ishii, m. mining
experimental data from materials science literature with large
language models: an evaluation study. sci. technol. adv. mater.
methods 4, 2356506 (2024).
######
Output:
{
"status": "exists"
"identifier": "received:\\s*6 december 2024;\\s*accepted:\\s*23 may 2025;\\s*14\\s*bibliography"
"regex": true
}

Example 3:
TEXT:
######
generative ai is becoming inevitable in academic writing. while it accelerates idea documentation, the issue of
fabricated references remains a critical barrier to trust. this
paper introduces a python-based citation verification tool that
leverages the openalex api to check the authenticity of ai
generated references
######
Output:
{
"status": "error"
"message": "no reference list found"
}
                """
            }
        ],
        temperature=0,
        top_p=0.9
    )

    log_usage(reply.usage.total_tokens, reply.usage.prompt_tokens,
              reply.usage.completion_tokens, "findUniqueIdentifier", reply.model)

    return split_reply(model_reply=reply.choices[0].message.content, identifier='json')


def classify_reference_list_section(subsection: str) -> str:
    client, MODEL = sync_client()
    reply = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """"You are a strict JSON-only classification engine.
Your task is to analyze the provided text and determine whether it is part of a reference list or bibliography section.

Rules:
- You must always return valid JSON.
- Do not include any text outside of the JSON object.
- Do not include any explanation of the output.
- If a reference list exists, return:
  - "exists": "true"
- If no reference list exists, return:
  - "exists": "false"
  - "message": a brief explanation describing why there there is no previous sentence.
    """
            },
            {
                "role": "user",
                "content": """Analyze the following text and determine whether it contains a reference list.
Return only JSON.

Example 1:
TEXT:
######
we want to thank the sutd-zju idea visiting professor grant (sutd-zju (vp) 202103 and the sutd-zju thematic research grant (sutd-zju (tr) 202204) for supporting this work. this work is supported under deeps-ai (design,education, engineering, psychology, and science with ai).\nreferences\n
######

Output:
{
"exists": "false",
"message": "no references list found."
}

Example 2:
TEXT:
######
1. foppiano, l., lambard, g., amagasa, t. & ishii, m. mining
experimental data from materials science literature with large
language models: an evaluation study. sci. technol. adv. mater.
methods 4, 2356506 (2024).
######

Output:
{
"exists": "true"
}

Example 3:
TEXT:
######
cantor, j., beckman, r., collins, r. l., ghosh -dastidar, m., richardson, a. s., & dubowitz, t. 
(2020). snap participants improved food security and diet after a full -service supermarket 
opened in an urban food desert. health affairs , 39(8), 1386–1394.  
https://doi.org/10.1377/hlthaff.2019.01309   
coleman -jensen, a., rabbitt, m. p., gregory, c. a., & singh, a. (2020). household food security in 
the united states in 2019  (economic research report no. 275 ). united states department 
of agriculture. https://www.ers.usda.gov/webdocs/publications/99282/err -
275.pdf?v=744.4   
dubowitz, t., ghosh -dastidar , m., eibner, c., slaughter, m. e., fernandes, m., whitsel, e. a., bird, 
c. e., jewell, a., margolis, k. l., li, w., michael, y. l., shih, r. a., manson, j. e., & escarce, j. 
j. (2012). the women’s health initiative: the food environment, neighborhood 
socio economic status, bmi, and blood pressure. obesity , 20(4), 862–871.  
https://doi.org/10.1038/oby.2011.141   
######

Output:
{
"exists": "true"
}

Your turn:
TEXT:
######
""" + subsection + """
######

Output:
"""
            }
        ],
        temperature=0,
        top_p=0.9
    )

    log_usage(reply.usage.total_tokens, reply.usage.prompt_tokens,
              reply.usage.completion_tokens, "classify_reference_list_section", reply.model)

    return split_reply(model_reply=reply.choices[0].message.content, identifier='json')


async def async_classify_reference_list_section(subsection: str, client, MODEL: str) -> str:
    reply = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """"You are a strict JSON-only classification engine.
Your task is to analyze the provided text and determine whether it is part of a reference list or bibliography section.

Rules:
- You must always return valid JSON.
- Do not include any text outside of the JSON object.
- Do not include any explanation of the output.
- If a reference list exists, return:
  - "text_id": "<text id provided>"
  - "exists": "true"
- If no reference list exists, return:
  - "text_id": "<text id provided>"
  - "exists": "false"
  - "message": a brief explanation describing why there there is no previous sentence.
    """
            },
            {
                "role": "user",
                "content": """Analyze the following text and determine whether it contains a reference list.
Return only JSON.

Example 1:
TEXT:
######
TEXT_ID_92::: [we want to thank the sutd-zju idea visiting professor grant (sutd-zju (vp) 202103 and the sutd-zju thematic research grant (sutd-zju (tr) 202204) for supporting this work. this work is supported under deeps-ai (design,education, engineering, psychology, and science with ai).\nreferences\n]
######

Output:
{
"text_id": "TEXT_ID_92",
"exists": "false",
"message": "no references list found."
}

Example 2:
TEXT:
######
TEXT_ID_78::: [1. foppiano, l., lambard, g., amagasa, t. & ishii, m. mining
experimental data from materials science literature with large
language models: an evaluation study. sci. technol. adv. mater.
methods 4, 2356506 (2024).]
######

Output:
{
"text_id": "TEXT_ID_78",
"exists": "true"
}

Example 3:
TEXT:
######
TEXT_ID_12::: [cantor, j., beckman, r., collins, r. l., ghosh -dastidar, m., richardson, a. s., & dubowitz, t. 
(2020). snap participants improved food security and diet after a full -service supermarket 
opened in an urban food desert. health affairs , 39(8), 1386–1394.  
https://doi.org/10.1377/hlthaff.2019.01309   
coleman -jensen, a., rabbitt, m. p., gregory, c. a., & singh, a. (2020). household food security in 
the united states in 2019  (economic research report no. 275 ). united states department 
of agriculture. https://www.ers.usda.gov/webdocs/publications/99282/err -
275.pdf?v=744.4   
dubowitz, t., ghosh -dastidar , m., eibner, c., slaughter, m. e., fernandes, m., whitsel, e. a., bird, 
c. e., jewell, a., margolis, k. l., li, w., michael, y. l., shih, r. a., manson, j. e., & escarce, j. 
j. (2012). the women’s health initiative: the food environment, neighborhood 
socio economic status, bmi, and blood pressure. obesity , 20(4), 862–871.  
https://doi.org/10.1038/oby.2011.141]
######

Output:
{
"text_id": "TEXT_ID_12",
"exists": "true"
}

Your turn:
TEXT:
######
""" + subsection + """
######

Output:
"""
            }
        ],
        temperature=0,
        top_p=0.9
    )

    log_usage(reply.usage.total_tokens, reply.usage.prompt_tokens,
              reply.usage.completion_tokens, "async_classify_reference_list_section", reply.model)

    return split_reply(model_reply=reply.choices[0].message.content, identifier='json')


def extract_fact(citation: str):
    client, MODEL = sync_client()
    reply = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are an information extraction engine that rewrites citation-bearing text into pure factual claims.

Your task for each item:
- Extract the core factual statement supported by the citation.
- Remove discourse, commentary, author attribution, and reporting phrases (e.g., “this is concerning”, “according to”, “X found that”, “the authors show that”).
- Preserve the original meaning, entities, numbers, conditions, and qualifiers.
- Do not add new information or reinterpret the claim.
- Do not include citation markers, reference numbers, or author names in the extracted fact.
- Remove author names/ citation markers (e.g. "zafer et al.", "(gorenshtein, 2018)", "[2]")

Output rules:
- Always output valid JSON only, with no extra text.
- Return a JSON array with one object per input item, in the same order.
- For each item, return:
  {
    "id": "<id of claim>",
    "status": "ok",
    "fact": "<pure factual claim>"
  }
- If an item does not contain a clear extractable factual claim, return:
  {
    "id": "<id of claim>,
    "status": "error",
    "error_code": "NO_CLEAR_FACT",
    "message": "No clear factual claim could be extracted from the input."
  }

Quality constraints:
- Each extracted fact must be strictly entailed by its corresponding input.
- Keep wording as close as possible to the original factual content.
- Preserve uncertainty or conditions if present (e.g., “may”, “in some cases”, “under X conditions”).
                    """
            },
            {
                "role": "user",
                "content": """Extract the pure factual claim from each of the following citation-bearing texts. Return JSON only.
Example:
TEXT:
######
ID_31:: [this is concerning, as linardon et 
al. [1] found that only 54% of citations generated by chatgpt, gpt-4o, were accurate.]
ID_2:: [i 
really love apples, 
don't you?]
ID_200:: [content-based strategies include lexical, syntactic,\ntopical, and writing style analysis 
[3].]
ID_71:: [this is similar to our main research question, but instead of specifically aiming to discover methods for verifying citations, foltynek et al. [2] aims to discover a wider range of methods for plagiarism detection.]
######
Output:
[
{
   "id": "ID_31",
   "status": "ok",
   "fact": "only 54% of citations generated by chatgpt, gpt-4o, were accurate."
},
{
   "id": "ID_2",
   "status": "error",
   "error_code": "NO_CLEAR_FACT",
   "message": "No clear factual claim could be extracted from the input."
},
{
   "id": "ID_200",
   "status": "ok",
   "fact": "content-based strategies include lexical, syntactic,\ntopical, and writing style analysis"
},
{
   "id": "ID_71",
   "status": "ok",
   "fact": "discovers a wider range of methods for plagiarism detection"
}
]


Your turn:
TEXT:
######
"""
                           + citation +
                           """
                           ######
                           Output:
                           """
            }
        ],
        temperature=0.3,
        top_p=0.9
    )

    log_usage(reply.usage.total_tokens, reply.usage.prompt_tokens,
              reply.usage.completion_tokens, "extract_fact", reply.model)

    return split_reply(model_reply=reply.choices[0].message.content, identifier='json')


def extract_reference_list(references: str) -> str:
    client, MODEL = sync_client()
    reply = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are a precision information extraction engine.

Your task is to extract ONLY the reference list from an academic paper's raw extracted text.

You must follow these rules strictly:
1. Extract ONLY the references section.
   - Ignore everything before the references.
   - Ignore appendices, acknowledgments, footnotes, tables, or author bios.
   - If multiple sections appear after references, stop extraction at the end of the reference list.
2. Detect reference style:
   - If references are numbered (e.g., [1], 1., 1), use the number as the JSON key (as a string).
   - If references are not numbered (APA style), use:
       "(FirstAuthorLastName, Year)"
     as the JSON key.
   - If more than one author:
       Use "(FirstAuthorLastName et al., Year)"
   - Extract the publication year from the reference entry.
   - If year is missing, use "(FirstAuthorLastName, UnknownYear)".
   - If the reference is for a website, add an "error" key to the JSON output.
3. Preserve the FULL reference text exactly as written.
   - Do NOT rewrite.
   - Do NOT correct formatting.
   - Do NOT remove DOIs or URLs.
   - Keep line breaks if present.
4. Output STRICT JSON.
   - No explanations.
   - No markdown.
   - No comments.
   - No trailing commas.
   - Ensure valid JSON.
"""
            },
            {
                "role": "user",
                "content": """Below is the full raw extracted text of an academic paper.

Your task: extract the reference list according to the system instructions and return STRICT JSON only.

Example 1:
Reference List:
######
brownell, c. a. (2016). prosocial behavior in infancy: the role of socialization. child development  
perspectives , 10(4), 222–227. https://doi.org/10.1111/cdep.12189  
brownell, c. a., svetlova, m., anderson, r., nichols, s. r., & drummond, j. (2013). socialization of 
early prosocial behavior: parents’ talk about emotions is associated with sharing and 
helping in toddlers. infancy, 18(1), 91–119. https://doi.org/10.1111/j.1532-
7078.2012.00125.x   
hammond, s. i., & brownell, c. a. (2018). happily unhelpful: infants' everyday helping and its 
connections to early prosocial development. frontiers in psychology , 9, article 1770. 
https://doi.org/10.3389/fpsyg.2018.01770
nelson, n. l., and russell, j. a. (2013). universality revisited. emotion review , 5(1), 8–15. 
https://doi.org/10.1177/1754073912457227
GCSE - England - BBC Bitesize. (n.d.). BBC Bitesize. https://www.bbc.co.uk/bitesize/levels/z98jmp3
######

Output:
[{
    "key": "(brownell, 2016)",
    "reference": "brownell, c. a. (2016). prosocial behavior in infancy: the role of socialization. child development  
perspectives , 10(4), 222–227. https://doi.org/10.1111/cdep.12189  "
},
{
    "key": "(brownell et al., 2016)",
    "reference": "brownell, c. a., svetlova, m., anderson, r., nichols, s. r., & drummond, j. (2013). socialization of 
early prosocial behavior: parents’ talk about emotions is associated with sharing and 
helping in toddlers. infancy, 18(1), 91–119. https://doi.org/10.1111/j.1532-
7078.2012.00125.x"
}
{
    "key": "(hammond and brownell, 2016)",
    "reference": "hammond, s. i., & brownell, c. a. (2018). happily unhelpful: infants' everyday helping and its 
connections to early prosocial development. frontiers in psychology , 9, article 1770. 
https://doi.org/10.3389/fpsyg.2018.01770"
}
{
    "key": "(nelson and russell, 2013)",
    "reference": "nelson, n. l., and russell, j. a. (2013). universality revisited. emotion review , 5(1), 8–15. 
https://doi.org/10.1177/1754073912457227"
}
{
    "key": "(BBC Bitesize, UnknownYear)",
    "reference": "GCSE - England - BBC Bitesize. (n.d.). BBC Bitesize. https://www.bbc.co.uk/bitesize/levels/z98jmp3",
    "error": "website"
}
]

Example 2:
Reference List:
######
[1] H. Zhang, H. Zhang, Y. Cui, Y. Zheng, and Q. Yang, “Fake News 
Detection with Deep Learning: A Survey,” IEEE Transactions on 
Neural Networks and Learning Systems, vol. 33, no. 11, pp. 6175– 
6196, Nov. 2022. doi: 10.1109/TNNLS.2022.3147482 
 
2. S. Khan, H. Cambria, and M. A. Tahir, “A Review on Sentiment and 
Emotion Analysis for Fake News Detection,” IEEE Intelligent 
Systems,   vol.   36,   no.   5,   pp.   8–17,   Sep.–Oct.   2021.   doi: 
10.1109/MIS.2021.3101655 

3 “GCSE - England - BBC Bitesize,” BBC Bitesize. https://www.bbc.co.uk/bitesize/levels/z98jmp3

4 R. Oshikawa, J. Qian, and W. Y. Wang, “A Survey on Natural 
Language Processing for Fake News Detection,” in Proceedings of 
ACL 2020, pp. 357–366, Jul. 2020. 
######

Output:
[{
    "key": "[1[",
    "reference": "H. Zhang, H. Zhang, Y. Cui, Y. Zheng, and Q. Yang, “Fake News 
Detection with Deep Learning: A Survey,” IEEE Transactions on 
Neural Networks and Learning Systems, vol. 33, no. 11, pp. 6175– 
6196, Nov. 2022. doi: 10.1109/TNNLS.2022.3147482"
},
{
    "key": "[2]",
    "reference": "S. Khan, H. Cambria, and M. A. Tahir, “A Review on Sentiment and 
Emotion Analysis for Fake News Detection,” IEEE Intelligent 
Systems,   vol.   36,   no.   5,   pp.   8–17,   Sep.–Oct.   2021.   doi: 
10.1109/MIS.2021.3101655 "
},
{
    "key": "[3]",
    "reference": "“GCSE - England - BBC Bitesize,” BBC Bitesize. https://www.bbc.co.uk/bitesize/levels/z98jmp3",
    "error": "website
},
{
    "key": "[4]",
    "reference": "R. Oshikawa, J. Qian, and W. Y. Wang, “A Survey on Natural 
Language Processing for Fake News Detection,” in Proceedings of 
ACL 2020, pp. 357–366, Jul. 2020. "
}
]

Reference List:
######
""" + references + """
######

Output:
"""
            }
        ],
        temperature=0,
        top_p=0.9
    )

    log_usage(reply.usage.total_tokens, reply.usage.prompt_tokens,
              reply.usage.completion_tokens, "extract_reference_list", reply.model)

    return split_reply(model_reply=reply.choices[0].message.content, identifier='json')


async def validate_in_text_citations(citation: str, reference: str, example_in_text_citations: str, client, MODEL: str):
    reply = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are an expert academic editor and citation standardization specialist. 
Your task is to analyze a provided list of in-text citations from a research paper, determine the dominant citation style, and reformat a specific target in-text citation to match that style consistently.

Follow these strict guidelines:
1. STYLE INFERENCE: Analyze the provided list of all in-text citations to identify the predominant citation format (e.g., APA/Harvard (Author, Year), Vancouver/IEEE [Number], Chicago Author-Date, etc.). Look for consistent patterns and prioritize the format used by the majority of citations.
2. CITATION GENERATION: Use the target citation's associated full reference to extract necessary metadata (author surnames, publication year, etc.). Apply the inferred style's formatting rules precisely (e.g., correct use of "&" vs "and", "et al." thresholds, numeric bracket placement, comma/spacing conventions).
3. NUMERIC STYLES: If the dominant style is numeric (e.g., [1], (1)), assign the correct number based on the reference's position in the reference list or its existing in-text citation number in the paper. Preserve bracket type and spacing consistently.
4. DATA INTEGRITY: Never modify the original reference text. Return it exactly as provided.
5. FALLBACK: If the style is highly mixed or cannot be confidently determined, default to APA 7th edition parenthetical format: (AuthorLastName, Year) or (AuthorLastName et al., Year) for 3+ authors.
6. OUTPUT CONSTRAINTS: Return ONLY a valid, parseable JSON object with exactly three keys: "reference", "previous_key", and "new_key". Do NOT include markdown formatting, explanations, or any additional text. Properly escape quotes, newlines, and special characters within strings.
7. OUTPUT FORMAT:
Return this EXACT valid JSON format:
{
 "reference": "...",
 "previous_key": "...",
 "new_key": "..."
}
"""
            },
            {
                "role": "user",
                "content": """Please analyze the following data and return the standardized citation in the specified JSON format.

TARGET IN-TEXT CITATION (to be corrected):
######
""" + citation + """
######

ASSOCIATED REFERENCE (from the reference list):
######
""" + reference + """
######

ALL IN-TEXT CITATIONS FOUND IN THE PAPER (for style inference):
######
""" + example_in_text_citations + """
######

Instructions:
- Examine the "ALL IN-TEXT CITATIONS FOUND IN THE PAPER" list to determine the dominant citation style.
- Reformat the TARGET IN-TEXT CITATION to match that style, using the ASSOCIATED REFERENCE to extract required metadata.
- Set "previous_key" to the exact original TARGET IN-TEXT CITATION.
- Set "new_key" to the correctly formatted citation.
- Set "reference" to the exact ASSOCIATED REFERENCE provided above.
- Output ONLY the JSON object. No extra text.

TARGET IN-TEXT CITATION (to be corrected):
######
""" + citation + """
######

ASSOCIATED REFERENCE (from the reference list):
######
""" + reference + """
######

Return STRICT JSON only.
Output:
"""
            }
        ],
        temperature=0.1
    )

    log_usage(reply.usage.total_tokens, reply.usage.prompt_tokens,
              reply.usage.completion_tokens, "validate_in_text_citation", reply.model)

    return split_reply(model_reply=reply.choices[0].message.content, identifier='json')


async def find_citations_in_sentences(sentences: str, client, MODEL: str):
    # client, MODEL = async_client()
    reply = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """"You are an information extraction system.

Task:
From the provided list of sentences, extract ONLY the sentences that contain citations.

Citations may appear in formats such as:
- Numeric style: [1], [2], [3–5]
- Parenthetical style: (Smith, 2020), (Smith & Lee, 2019), (Smith et al., 2018), (Smith and Lee, 2000)

For each sentence containing a citation:

1. Extract the original sentence exactly as written.
2. Extract each citation string exactly as written.
3. Do NOT rewrite, summarize, or paraphrase any content.
4. If multiple citations appear in one sentence, return the sentence seprately with the different citations.

Output format:
Return a JSON array only. Do not include explanations.

Each item must follow this structure:

{
  "sentence_id": "<original sentence ID>",
  "original_sentence": "<full original sentence>",
  "citation": "<citation string exactly as written>",
}

Important rules:
- If a sentence contains no citation, do not include it.
- Do not invent citations.
- Do not modify punctuation.
- Do not combine sentences.
- Extract text exactly as they appear.
- If a sentence contains multiple citations, return multiple objects.
- Output valid JSON only."""
            },
            {
                "role": "user",
                "content": """Below is a list of sentences. Each sentence has a unique sentence_id in square brackets.

Extract citation information according to the system instructions.

Example:
Sentences:
######
ID_51::: [Deep learning has significantly improved image recognition accuracy (Krizhevsky et al., 2012, 2025).]
ID_210::: [Transformer models outperform recurrent architectures in many NLP tasks [3].]
ID_129::: [This approach requires substantial computational resources.]
ID_5::: [Several studies (Smith & Lee, 2019; Johnson, 2021) highlight the importance of data augmentation.]
ID_819::: [According to (Brown et al., 2020), scaling laws predict performance improvements.]
ID_472::: [The algorithm achieves state-of-the-art results [4–6] in benchmark datasets.]
######

Output:
[
  {
    "sentence_id": "ID_51",
    "original_sentence": "Deep learning has significantly improved image recognition accuracy (Krizhevsky et al., 2012, 2025).",
    "citation": "(Krizhevsky et al., 2012)",
  },
  {
    "sentence_id": "ID_51",
    "original_sentence": "Deep learning has significantly improved image recognition accuracy (Krizhevsky et al., 2012, 2025).",
    "citation": "(Krizhevsky et al., 2025)",
  },
  {
    "sentence_id": "ID_210",
    "original_sentence": "Transformer models outperform recurrent architectures in many NLP tasks [3].",
    "citation": "[3]",
  },
  {
    "sentence_id": "ID_5",
    "original_sentence": "Several studies (Smith & Lee, 2019; Johnson, 2021) highlight the importance of data augmentation.",
    "citation": "(Smith & Lee, 2019)",
  },
  {
    "sentence_id": "ID_5",
    "original_sentence": "Several studies (Smith & Lee, 2019; Johnson, 2021) highlight the importance of data augmentation.",
    "citation": "(Johnson, 2021)",
  },
  {
    "sentence_id": "ID_819",
    "original_sentence": "According to (Brown et al., 2020), scaling laws predict performance improvements.",
    "citation": "(Brown et al., 2020)",
  },
  {
    "sentence_id": "ID_472",
    "original_sentence": "The algorithm achieves state-of-the-art results [4–6] in benchmark datasets.",
    "citation": "[4]",
  }
  {
    "sentence_id": "ID_472",
    "original_sentence": "The algorithm achieves state-of-the-art results [4–6] in benchmark datasets.",
    "citation": "[5]",
  }
  {
    "sentence_id": "ID_472",
    "original_sentence": "The algorithm achieves state-of-the-art results [4–6] in benchmark datasets.",
    "citation": "[6]",
  }
]

Sentences:
######
""" + sentences + """
######

Output:
"""
            }
        ],
        temperature=0.1,
        top_p=0.9
    )

    log_usage(reply.usage.total_tokens, reply.usage.prompt_tokens,
              reply.usage.completion_tokens, "find_citations_in_sentences", reply.model)

    return split_reply(model_reply=reply.choices[0].message.content, identifier='json')


def match_references(unmatched: str, known_set: str):
    client, MODEL = sync_client()
    reply = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """"You are a reference-matching assistant.

Your task is to match a set of unmatched in-text references to the most likely item from a known set of in-text references.

Rules:
- Use semantic similarity (author names, year, citation style).
- Use the provided sentence context of the unmatched reference.
- Match each unmatched reference to at most one known reference.
- If no confident match exists, return "NO_MATCH".
- Do NOT generate new references.
- Do NOT modify IDs.
- Only match against the provided known set.
- Output strictly in valid JSON.

Matching Guidelines
- Treat minor formatting differences as equivalent ( (e.g., "Smith et al., 2020") vs ("Smith et al., 2020") ).
- Allow slight year mismatches if clearly referring to the same work.
- Use sentence context to disambiguate similar author-year combinations.
- If two known references are plausible, choose the most specific topical match.

Required Output Format:
[
  {
    "unmatched_id": "<string>",
    "matched_known_id": "<string or NO_MATCH>",
  }
]"""
            },
            {
                "role": "user",
                "content": """Match the following unmatched in-text references to the known reference set.

Example:
Known References:
######
KNOWN_ID_0:: [(reschke et al., 2017)]
KNOWN_ID_1:: [(rheingold, 1982)]
KNOWN_ID_5:: [(rogoff, 2003)]
KNOWN_ID_10:: [(rogoff, 2014)]
KNOWN_ID_392:: [(ekman, 1992)]
KNOWN_ID_58:: [(russell, 2015)]
KNOWN_ID_21:: [(shanon, 2008)]
KNOWN_ID_87:: [(silvia, 2006)]
######

Unmatched:
######
UNMATCHED_REFERENCE_ID_0::[('Reschke, Walle, and  Dukes', ' 2017)')]::SENTENCE::[a critical examination of the  role of emotions in prosocial behavior is timely given that, on the basis of recent studies,  researchers have claimed that helping others is itself a type of practical emotion understanding  emotions and infants' prosocial behavior            4  (e.g., reschke, walle, & dukes, 2017).]
UNMATCHED_REFERENCE_ID_4::[('', 'ekman (1992)')]::SENTENCE::[ekman (1992)  expressed doubts over interest being an emotion, and only a small number of contemporary  researchers study the emotion of interest (e.g., camras et al., 2002; clément & dukes, 2013; b. s.  izard & izard, 1977; c. e. izard, 2007, 2010; silvia, 2006; see also panskepp, 1998, on the emotion  of seeking), although historically pragmatic theorists like dewey and mead (ward & throop, 1989)  and developmentalist piaget regarded interest as an important human emotion (see sokol &  hammond, 2009).]
UNMATCHED_REFERENCE_ID_92::[('(',  '18')]::SENTENCE::[there are a total of ten apples (10).]
######

Output:
[
{
    "unmatched_id": "UNMATCHED_REFERENCE_ID_0",
    "matched_known_id": "KNOWN_ID_0"
},
{
    "unmatched_id": "UNMATCHED_REFERENCE_ID_4",
    "matched_known_id": "KNOWN_ID_392",
},
{
    "unmatched_id": "UNMATCHED_REFERENCE_ID_92",
    "matched_known_id": "NO_MATCH",
}
]

Known References:
######
""" + known_set + """
######

Unmatched:
######
""" + unmatched + """
######

Output:
"""
            }
        ],
        temperature=0,
        top_p=0.9
    )

    log_usage(reply.usage.total_tokens, reply.usage.prompt_tokens,
              reply.usage.completion_tokens, "match_references", reply.model)

    return split_reply(model_reply=reply.choices[0].message.content, identifier='json')


def extract_reference_metadata(references: str):
    client, MODEL = sync_client()
    reply = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """"You are an information extraction assistant.

Your task is to extract structured bibliographic metadata from academic references.

For each reference provided:
- Extract the following fields:
  - id
  - title
  - year
  - first_author
  - co_authors
  - doi

Definitions:
- id: The unique identifier provided before each reference.
- title: The full title of the work exactly as written.
- year: The 4-digit publication year.
- first_author: The first listed author exactly as written.
- co_authors: A list of all remaining authors in order, exactly as written. If there are no co-authors, return an empty list.
- doi: digital identifier, typically in url form.

Rules:
- Preserve punctuation in titles.
- Preserve author formatting exactly as written (including initials and spacing).
- Do not infer or hallucinate missing data.
- If a field cannot be found, return null for that field.
- Output must be valid JSON.
- Output must be a JSON array.
- Do not include explanations or extra text.
"""
            },
            {
                "role": "user",
                "content": """Extract the required fields from the following references.

Example:
References:
######
REFERENCE_ID_0:: [S. Yao, D. Yu, J. Zhao, I. Shafran, T. L. Griffiths, Y. Cao, and K. Narasimhan (2023)
Tree of thoughts: deliberate problem solving with large language models.]
REFERENCE_ID_872:: [Camras, L., Meng, Z., Ujiie, T., Dharamsi, S., Miyake, K., Oster, H., Wang, L., Cruz, J., Murdoch, A., & 
Campos, J. (2002). Observing emotion in infants: Facial expression, body behavior, and 
rater judgments of responses to an expectancy-violating event. Emotion, 2(2), 179–193. 
https://doi.org/10.1037/1528-3542.2.2.179  ]
######

Output:
[
  {
    "id": "REFERENCE_ID_0",
    "title": "Tree of thoughts: deliberate problem solving with large language models.",
    "year": "2023",
    "first_author": "S. Yao",
    "co_authors": [
      "D. Yu",
      "J. Zhao",
      "I. Shafran",
      "T. L. Griffiths",
      "Y. Cao",
      "K. Narasimhan"
    ]
    "doi": "none"
  },
  {
    "id": "REFERENCE_ID_872",
    "title": "Observing emotion in infants: Facial expression, body behavior, and rater judgments of responses to an expectancy-violating event",
    "year": "2002",
    "first_author": "Camras, L.",
    "co_authors": [
      "Meng, Z.",
      "Ujiie, T.",
      "Dharamsi, S.",
      "Miyake, K.",
      "Oster, H.",
      "Wang, L.",
      "Cruz, J.",
      "Murdoch, A.",
      "Campos, J."
    ]
    "doi": "https://doi.org/10.1037/1528-3542.2.2.179"
  }
]

Your turn:
References:
######
""" + references + """
######

Output:
"""
            }
        ],
        temperature=0,
        top_p=0.9
    )

    log_usage(reply.usage.total_tokens, reply.usage.prompt_tokens,
              reply.usage.completion_tokens, "extract_reference_metadata", reply.model)

    return split_reply(model_reply=reply.choices[0].message.content, identifier='json')


async def extract_title_from_provided_pdf(references: str, reference_text: str, client, MODEL: str):
    reply = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """"You are an information extraction system.

Your task is to:
1. Read an input academic paper.
2. Determine the title of the input paper.
3. Compare the title to a provided list of references.
4. Identify the reference that corresponds to the same paper.

Important rules:
- The title may appear at the beginning of the document but could also appear elsewhere.
- Ignore author names, affiliations, page headers/footers, and formatting artifacts.
- Titles may differ slightly due to punctuation, capitalization, abbreviations, or minor wording differences.
- Match based on semantic similarity of the title.
- If no match exists, return None.
- Never provide any explanation.

Reference format:
REF_ID_X::: [reference text]

Paper format:
PAPER_ID_X
Filename: <filename>

Return the REF_ID_ that best matches the paper title.

If no reference matches the paper, return None.

Output format (JSON only):
{
  "matched_reference_id": "<REF_ID_X or null>",
  "paper_id": "<REF_ID_X or null>"
}

Output format if no match (JSON only):
{
  "matched_reference_id": "None",
  "paper_id": "<REF_ID_X or null>"
}

Strict rules:
- Output ONLY valid JSON.
- Do not include explanations.
- Do not include reasoning.
- Do not include confidence scores.
- Do not include extra fields.
"""
            },
            {
                "role": "user",
                "content": """Match the input paper to one reference.

Reference List:
######
""" + references + """
######

PDF Paper text:
######
""" + reference_text + """
######

Output:
"""
            }
        ],
        temperature=0,
        top_p=0.9
    )

    log_usage(reply.usage.total_tokens, reply.usage.prompt_tokens,
              reply.usage.completion_tokens, "extract_title_from_provided_pdf", reply.model)

    return split_reply(model_reply=reply.choices[0].message.content, identifier='json')


async def extract_referenced_paragraphs(paragraphs: str, client, MODEL: str):
    """
    Extract paragraphs where one citation refers to the whole paragraph.
    :param paragraphs: Paragraphs with ids in string format.
    :param client: async client.
    :param MODEL: Model to use.
    :return: String model reply.
    """
    reply = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """"You are an information extraction system.

Your task is to detect when a single in-text citation refers to an entire paragraph.

A citation refers to an entire paragraph when:
- There is exactly ONE citation covering the entire paragraph content.
- The citation appears at the end of the paragraph or clearly refers to the paragraph as a whole.
- The paragraph presents information that appears attributed to that single source.

Examples of citation formats include:
- Numeric citations: [1], [12]
- Parenthetical citations: (Smith, 2020), (Smith & Jones, 2021)
- Narrative references followed by a parenthetical citation

DO NOT return anything if:
- The paragraph has no citation.
- The paragraph has multiple citations.
- The citation refers only to a specific sentence or claim rather than the entire paragraph.

Output requirements:
- Output MUST be valid JSON.
- Output MUST follow this schema:

[
    {
      "paragraph_id": "PARAGRAPH_ID_<integer>",
      "citation": "<exact citation text>"
    }
]


Rules:
- If no paragraphs match, don't return anything.
- Extract the citation exactly as written in the paragraph.
- Do not modify citation text.
- Do not include explanations or extra text.
- Only output JSON."""
            },
            {
                "role": "user",
                "content": """Analyze the following paragraphs and identify paragraphs where a single citation refers to the entire paragraph.

Each paragraph has a unique paragraph ID.

Return matches only.

EXAMPLE:
Paragraphs:
######
PARAGRAPH_ID_0::: [Furthermore, Jowore and Turpin [4] analysed 25 research
articles in a review aimed at finding methods to identify ”fake
news” in social media. They construct a guideline on how to
manually identify fake news on social media, that is: check
the source of the information, beware of the presence of bots
and mimics, and take into account biases of sources.]
PARAGRAPH_ID_72::: [Bodaghi et al. (2000), Zafer et al. (1992), and Jowore and Turpin
(2024) all investigate misinformation detection systems. This
aligns with our main research goal, as it aims to combat
misinformation specifically in research papers.]
PARAGRAPH_ID_302::: [Zafer et al. (2020 also conduct a systematic literature review
on 76 studies to discover methods for the detection of misin
formation in social media. They describe that several Machine
Learning (ML) and Deep Learning (DL) methods have been
utilised to classify social media claims as misinformation or
not [5]. Furthermore, they mention that transformers have also
been utilised in this area, namely BERT.]
######

Output:
[
  {
    "paragraph_id": "PARAGRAPH_ID_0",
    "citation": "[4]"
  },
  {
    "paragraph_id": "PARAGRAPH_ID_302",
    "citation": "Zafer et al. (2020)"
  }
]

Your turn:
Paragraphs:
######
""" + paragraphs + """
######

Output:
"""
            }
        ],
        temperature=0.1,
        top_p=0.9
    )

    log_usage(reply.usage.total_tokens, reply.usage.prompt_tokens,
              reply.usage.completion_tokens, "extract_referenced_paragraphs", reply.model)

    return split_reply(model_reply=reply.choices[0].message.content, identifier='json')


async def naive_RAG_justification(context: str, fact: str, initial_verdict:str, client, MODEL, prompt_name):
    reply = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are a factual verification assistant.

Your task is to generate a justification, using the provided context, for a given verdict for a given claim. 

You must strictly follow these rules:

1. Base your judgment ONLY on the provided CONTEXT and EVALUATION RUBRIC. Do not use prior knowledge.
2. Classification Explanation:
   - "TRUE": The context supports the claim.
   - "FALSE": The context contradicts the claim.
   - "INCONCLUSIVE": The context is insufficient, unclear, or only partially related to the claim.
3. Justify your classification in at least three sentences. 
   - Justification must NOT reference the given context/chunks/evidence or IDs- specify what you mean/ what you are referring to.
   - Never make a table in your justification.
4. Do NOT include explanations.
5. Do NOT use your own knowledge to classify the claim or justify your claim.

6. Output ONLY VALID JSON in the following format:
{
    "justification": "write your justification here!"
}"""
        },
        {
            "role": "user",
            "content": """Evaluate whether the CLAIM is supported by the CONTEXT.

Example:
CONTEXT:
######
CHUNK_ID_82::: [Apples are fruit.]
CHUNK_ID_21::: [Apples are fat free and full of fiber.]
CHUNK_ID_51::: [Apples generally have a low to moderate GI score, so eating apples may help with managing blood sugar levels.]
CHUNK_ID_96::: [Fruit are healthy.]
######
CLAIM:
######
FACT_ID_271::: [Apples are healthy for humans.]
######
INITIAL VERDICT: TRUE
######
Return only the JSON result.
Output:
{
    "justification": "Apples are fruit, which is considered healthy. Moreover, apples may help manage blood sugar lebels due to their low GI score."
}

Your turn:
CONTEXT:
######
""" + context + """
######
CLAIM:
######
""" + fact + """
######
INITIAL VERDICT: """ + initial_verdict + """
######
Return only the JSON result.
Output:"""
            }
        ],
        temperature=0.5,
        top_p=0.5,
    )

    log_usage(reply.usage.total_tokens, reply.usage.prompt_tokens,
              reply.usage.completion_tokens, prompt_name, reply.model)

    return split_reply(model_reply=reply.choices[0].message.content, identifier='json')

async def naive_RAG_check(context: str, fact: str, client, MODEL, prompt_name):
    reply = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are a factual verification assistant.

Your task is to determine whether a given CLAIM is supported by the provided CONTEXT.

You must strictly follow these rules:

1. Base your judgment ONLY on the provided CONTEXT. Do not use prior knowledge.
2. Compare the CLAIM against the CONTEXT carefully.
3. Classify the result into one of three categories:
   - "TRUE": The context supports the claim.
   - "FALSE": The context contradicts the claim.
   - "INCONCLUSIVE": The context is insufficient, unclear, or only partially related to the claim.
4. Justify your classification in at least three sentences. 
   - Justification must not reference the given context/chunks/evidence - specify what you mean/ what you are referring to.
   - Never make a table in your justification.
5. Classification rules:
   - Return "TRUE" if the claim is supported in meaning, even if phrased differently or approximately.
   - Approximate numbers (e.g., "about", "approximately") should be considered correct if they are reasonably close to the context values.
   - Minor differences in wording or rounding do NOT invalidate support.
   - Return "FALSE" only if the claim contradicts the context.
   - Return "INCONCLUSIVE" only if the context truly lacks enough information.
   - For numerical claims with "approximately", differences within ~10% should be considered supported unless the exact value is critical.

6. Do NOT include explanations.
7. Do NOT use your own knowledge to classify the claim or justify your claim.

7. Output ONLY VALID JSON in the following format (replace X and Y with the input integers):
{
    "input_id": "PAPER_ID_X_FACT_ID_Y",
    "output": "TRUE" | "FALSE" | "INCONCLUSIVE",
    "justification": "write your justification here!"
}"""
            },
            {
                "role": "user",
                "content": """Evaluate whether the CLAIM is supported by the CONTEXT.

Example:
CONTEXT:
######
CHUNK_ID_82::: [Apples are fruit.]
CHUNK_ID_21::: [Apples are fat free and full of fiber.]
CHUNK_ID_51::: [Apples generally have a low to moderate GI score, so eating apples may help with managing blood sugar levels.]
CHUNK_ID_96::: [Fruit are healthy.]
######
CLAIM:
######
FACT_ID_271::: [Apples are healthy for humans.]
######
Evaluate whether the CLAIM is supported by the CONTEXT.
Return only the JSON result.
Output:
{
    "input_id": "FACT_ID_271",
    "output": "TRUE",
    "justification": "Apples are fruit, which is considered healthy. Moreover, apples may help manage blood sugar lebels due to their low GI score."
}

Example:
CONTEXT:
######
CHUNK_ID_82::: [Grass is green.]
CHUNK_ID_21::: [Grass is a plant.]
CHUNK_ID_96::: [Carnivores eat other animals.]
CHUNK_ID_96::: [Animals are not plants.]
######
CLAIM:
######
FACT_ID_102::: [Carnivores eat grass.]
######
Evaluate whether the CLAIM is supported by the CONTEXT.
Return only the JSON result.
Output:
{
    "input_id": "FACT_ID_102",
    "output": "FALSE",
    "justification": "Carnivores eat other animals, which aren't plants."
}

Your turn:
CONTEXT:
######
""" + context + """
######
CLAIM:
######
""" + fact + """
######

Evaluate whether the CLAIM is supported by the CONTEXT.
Return only the JSON result.
Output:
"""
            }
        ],
        temperature=0.35,
        top_p=0.7,

    )

    log_usage(reply.usage.total_tokens, reply.usage.prompt_tokens,
              reply.usage.completion_tokens, prompt_name, reply.model)

    return split_reply(model_reply=reply.choices[0].message.content, identifier='json')


async def get_quotes_from_chunks(chunks: str, fact: str, client, MODEL):
    reply = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are a factual verification assistant.

Your task is to determine whether a given CLAIM is supported by the provided CONTEXT.

You must strictly follow these rules:

1. Base your judgment ONLY on the provided CONTEXT. Do not use prior knowledge.
2. Compare the CLAIM against the CONTEXT carefully.
3. Classify the result into one of three categories:
   - "TRUE": The context clearly supports the claim.
   - "FALSE": The context clearly contradicts the claim.
   - "INCONCLUSIVE": The context is insufficient, unclear, or only partially related to the claim.

4. Be conservative:
   - If evidence is weak, incomplete, or ambiguous → return "INCONCLUSIVE".
   - Only return "TRUE" if the claim is directly and unambiguously supported.
   - Only return "FALSE" if the claim is directly contradicted.

5. Do NOT include explanations.

6. Output ONLY valid JSON in the following format (replace X and Y with the input integers):
{
    "input_id": "PAPER_ID_X_FACT_ID_Y",
    "output": "TRUE" | "FALSE" | "INCONCLUSIVE"
}"""
            },
            {
                "role": "user",
                "content": """Evaluate whether the CLAIM is supported by the CONTEXT.

Example:
CONTEXT:
######
CHUNK_ID_82::: [Apples are fruit.]
CHUNK_ID_21::: [Apples are fat free and full of fiber.]
CHUNK_ID_51::: [Apples generally have a low to moderate GI score, so eating apples may help with managing blood sugar levels.]
CHUNK_ID_96::: [Fruit are healthy.]
######
CLAIM:
######
FACT_ID_271::: [Apples are healthy for humans.]
######
Evaluate whether the CLAIM is supported by the CONTEXT.
Return only the JSON result.
Output:
{
    "input_id": "FACT_ID_271",
    "output": "TRUE"
}

Example:
CONTEXT:
######
CHUNK_ID_82::: [Grass is green.]
CHUNK_ID_21::: [Grass is a plant.]
CHUNK_ID_96::: [Carnivores eat other animals..]
######
CLAIM:
######
FACT_ID_102::: [Carnivores eat grass.]
######
Evaluate whether the CLAIM is supported by the CONTEXT.
Return only the JSON result.
Output:
{
    "input_id": "FACT_ID_102",
    "output": "FALSE"
}

Your turn:
CONTEXT:
######
""" + context + """
######
CLAIM:
######
""" + fact + """
######

Evaluate whether the CLAIM is supported by the CONTEXT.
Return only the JSON result.
Output:
"""
            }
        ],
        temperature=0.2,
        top_p=0.9
    )

    log_usage(reply.usage.total_tokens, reply.usage.prompt_tokens,
              reply.usage.completion_tokens, "get_quotes_from_chunks", reply.model)

    return split_reply(model_reply=reply.choices[0].message.content, identifier='json')


async def evaluate_chunk(chunk: str, fact: str, client, MODEL, prompt_name:str):
    reply = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are an expert research evaluator assisting a retrieval-augmented generation (RAG) system.

Your task is to evaluate a single retrieved text chunk against a user’s factual claim.

You must score the chunk on FOUR criteria, each from 1 to 5:

1. Relevance: How directly the chunk relates to the claim.
3. Clarity: Whether the information is coherent and makes sense.
4. Applicability: Whether the information can be directly used to verify or refute the claim.
5. Quality: Overall credibility, and reliability of the information.

Scoring rules:
- 1 = very poor
- 2 = weak
- 3 = moderate
- 4 = strong
- 5 = excellent

You must:
- Be critical and avoid inflating scores.
- Prefer precise, evidence-based chunks over vague or generic ones.
- Penalize speculative or low-credibility content.
- Treat missing information as a weakness.
- Do NOT include any reasoning or explanation.
- Return ONLY JSON format.

Output format (STRICT JSON only, no extra text):
{
  "relevance": <int>,
  "clarity": <int>,
  "applicability": <int>,
  "quality": <int>
}
"""
            },
            {
                "role": "user",
                "content": """Evaluate the following retrieved chunk in the context of the claim.

Instructions:
- Score the chunk using the four criteria.
- Be strict and analytical.
- If the chunk lacks information needed to judge a category, assign a lower score.

Example:
Chunk:
######
An apple is the round, edible fruit of an apple tree (Malus spp.). Fruit trees of the orchard or domestic apple (Malus domestica), the most widely grown in the genus, are cultivated worldwide. The tree originated in Central Asia, where its wild ancestor, Malus sieversii, is still found. Apples have been grown for thousands of years in Eurasia before they were introduced to North America by European colonists. Apples have cultural significance in many mythologies (including Norse and Greek) and religions (such as Christianity in Europe).
######

Claim:
######
Apples are consumed all around the world, and have appeared in multiple different mythologies.
######

Output:
{
  "relevance": 5,
  "clarity": 4,
  "applicability": 4,
  "quality": 5
}

Your turn:
Chunk:
######
""" + chunk + """
######

Claim:
######
""" + fact + """
######

Output:
"""
            }
        ],
        temperature=0,
        top_p=0.9
    )

    log_usage(reply.usage.total_tokens, reply.usage.prompt_tokens,
              reply.usage.completion_tokens, prompt_name, reply.model)

    return split_reply(model_reply=reply.choices[0].message.content, identifier='json')


async def get_quotes_from_chunks(chunks: str, fact: str, client, MODEL):
    reply = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are a careful fact-verification assistant.

Your task is to extract verbatim quotes from provided text chunks that either SUPPORT or CONTRADICT a given fact.

Rules:
1. Only use exact text spans from the chunks. Do NOT paraphrase.
2. Quotes must be directly relevant to the fact.
3. Keep quotes concise (prefer 1–3 sentences or less).
4. Each quote must clearly support OR contradict the fact.
5. If a quote is ambiguous or only loosely related, ignore it.
6. Do NOT use outside knowledge. Only use the provided chunks.
7. Avoid duplicates or near-duplicates.
8. A single chunk may yield multiple quotes if they express distinct evidence.
9. Do NOT include any explanation or reasoning.

Output format (strict JSON):
{
  "fact_id": "FACT_ID_X",
  "supports": [
        "quote 1", "another quote"
  ],
  "contradicts": [
    "counter quote", "another counter quote"
  ]
}

If no quotes are found for a category, return an empty list.

Do not include any explanation or text outside the JSON."""
            },
            {
                "role": "user",
                "content": """Extract verbatim quotes from the text chunks that support or contradict the fact.

Example:
Text Chunks:
######
CHUNK_ID_819::: [An apple is the round, edible fruit of an apple tree (Malus spp.). Fruit trees of the orchard or domestic apple (Malus domestica), the most widely grown in the genus, are cultivated worldwide. The tree originated in Central Asia, where its wild ancestor, Malus sieversii, is still found. Apples have been grown for thousands of years in Eurasia before they were introduced to North America by European colonists. Apples have cultural significance in many mythologies (including Norse and Greek) and religions (such as Christianity in Europe).]
CHUNK_ID_70::: [There are more than 7,500 cultivars of apples. Different cultivars are bred for various tastes and uses, including cooking, eating raw, and cider or apple juice production. Trees and fruit are prone to fungal, bacterial, and pest problems, which can be controlled by a number of organic and non-organic means. In 2010, the fruit's genome was sequenced as part of research on disease control and selective breeding in apple production.]
######

Fact:
######
FACT_ID_94::: [There are many apples from different regions that can be consumed.]
######

Output:
{
  "fact_id": "FACT_ID_94",
  "supports": [
        "An apple is the round, edible fruit of an apple tree", "Apples have been grown for thousands of years in Eurasia before they were introduced to North America", "There are more than 7,500 cultivars of apples", "Different cultivars are bred for various tastes and uses, including cooking, eating raw, and cider or apple juice production"
  ],
  "contradicts": [
    "Trees and fruit are prone to fungal, bacterial, and pest problems"
  ]
}

Your turn:
Text Chunks:
######
""" + chunks + """
######

Fact:
######
""" + fact + """
######

Return only valid JSON in the specified format.
Output:
"""
            }
        ],
        temperature=0.2,
        top_p=0.9
    )

    log_usage(reply.usage.total_tokens, reply.usage.prompt_tokens,
              reply.usage.completion_tokens, "quotes_RAG_get_quotes_from_chunks", reply.model)

    return split_reply(model_reply=reply.choices[0].message.content, identifier='json')


async def quotes_RAG_check(quotes: str, fact: str, client, MODEL, prompt_name):
    reply = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are a fact verification assistant.

Your task is to determine whether a given fact is TRUE, FALSE, or INCONCLUSIVE based ONLY on the provided quotes.

You must strictly follow these rules:
1. Base your judgment ONLY on the provided quotes. Do not use prior knowledge.
2. Base your decision on the strength and clarity of evidence, not the number of quotes.
4. Ignore duplicate or redundant quotes.
3. Classify the result into one of three categories:
   - "TRUE": The context supports the claim.
   - "FALSE": The context contradicts the claim.
   - "INCONCLUSIVE": The context is insufficient, unclear, or only partially related to the claim.
4. Justify your classification in at least three sentences. 
   - Justification must NOT reference the given context/chunks/evidence or IDs- specify what you mean/ what you are referring to.
   - Never make a table in your justification.
5. Classification rules:
   - Return "TRUE" if the claim is supported in meaning, even if phrased differently or approximately.
   - Approximate numbers (e.g., "about", "approximately") should be considered correct if they are reasonably close to the context values.
   - Minor differences in wording or rounding do NOT invalidate support.
   - Return "FALSE" only if the claim contradicts the context.
   - Return "INCONCLUSIVE" only if the context truly lacks enough information.
   - For numerical claims with "approximately", differences within ~10% should be considered supported unless the exact value is critical.
   - If both sides exist, evaluate which side has stronger, more direct evidence.

6. Do NOT include explanations.
7. Do NOT use your own knowledge to classify the claim or justify your claim.
8. Return VALID JSON format ONLY.

Output format (strict VALID JSON):
{
  "fact_id": "FACT_ID_X",
  "output": "TRUE" | "FALSE" | "INCONCLUSIVE",
  "justification": "write your justification here!"
}

Do not include any explanation or additional text outside the JSON."""
            },
            {
                "role": "user",
                "content": """Determine whether the fact is TRUE, FALSE, or INCONCLUSIVE based only on the provided quotes.

Example:
Quotes:
######
{
  "fact_id": "FACT_ID_94",
  "supports": [
        "An apple is the round, edible fruit of an apple tree", "Apples have been grown for thousands of years in Eurasia before they were introduced to North America", "There are more than 7,500 cultivars of apples", "Different cultivars are bred for various tastes and uses, including cooking, eating raw, and cider or apple juice production"
  ],
  "contradicts": [
    "Trees and fruit are prone to fungal, bacterial, and pest problems"
  ]
}
######

Fact:
######
FACT_ID_94::: [There are many apples from different regions that can be consumed.]
######

Output:
{
  "fact_id": "FACT_ID_94",
  "output": "TRUE",
  "justification": "Apples have been grown in Eurasia and North America."
}


Your turn:
Quotes:
######
""" + quotes + """
######

Fact:
######
""" + fact + """
######

Output:
"""

            }
        ],
        temperature=0.2,
        top_p=0.9
    )

    log_usage(reply.usage.total_tokens, reply.usage.prompt_tokens,
              reply.usage.completion_tokens, prompt_name, reply.model)

    return split_reply(model_reply=reply.choices[0].message.content, identifier='json')
