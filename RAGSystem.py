from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import asyncio
from LLM import *
import json
import json5
from utils import *
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from itertools import combinations
import random
import torch

if torch.cuda.is_available():
    device = 'cuda'
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("WARNING: CUDA is not available. Running on CPU.")
sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
# sentence_transformer_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
# sentence_transformer_model = SentenceTransformer("MongoDB/mdbr-leaf-mt", device=device)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into chunks with some overlap.
    :param text: Input string of text.
    :param chunk_size: How many characters a chunk should be.
    :param overlap: How many characters to overlap. Overlaps into the next chunk (not before).
    :return: List of chunks as strings.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def paragraph_chunking(text:str):
    x = text.split("\n\n")
    return x


def min_max_chunk_text(text: str, hard_threshold: float = 0.6, c_parameter: float = 0.9, init_constant: float = 1.5) -> list[str]:
    """
    Uses min-max chunking technique, by https://link.springer.com/article/10.1007/s10791-025-09638-7
    :param text: Text from reference PDFs.
    :param hard_threshold: Hard threshold, default value taken from literature.
    :param c_parameter: C parameter, default value taken from literature.
    :param init_constant: initialisation constand, default value from literature.
    :return: Chunks - a list of strings.
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    all_sentences = join_citations(tokenize_sentences(text))
    encodings = {s: sentence_transformer_model.encode(s) for s in all_sentences}
    all_chunks = []

    chunk = []
    for sentence in all_sentences:
        if len(chunk) == 0:
            chunk.append(sentence)
        elif len(chunk) == 1:
            sentence_1 = chunk[0]
            similarity = cosine_similarity(encodings[sentence_1].reshape(1, -1), encodings[sentence].reshape(1, -1))[0][0]

            if init_constant * similarity >= hard_threshold:
                chunk.append(sentence)
            else:
                string_chunk = " ".join(chunk)
                all_chunks.append(string_chunk)
                chunk = [sentence]
        else:
            minimum_similarity = min(
                cosine_similarity(encodings[s1].reshape(1, -1), encodings[s2].reshape(1, -1))[0][0] for s1, s2 in
                combinations(chunk, 2))
            maximum_similarity = max(
                cosine_similarity(encodings[sentence].reshape(1, -1), encodings[s].reshape(1, -1))[0][0] for s in chunk)

            chunk_threshold = max(c_parameter * minimum_similarity * (sigmoid(len(chunk))), hard_threshold)
            if maximum_similarity >= chunk_threshold:
                chunk.append(sentence)
            else:
                string_chunk = " ".join(chunk)
                all_chunks.append(string_chunk)
                chunk = [sentence]

    if len(chunk) > 0:
        string_chunk = " ".join(chunk)
        all_chunks.append(string_chunk)
        chunk = []

    chunks = [(chunk, custom_count_tokens(chunk)) for chunk in all_chunks]
    parsed_chunks = []
    for c in chunks:
        if c[1] > 5000:
            with open(f"DUMPS/RAG_CHUNKING_RID_{random.randint(0,9999)}", "w", encoding="utf-8") as f:
                f.write(c[0])
        else:
            parsed_chunks.append(c[0])

    return parsed_chunks


def create_FAISS_indices(references: dict[str, str]) -> tuple[dict, dict]:
    """
    Creates the FAISS indices for the input references.
    :param references: Dictionary of references, like {"in text citation": "paper_text"}
    :return: The Faiss indices and the chunks of text, need both to get the text after querying.
    """
    # chunked_text = {ref: chunk_text(references.get(ref)) for ref in references}
    chunked_text = {ref: min_max_chunk_text(references.get(ref)) for ref in references}
    before = len(chunked_text)
    chunked_text = {ref: chunked_text.get(ref) for ref in chunked_text if chunked_text.get(ref) is not None}
    print(f"Parsed out {before-len(chunked_text)} chunks")
    # chunked_text = {ref: ref.split("\n\n") for ref in references} # Split on paragraph
    embedded_texts = {paper: sentence_transformer_model.encode(chunked_text.get(paper)) for paper in
                      chunked_text.keys()}

    faiss_indexes = {}
    chunk_stores = {}

    for paper, embeddings in embedded_texts.items():
        if len(chunked_text.get(paper)) == 0:
            continue
        embeddings = np.array(embeddings).astype("float32")

        try:
            faiss.normalize_L2(embeddings)
        except Exception:
            try:
                print(f"Shape of {paper} embeddings: {embeddings.shape}")
            except Exception:
                pass
            with open(f"Output/faiss_problem_{paper}.json", "w", encoding='utf-8') as f:
                json.dump(chunked_text, f, ensure_ascii=False, indent=4)
                json.dump(embedded_texts, f, ensure_ascii=False, indent=4)
                json.dump(faiss_indexes, f, ensure_ascii=False, indent=4)
            continue
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine similarity

        index.add(embeddings)

        # Store index + chunks
        faiss_indexes[paper] = index
        chunk_stores[paper] = chunked_text[paper]

    return faiss_indexes, chunk_stores


def query_faiss_index(query: str, faiss_index, chunk_store, top_k=15) -> list[str]:
    """
    Extract k many chunks based on the fact and the in_text_citation.
    :param query: Fact needing to be verified
    :param faiss_index: Faiss indexes of the papers.
    :param chunk_store: Dictionary of chunks stores.
    :param top_k: Number of chunks to return.
    :return: List of chunks of text from the relevant paper.
    """
    query_embedding = sentence_transformer_model.encode([query]).astype("float32")

    # normalize for cosine similarity
    faiss.normalize_L2(query_embedding)

    # index = faiss_index[paper_id]
    # chunks = chunk_store[paper_id]

    D, I = faiss_index.search(query_embedding, top_k)
    # Seems like results only have Chunk index, score
    results = [chunk_store[i] for i in I[0]]
    return results


async def evaluate_chunks(fact: str, chunks: list[str], semaphore: asyncio.Semaphore, max_returns: int = 10, prompt_name:str="enhancedNaive_rag_evaluate_chunks") -> list[
    str]:
    client, model = async_client()

    async def evaluate_one_chunk(chunk: str, fact: str, semaphore, chunk_id):
        async with semaphore:
            return (chunk_id, await evaluate_chunk(chunk=chunk, fact=fact, client=client, MODEL=model, prompt_name=prompt_name))

    tasks = []
    for i in range(len(chunks)):
        tasks.append(
            asyncio.create_task(evaluate_one_chunk(chunk=chunks[i], fact=fact, semaphore=semaphore, chunk_id=i)))

    responses = await asyncio.gather(*tasks)

    # Calculate total score for each chunk
    all_scores = []
    for tup in responses:
        chunk_index = tup[0]
        try:
            scores = json.loads(tup[1])
        except json.decoder.JSONDecodeError:
            print(f"JSONifying model reply went wrong!\n{tup[1]}")
            continue
        try:
            total_score = int(scores.get('relevance')) + int(scores.get('clarity')) + int(
                scores.get('applicability')) + int(scores.get('quality'))
        except Exception:
            # If something goes wrong, score is 0.
            print(f"Calculating total score from chunk evaluation went wrong!\n{scores}")
            total_score = 0
        all_scores.append({"chunk_index": chunk_index, "score": total_score})

    # Sort highest first
    sorted_scores = sorted(all_scores, key=lambda x: x["score"], reverse=True)
    # Only keep up to the number of max_returns
    top_scores = sorted_scores[:max_returns]
    top_chunks = [chunks[top_scores[i].get('chunk_index')] for i in range(len(top_scores))]

    return top_chunks


async def naive_RAG_query(faiss_indexes: dict, chunk_stores: dict, groups: dict, facts: dict,
                          semaphore: asyncio.Semaphore, chunk_evaluation: bool = False, max_chunks: int = 10,
                          enhancement_chunks: int = 15) -> dict[str, tuple[str, str, str, str]]:
    """
    Naive RAG - collects chunks and passes these directly to the LLM as context to verify a fact.
    :param faiss_indexes: FAISS Index for each paper.
    :param chunk_stores: Chunk storage for each paper.
    :param groups: Facts groups {paper: [fact ids]}
    :param facts: Facts {"fact id": ('in_text_citation', 'fact', 'original sentence')}
    :param semaphore: Semaphore to limit spam to LLM.
    :param chunk_evaluation: To add chunk evaluation before passing to LLM.
    :param max_chunks: Max number of chunks to use for LLM.
    :param enhancement_chunks: Number of extra chunks to retrieve for the enhancement.
    :return: Dictionary like: {"fact id": ('in_text_citation', 'fact', 'original sentence', 'verdict')}
    """
    client, model = async_client()

    async def naive_check_fact(context: str, fact: str, semaphore, fact_id, prompt_name):
        async with semaphore:
            return (fact_id, await naive_RAG_check(context=context, fact=fact, client=client, MODEL=model, prompt_name=prompt_name))

    tasks = []
    for paper in faiss_indexes.keys():
        linked_facts = groups.get(paper)
        if linked_facts:
            for i in range(len(linked_facts)):
                fact_key = linked_facts[i]
                fact = facts.get(fact_key)[-1]
                prompt_name = "naive_RAG_check"
                if chunk_evaluation:
                    assert enhancement_chunks + max_chunks >= max_chunks
                    prompt_name = "enhancedNaive_RAG_check"
                    retrieved = query_faiss_index(query=fact, faiss_index=faiss_indexes.get(paper),
                                                  chunk_store=chunk_stores.get(paper),
                                                  top_k=max_chunks + enhancement_chunks)
                    chunks = await evaluate_chunks(fact=fact, chunks=retrieved, semaphore=semaphore, max_returns=max_chunks, prompt_name="enhancedNaive_RAG_evaluate_chunks")
                else:
                    chunks = query_faiss_index(query=fact, faiss_index=faiss_indexes.get(paper),
                                               chunk_store=chunk_stores.get(paper), top_k=max_chunks)
                string_chunks = "\n".join([f"CHUNK_ID_{j}::: [{chunks[j]}]" for j in range(len(chunks))])
                string_fact = f"FACT_ID_{fact_key}::: [{fact}]"
                tasks.append(asyncio.create_task(
                    naive_check_fact(context=string_chunks, fact=string_fact, semaphore=semaphore, fact_id=fact_key, prompt_name=prompt_name)))

    responses = await asyncio.gather(*tasks)

    facts_evidence = {}
    for tup in responses:
        fact_id = tup[0]
        reply = tup[1]
        try:
            json_text = json.loads(reply)
            verdict = json_text.get("output")
            justification = json_text.get("justification")
        except json.decoder.JSONDecodeError:
            # Try more robuse version, then just fill in a nerror.
            try:
                json_text = json5.loads(reply)
                verdict = json_text.get("output")
                justification = json_text.get("justification")
            except Exception as e:
                print(
                    f"Could not JSONify model reply!\nModel Reply: {reply}\n^Model Reply\n Using INCONCLUSIVE verdict instead!")
                verdict = "INCONCLUSIVE"
                justification = "Error, using INCONCLUSIVE verdict instead"
        fact = facts.get(fact_id)
        if fact is None:
            print(f"Fact ID: {fact_id}\tModel Reply: {json_text}\nFacts: {facts}")
            break
        # ('in_text_citation', 'fact', 'original sentence', 'verdict')
        fact = (fact[0], fact[1], fact[2], verdict, justification)
        facts_evidence[fact_id] = fact

    return facts_evidence


async def quotes_check_fact(faiss_indexes: dict, chunk_stores: dict, groups: dict, facts: dict,
                            semaphore: asyncio.Semaphore, chunk_evaluation: bool = False, max_chunks: int = 10,
                            enhancement_chunks: int = 15) -> dict[str, tuple[str, str, str, str]]:
    """
    RAG to get chunks, find quotes and then uses quotes to verify fact.
    :param faiss_indexes: FAISS Index for each paper.
    :param chunk_stores: Chunk storage for each paper.
    :param groups: Facts groups {paper: [fact ids]}
    :param facts: Facts [(in_text_citation, fact, original_sentence)]
    :param semaphore: Semaphore to limit spam to LLM.
    :param chunk_evaluation: To add chunk evaluation before passing to LLM.
    :param max_chunks: Max number of chunks to use for LLM.
    :param enhancement_chunks: Number of extra chunks to retrieve for the enhancement.
    :return: Dictionary like: {"fact id": ('in_text_citation', 'fact', 'original sentence', 'verdict')}
    """
    client, model = async_client()

    async def quote_check_fact(context: str, fact: str, semaphore, fact_id, prompt_name):
        async with semaphore:
            quotes = await get_quotes_from_chunks(chunks=context, fact=fact, client=client, MODEL=model)
            return (fact_id, await quotes_RAG_check(quotes=quotes, fact=fact, client=client, MODEL=model, prompt_name=prompt_name))

    tasks = []
    for paper in faiss_indexes.keys():
        linked_facts = groups.get(paper)
        if linked_facts:
            for i in range(len(linked_facts)):
                fact_key = linked_facts[i]
                fact = facts.get(fact_key)[-1]
                prompt_name = "quotes_RAG_check"
                if chunk_evaluation:
                    assert enhancement_chunks + max_chunks >= max_chunks
                    prompt_name = "enhancedQuotes_RAG_check"
                    retrieved = query_faiss_index(query=fact, faiss_index=faiss_indexes.get(paper),
                                                  chunk_store=chunk_stores.get(paper),
                                                  top_k=max_chunks + enhancement_chunks)
                    chunks = await evaluate_chunks(fact=fact, chunks=retrieved, semaphore=semaphore, max_returns=max_chunks, prompt_name="enhancedQuotes_RAG_evaluate_chunks")
                else:
                    chunks = query_faiss_index(query=fact, faiss_index=faiss_indexes.get(paper),
                                               chunk_store=chunk_stores.get(paper), top_k=max_chunks)
                string_chunks = "\n".join([f"CHUNK_ID_{j}::: [{chunks[j]}]" for j in range(len(chunks))])
                string_fact = f"FACT_ID_{fact_key}::: [{fact}]"
                tasks.append(
                    asyncio.create_task(
                        quote_check_fact(context=string_chunks, fact=string_fact, semaphore=semaphore, fact_id=fact_key, prompt_name=prompt_name)))

    responses = await asyncio.gather(*tasks)

    facts_evidence = {}
    for tup in responses:
        fact_id = tup[0]
        reply = tup[1]
        try:
            json_text = json.loads(reply)
            verdict = json_text.get("output")
            justification = json_text.get("justification")
        except json.decoder.JSONDecodeError:
            print(
                f"Could not JSONify model reply!\nModel Reply: {reply}\n^Model Reply\n Using INCONCLUSIVE verdict instead!")
            verdict = "INCONCLUSIVE"
            justification = "Model failed! Using 'INCONCLUSIVE' verdict instead!"
        fact = facts.get(fact_id)
        fact = (fact[0], fact[1], fact[2], verdict, justification)
        facts_evidence[fact_id] = fact

    return facts_evidence


if __name__ == "__main__":
    from ParseInputPDF import extract_text
    text, _ = extract_text("Research_Topics.pdf")
    min_max_chunk_text(text=text)