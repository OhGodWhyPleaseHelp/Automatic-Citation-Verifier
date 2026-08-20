import threading
import asyncio
import jsonlines
from rapidfuzz import process, fuzz
import re
import unicodedata
from uuid import uuid4
import nltk
import requests
from lxml import etree
from bs4 import BeautifulSoup
import pymupdf
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from count_tokens import count
import os
import shutil
import random
from pathlib import Path


def refresh_output_folder(output_path:str):
    folder_path = Path(output_path)

    if folder_path.exists() and folder_path.is_dir():
        shutil.rmtree(folder_path)
        print(f"Deleted existing directory: {folder_path}")

    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"Created fresh directory: {folder_path}")

_write_tokens_file = None
_write_tokens_lock = threading.Lock()


def custom_count_tokens(text: str) -> int:
    return count(text=text)


def count_messages_tokens(messages: list[str]) -> int:
    return sum([custom_count_tokens(count(m.content)) for m in messages])


async def init_logger(output_filename="Output\\token_usage.log"):
    """Call this ONCE at startup"""
    global _write_tokens_file
    if os.path.exists(output_filename):
        os.remove(output_filename)
    _write_tokens_file = open(f"{output_filename}", "a", encoding="utf-8")


def close_logger():
    """Call this before your app exits"""
    global _write_tokens_file
    if _write_tokens_file and not _write_tokens_file.closed:
        _write_tokens_file.close()


def log_usage(total_tokens: int, input_tokens: int, output_tokens: int, prompt_name: str, model: str = "UTM"):
    """Safely append token usage to the log file (async-safe)"""
    entry = json.dumps({
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "prompt_name": prompt_name,
        "model": model
    }) + "\n"

    with _write_tokens_lock:
        if _write_tokens_file:
            _write_tokens_file.write(entry)


def log_entry(entry):
    """Use if entry has been made already."""
    entry = json.dumps(entry) + "\n"
    with _write_tokens_lock:
        if _write_tokens_file:
            _write_tokens_file.write(entry)


def get_year(intext_citation: str):
    split = intext_citation.split(",")
    year = split[-1]
    year = year.replace(")", "").replace(" ", '')
    return year


def get_headers():
    return {
        'User-Agent': random.choice(USER_AGENTS)
    }


def normalise(s):
    s = s.lower()
    if 'e.g.' in s:
        s = s.replace('e.g.', '')
    s = ''.join(
        c for c in unicodedata.normalize('NFC', s)
        if unicodedata.category(c) != 'Mn'
    )
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def create_author_in_text_citation(author: list[str], year: str):
    string_author = (" ".join(author)).lower().replace(".", "").replace(", ", " ").replace("& ", "").replace("and", "")
    only_last_names = re.sub(r'\b[a-zA-Z]\b', '', string_author)
    list_authors = [x for x in only_last_names.split(" ") if x != '']
    year = year.replace(" ", "")
    if len(list_authors) == 0:
        return ""
    elif len(list_authors) == 1:
        in_text_citation = f"({list_authors[0]}, {year})"
    elif len(list_authors) == 2:
        in_text_citation = f"({list_authors[0]} and {list_authors[1]}, {year})"
    else:
        in_text_citation = f"({list_authors[0]} et al., {year})"
    return in_text_citation


def find_author(author: list):
    string_author = (" ".join(author)).lower().replace(".", "").replace(", ", " ")
    # only_last_names = re.sub(r'\b[a-zA-Z]\b', '', string_author)
    # list_authors = [x for x in only_last_names.split(" ") if x != '']
    return string_author


def format_citation(citation: str):
    # idk how the hell it became a string somewhere.
    # Check for APA/ numerical
    if type(citation) is str and bool(re.search('[a-zA-Z]', citation)):
        parts = citation.split(",")
        year = parts[-1]
        parts.pop()
        # authors = " ".join(parts)
        return create_author_in_text_citation(parts, year)
    # This should catch numerical '[1]' etc.
    elif type(citation) is str:
        return f"[{citation.strip('[').strip(']')}]"
    else:
        print(f"Couldn't format citation: {citaoin}")


def tokenize_sentences(text: str) -> list[str]:
    """
    Tokenizes the text into sentences.
    Sometimes sentences with dates don't get split correctly, this aims to fix that.
    :param text: text to tokenize.
    :return: List of sentences.
    """
    # Unique marker that won't appear in normal text
    marker = f"A{uuid4().hex}"

    # Insert marker right after a digit+period, but keep the period with the number
    text = re.sub(r'(\d\.)(?=\s+[a-z])', r'\1' + marker, text)

    # Split on the marker – this forces a hard boundary exactly where we want it
    parts = text.split(marker)

    sentences = []
    for part in parts:
        # Use sent_tokenize on each part; the period is already at the end of the part
        sentences.extend(nltk.sent_tokenize(part.strip()))

    return sentences


def join_citations(sentences: list[str]) -> list[str]:
    """
    Merges sentences where the current sentence ends with common abbreviations
    (like 'et al.', 'e.g.', 'i.e.', 'etc.') that often cause tokenizers to
    incorrectly split a single sentence.

    :param sentences: Sentences found using a sentence tokenizer (e.g., nltk).
    :return: List of merged sentences.
    """
    result = []
    i = 0

    # Suffixes that commonly cause false sentence splits.
    # Using 'et al.' instead of 'al.' prevents false positives with names like "Al."
    false_split_suffixes = ('et al.', 'e.g.', 'i.e.', 'etc.')

    while i < len(sentences):
        current = sentences[i]

        # Merge while the next sentence exists and current ends with a false split suffix
        while (i + 1 < len(sentences) and
               current.rstrip().lower().endswith(false_split_suffixes)):
            # Use lstrip() on the next sentence to prevent double spaces
            current += f" {sentences[i + 1].lstrip()}"
            i += 1

        result.append(current)
        i += 1

    return result


def highlight_normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r'page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[(\d+)\]', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*/\s*', '/', text)
    return text.strip()


def find_best_page(doc, sentence, threshold=85):
    target = highlight_normalise(sentence)
    best_page = None
    best_score = 0
    relevant_sentence = ""
    for page_num, page in enumerate(doc):
        raw_text = page.get_text()
        page_text = highlight_normalise(raw_text)
        score = fuzz.partial_ratio(target, page_text)
        if score > best_score:
            best_score = score
            best_page = page_num
            best_sentence_score = 0
            all_sentences = join_citations(tokenize_sentences(raw_text))
            for s in all_sentences:
                s_norm = highlight_normalise(s)
                sentence_score = max(fuzz.ratio(target, s_norm), fuzz.token_sort_ratio(target, s_norm))
                if sentence_score > best_sentence_score:
                    best_sentence_score = sentence_score
                    relevant_sentence = s
            if score == 100.0:
                break
    if best_score >= threshold:
        return best_page, best_score, relevant_sentence
    return None, best_score, None


def remove_page_break(sentence: str):
    if "page **" in sentence:
        pattern = r'page\s*\*\*.+?\*\*\s*of\s*\*\*.+?\*\*'
        cleaned = re.sub(pattern, '', sentence, flags=re.IGNORECASE)
        cleaned = cleaned.split("\n\n")
        cleaned = [re.sub(r' {2,}', ' ', c) for c in cleaned]
        cleaned = [re.sub(r'\s+([.,!?;])', r'\1', c).strip() for c in cleaned]
        cleaned = [c for c in cleaned if c != " " and c != ""]
        return cleaned
    return [sentence]


def highlight_text(pdf_path: str, verdicts: dict[str, tuple[str, str, str, str]], verification_type: str,
                   output_path: str = "Output"):
    """
    Tokenizes the text into sentences.
    Sometimes sentences with dates don't get split correctly, this aims to fix that.
    :param text: text to tokenize.
    :return: List of sentences.
    """
    doc = pymupdf.open(pdf_path)
    path = pdf_path.split('\\')
    path = f"{verification_type}_{path[-1]}"
    output_name = f"{output_path}\\{path}"

    for fact in verdicts.values():
        # ('in_text_citation', 'fact', 'original sentence', 'verdict', 'justification')
        justification = fact[4]
        verdict = fact[3]
        sentence = fact[2]
        sentences = remove_page_break(sentence)
        colour_map = {
            "TRUE": (0.0, 1.0, 0.0),  # green
            "FALSE": (1.0, 0.0, 0.0),  # red
            "INCONCLUSIVE": (0.0, 0.0, 1.0),  # blue
            "DEFAULT": (1.0, 1.0, 0.0)  # yellow
        }
        colour = colour_map.get(verdict, colour_map["DEFAULT"])
        for s in sentences:
            page_num, score, best_sentence_match = find_best_page(doc, s)

            if page_num is None:
                print(f"NO PAGE MATCH: {s}")
                continue

            if best_sentence_match:
                page = doc[page_num]
                quads = page.search_for(s, quads=True)
                if not quads:
                    continue
                else:
                    for q in quads:
                        annot = page.add_highlight_annot(q)
                        annot.set_colors(stroke=colour)
                        annot.set_info(
                            content=f"Verdict: {verdict}\n\n{justification}",
                            subject="Verification Result"
                        )
                        annot.set_info()
                        annot.update()
            else:
                print(f"PAGE FOUND ({score:.1f}) "f"BUT NO TEXT MATCH:\n{sentence}\n")

    doc.save(output_name)
    doc.close()
    print("Completed!")


def merge_justifications(verdicts: dict[dict[str, tuple[str, str, str, str]]]):
    grouped = defaultdict(list)
    # Group by original sentence
    for key, value in verdicts.items():
        sentence = value[2]
        grouped[sentence].append((key, value))

    # Merge justifications
    result = {}

    for sentence, items in grouped.items():
        base_key, base_value = items[0]
        if len(items) > 1:
            # Take the first tuple as base
            all_verdicts = [tup[3] for _, tup in items]
            most_common_verdict = max(set(all_verdicts), key=all_verdicts.count)
            merged_justifications = [f"{in_text_citation}'s verdict: {tup[3]}\nJustification: {tup[4]}" for
                                     in_text_citation, tup in items]

            # Create new tuple with merged justifications
            new_value = (
                base_value[0],  # in_text_citation
                base_value[1],  # fact
                base_value[2],  # original sentence
                most_common_verdict,  # verdict
                "\n".join(merged_justifications)  # merged justification
            )
            result[base_key] = new_value
        else:
            result[base_key] = base_value

    return result


def evaluate(verdicts: dict, verification_type: str, reference_pdf_path: str, output_path: str) -> tuple[tuple, tuple]:
    """
    Provides the evaluation metrics for the verdicts and saves the confusion matrix in the correct path.
    :param verdicts: Dictionary of verdicts.
    :param verification_type: Type of verification used -> NaiveRAG/Vectorless RAG etc.
    :param reference_pdf_path: Path to reference PDFs.
    :param output_path: Path to output.
    :return: Accuracy, precision, recall, F1 score.
    """
    # Merge justifications per original sentence
    verdicts = merge_justifications(verdicts=verdicts)

    # with open(f"{output_path}\\{verification_type}_verdicts.json", "w") as f:
    #     json.dump(verdicts, f)

    LABELS = ["TRUE", "INCONCLUSIVE", "FALSE"]
    split = reference_pdf_path.split("\\")
    ground_truth_path = f"{split[0]}\\{split[1]}\\ground_truth.json"
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)

    gt_map = {}
    gt_map_stripped = {}
    for label, sentences in ground_truth.items():
        for s in sentences:
            gt_map[s.lower()] = label.upper()
            gt_map_stripped[re.sub(r'[^\w]', '', s.replace(" ", ''))] = label.upper()

    gt_sentences = list(gt_map.keys())

    y_true_eval, y_pred_eval = [], []
    y_true_other, y_pred_other = [], []

    already_processed = set()
    for key, fact_tuple in verdicts.items():
        if len(fact_tuple) < 4:
            print(f"For some reason, this isn't 4 things long! {fact_tuple}")
            continue
        pred_label = fact_tuple[3].upper()
        sentence = fact_tuple[2].lower()
        if sentence in already_processed:
            continue

        stripped_sentence = re.sub(r'[^\w]', '', sentence.replace(" ", ''))
        if sentence in gt_map:
            y_true_eval.append(gt_map[sentence])
            y_pred_eval.append(pred_label)
        elif stripped_sentence in gt_map_stripped:
            y_true_eval.append(gt_map_stripped[stripped_sentence])
            y_pred_eval.append(pred_label)
        else:
            # Fuzzy match: returns (match_string, score, index) or None if < cutoff
            match = process.extractOne(sentence, gt_sentences, score_cutoff=90)
            if match:
                y_true_eval.append(gt_map[match[0]])  # match[0] = matched GT sentence
                y_pred_eval.append(pred_label)
            else:
                # If no match, then the sentence doesn't belong to evaluation.
                y_true_other.append("TRUE")
                y_pred_other.append(pred_label)
        already_processed.add(sentence)

    # One set for only evaluation sentences
    acc = accuracy_score(y_true_eval, y_pred_eval)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_eval, y_pred_eval, average="macro",
                                                               zero_division=0)
    cm = confusion_matrix(y_true_eval, y_pred_eval, labels=LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot().figure_.savefig(f"{output_path}\\{verification_type}_evaluation_citations_only_confusion_matrix.png")

    # Add evaluation to all as well
    y_true_other.extend(y_true_eval)
    y_pred_other.extend(y_pred_eval)
    # One set for all citations within research paper
    acc1 = accuracy_score(y_true_other, y_pred_other)
    precision1, recall1, f11, _ = precision_recall_fscore_support(y_true_other, y_pred_other, average="macro",
                                                                  zero_division=0)
    cm = confusion_matrix(y_true_other, y_pred_other, labels=LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot().figure_.savefig(f"{output_path}\\{verification_type}_all_citations_confusion_matrix.png")

    return (acc, precision, recall, f1), (acc1, precision1, recall1, f11)


def justifications_count_tokens(token_filepath: str):
    preprocessing = {"total": 0, "input": 0, "output": 0}
    naive_tokens = {"total": 0, "input": 0, "output": 0}
    som_tokens = {"total": 0, "input": 0, "output": 0}
    qraftlite_tokens = {"total": 0, "input": 0, "output": 0}
    with jsonlines.open(token_filepath) as reader:
        for record in reader:
            if "SOM" in record["prompt_name"]:
                som_tokens["total"] += record["total_tokens"]
                som_tokens["input"] += record["input_tokens"]
                som_tokens["output"] += record["output_tokens"]
            elif "QraftLite" in record["prompt_name"]:
                qraftlite_tokens["total"] += record["total_tokens"]
                qraftlite_tokens["input"] += record["input_tokens"]
                qraftlite_tokens["output"] += record["output_tokens"]
            elif "enhancedNaive_RAG" in record["prompt_name"] or "naive_RAG" in record["prompt_name"]:
                naive_tokens["total"] += record["total_tokens"]
                naive_tokens["input"] += record["input_tokens"]
                naive_tokens["output"] += record["output_tokens"]
            else:
                preprocessing["total"] += record["total_tokens"]
                preprocessing["input"] += record["input_tokens"]
                preprocessing["output"] += record["output_tokens"]

    total, input, output = 0, 0, 0
    for dic in [preprocessing, qraftlite_tokens, naive_tokens, som_tokens]:
        total += dic["total"]
        input += dic["input"]
        output += dic["output"]

    return total, input, output, {"Prepocessing": preprocessing, "Naive RAG Enhanced": naive_tokens,
                                  "QraftLite": qraftlite_tokens, "SOM": som_tokens}


def all_evaluations_count_tokens(token_filepath: str):
    preprocessing = {"total": 0, "input": 0, "output": 0}
    naive_tokens = {"total": 0, "input": 0, "output": 0}
    naive_enhanced_tokens = {"total": 0, "input": 0, "output": 0}
    quotes_tokens = {"total": 0, "input": 0, "output": 0}
    quotes_enhanced_tokens = {"total": 0, "input": 0, "output": 0}
    vectorlessrag_tokens = {"total": 0, "input": 0, "output": 0}
    with jsonlines.open(token_filepath) as reader:
        for record in reader:
            if "enhancedNaive_RAG" in record["prompt_name"]:
                naive_enhanced_tokens["total"] += record["total_tokens"]
                naive_enhanced_tokens["input"] += record["input_tokens"]
                naive_enhanced_tokens["output"] += record["output_tokens"]
            elif "enhancedQuotes_RAG" in record["prompt_name"]:
                quotes_enhanced_tokens["total"] += record["total_tokens"]
                quotes_enhanced_tokens["input"] += record["input_tokens"]
                quotes_enhanced_tokens["output"] += record["output_tokens"]
            elif "naive_RAG" in record["prompt_name"]:
                naive_tokens["total"] += record["total_tokens"]
                naive_tokens["input"] += record["input_tokens"]
                naive_tokens["output"] += record["output_tokens"]
            elif "quotes_RAG" in record["prompt_name"]:
                quotes_tokens["total"] += record["total_tokens"]
                quotes_tokens["input"] += record["input_tokens"]
                quotes_tokens["output"] += record["output_tokens"]
            elif "VectorlessRAG" in record["prompt_name"]:
                vectorlessrag_tokens["total"] += record["total_tokens"]
                vectorlessrag_tokens["input"] += record["input_tokens"]
                vectorlessrag_tokens["output"] += record["output_tokens"]
            else:
                preprocessing["total"] += record["total_tokens"]
                preprocessing["input"] += record["input_tokens"]
                preprocessing["output"] += record["output_tokens"]

    total, input, output = 0, 0, 0
    for dic in [preprocessing, naive_tokens, naive_enhanced_tokens, quotes_tokens, quotes_enhanced_tokens,
                vectorlessrag_tokens]:
        total += dic["total"]
        input += dic["input"]
        output += dic["output"]

    return total, input, output, {"Preprocessing": preprocessing, "Naive RAG": naive_tokens,
                                  "Naive RAG Enhanced": naive_enhanced_tokens, "Quotes RAG": quotes_tokens,
                                  "Quotes RAG Enhanced": quotes_enhanced_tokens, "Vectorless RAG": vectorlessrag_tokens}


def agentic_pipeline_count_tokens(token_filepath: str):
    preprocessing = {"total": 0, "input": 0, "output": 0}
    naive_tokens = {"total": 0, "input": 0, "output": 0}
    naive_enhanced_tokens = {"total": 0, "input": 0, "output": 0}
    quotes_tokens = {"total": 0, "input": 0, "output": 0}
    quotes_enhanced_tokens = {"total": 0, "input": 0, "output": 0}
    vectorlessrag_tokens = {"total": 0, "input": 0, "output": 0}
    single_agent_tokens = {"total": 0, "input": 0, "output": 0}
    multi_agent_tokens = {"total": 0, "input": 0, "output": 0}
    som_tokens = {"total": 0, "input": 0, "output": 0}
    qraftlite_tokens = {"total": 0, "input": 0, "output": 0}
    with jsonlines.open(token_filepath) as reader:
        for record in reader:
            if "SOM" in record["prompt_name"]:
                som_tokens["total"] += record["total_tokens"]
                som_tokens["input"] += record["input_tokens"]
                som_tokens["output"] += record["output_tokens"]
            elif "QraftLite" in record["prompt_name"]:
                qraftlite_tokens["total"] += record["total_tokens"]
                qraftlite_tokens["input"] += record["input_tokens"]
                qraftlite_tokens["output"] += record["output_tokens"]
            elif "enhancedNaive_RAG" in record["prompt_name"]:
                naive_enhanced_tokens["total"] += record["total_tokens"]
                naive_enhanced_tokens["input"] += record["input_tokens"]
                naive_enhanced_tokens["output"] += record["output_tokens"]
            elif "enhancedQuotes_RAG" in record["prompt_name"]:
                quotes_enhanced_tokens["total"] += record["total_tokens"]
                quotes_enhanced_tokens["input"] += record["input_tokens"]
                quotes_enhanced_tokens["output"] += record["output_tokens"]
            elif "naive_RAG" in record["prompt_name"]:
                naive_tokens["total"] += record["total_tokens"]
                naive_tokens["input"] += record["input_tokens"]
                naive_tokens["output"] += record["output_tokens"]
            elif "quotes_RAG" in record["prompt_name"]:
                quotes_tokens["total"] += record["total_tokens"]
                quotes_tokens["input"] += record["input_tokens"]
                quotes_tokens["output"] += record["output_tokens"]
            elif "VectorlessRAG" in record["prompt_name"]:
                vectorlessrag_tokens["total"] += record["total_tokens"]
                vectorlessrag_tokens["input"] += record["input_tokens"]
                vectorlessrag_tokens["output"] += record["output_tokens"]
            elif "SingleAgent" in record["prompt_name"] or "Single_Agent" in record["prompt_name"] or "Single Agent" in \
                    record["prompt_name"]:
                single_agent_tokens["total"] += record["total_tokens"]
                single_agent_tokens["input"] += record["input_tokens"]
                single_agent_tokens["output"] += record["output_tokens"]
            elif "MulitAgent" in record["prompt_name"] or "Multi_Agent" in record["prompt_name"] or "MultiAgent" in \
                    record["prompt_name"]:
                multi_agent_tokens["total"] += record["total_tokens"]
                multi_agent_tokens["input"] += record["input_tokens"]
                multi_agent_tokens["output"] += record["output_tokens"]
            else:
                preprocessing["total"] += record["total_tokens"]
                preprocessing["input"] += record["input_tokens"]
                preprocessing["output"] += record["output_tokens"]

    names = ["Preprocessing", "Naive RAG", "Naive RAG Enhanced", "Quotes RAG", "Quotes RAG Enhanced", "Vectorless RAG",
             "SOM", "QraftLite", "Single Agent", "Multi Agent"]
    methods = [preprocessing, naive_tokens, naive_enhanced_tokens, quotes_tokens, quotes_enhanced_tokens,
               vectorlessrag_tokens, som_tokens, qraftlite_tokens, single_agent_tokens, multi_agent_tokens]
    has_tokens = {}
    total, input, output = 0, 0, 0
    for i in range(len(methods)):
        dic = methods[i]
        if dic["total"] > 0:
            total += dic["total"]
            input += dic["input"]
            output += dic["output"]
            has_tokens[names[i]] = dic

    return total, input, output, has_tokens


def create_token_graph(methods: dict, output_filepath: str):
    # Categories and methods
    categories = ["input", "output", "total"]
    # X-axis positions
    n_methods = len(methods)
    x = np.arange(len(categories))
    width = 0.8 / n_methods  # Ensures group width is always 0.8 (leaves 0.2 padding)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars
    for i, (label, data) in enumerate(methods.items()):
        values = [data[cat] for cat in categories]
        ax.bar(x + i * width, values, width, label=label)

    # Customize axes
    ax.set_xlabel("Token Type", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Token Counts", fontsize=14, fontweight='bold')

    # Dynamically center x-ticks under each group
    ax.set_xticks(x + (n_methods - 1) * width / 2)
    ax.set_xticklabels(categories)

    ax.legend(title="Step")
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Add value labels
    for i, (label, data) in enumerate(methods.items()):
        values = [data[cat] for cat in categories]
        for j, v in enumerate(values):
            ax.text(x[j] + i * width, v + 0.1, str(v),
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(f"{output_filepath}/token_usage.png", bbox_inches='tight')


def create_evaluation_metrics_graph(data: dict, output_filepath: str):
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

    n_metrics = len(metrics)
    n_methods = len(data)

    x = np.arange(n_metrics)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (method_name, values) in enumerate(data.items()):
        bar_positions = x - (n_methods - 1) * width / 2 + i * width
        ax.bar(bar_positions, values, width, label=method_name, alpha=0.9)

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)

    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    if len(data.keys()) > 1:
        ax.legend(title="Method", loc='upper left', bbox_to_anchor=(1, 1))

    text_offset = 0.02
    for i, (method_name, values) in enumerate(data.items()):
        bar_positions = x - (n_methods - 1) * width / 2 + i * width
        for j, v in enumerate(values):
            ax.text(bar_positions[j], v + text_offset, f"{v:.2f}",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    fig.savefig(output_filepath, bbox_inches='tight')
