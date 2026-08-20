# from PyPDF2 import PdfReader

import json
import asyncio
import os
# from marker.converters.pdf import PdfConverter
# from marker.models import create_model_dict
# from marker.output import text_from_rendered
# from marker.config.parser import ConfigParser
import pymupdf4llm
from refextract import extract_references_from_file, extract_references_from_string
import re

from LLM import *
from utils import *
from PaperRetrieval import *


def find_reference_index(paragraphs: list[str], name: str):
    try:
        return paragraphs.index(name)
    except ValueError:
        return None


def find_paragraphs(filename: str, find_reference_strings: bool = False) -> tuple[list[str], list[str]]:
    """
    Find the paragraphs in the PDF. May take a long time (~10-20 minutes).
    Uses marker
    :param filename: Path to pdf.
    :return: list of paragraphs.
    """
    md_text = pymupdf4llm.to_markdown(filename)
    paragraphs = md_text.split("\n\n")

    if find_reference_strings:
        filtered = [p.lower().replace(" ", '').replace("*", "").replace("#", '') for p in paragraphs]
        reference_index = None
        for tag in ["references", "reference", "bibliography"]:
            reference_index = find_reference_index(filtered, tag)
            if reference_index is not None:
                break
        if reference_index is not None:
            references_strings = paragraphs[reference_index + 1:]
            return paragraphs, references_strings

    return paragraphs, []


def evaluation_extract_texts(filename: str) -> str:
    """
    Extracts the text from the PDF.

    As the Evalution PDFs are modified, this version of extracting the text should be used!

    :param filename: Path to the PDF.
    :return: Lowercase text from PDF.
    """
    file = pymupdf.open(filename)
    full_text = ""
    for page in file:  # iterate the document pages
        full_text += page.get_text()

    return full_text.lower().replace("-\n", '').replace('\\', '/')


def extract_text(filename: str, find_reference_strings: bool = False) -> tuple[str, list[str]]:
    """
    Extracts the text from the PDF.
    :param filename: Path to the PDF.
    :param find_reference_strings: Try to extract the reference strings
    :return: Lowercase text from PDF.
    """
    paragraphs, reference_strings = find_paragraphs(filename=filename, find_reference_strings=find_reference_strings)
    text = "\n\n".join(paragraphs)
    text = text.replace("_","")

    return text.lower().replace("-\n", '').replace('\\', '/'), reference_strings


def extract_reference_list_string(filename):
    pages_text = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            # get_text() defaults to plain text. Equivalent to page.get_text("text")
            pages_text.append(page.get_text())
    # Join with a visible separator (optional but useful)
    full_text = "".join(pages_text)
    text, reference_list = split_text_references(full_text)
    return reference_list


def split_text_references(full_text: str) -> tuple[str, str]:
    """
    Splits the text into the text and separate reference list.
    :param full_text: PDF text.
    :return: text and reference list.
    """
    # Try splitting the text on some key words
    REF_HEADERS = r'(?mi)^\s*(?:\d+\s*[\.\)]?\s*)?(?:references|bibliography|works\s+cited|appendix|appendices)\s*[\.:]?\s*$'
    parts = re.split(REF_HEADERS, full_text, flags=re.IGNORECASE)
    # If there's exactly two parts, then this is likely the text and reference list
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    # If there is only one part, then the split didn't work.
    elif len(parts) == 1:
        raise Exception("Splitting text and reference list went wrong! Did not split!")

    # Otherwise, iterate backwards through the parts and use the LLM to check if the part is a reference list part.
    reference_list = ""
    reference_start = 0
    for i in range(len(parts), 0, -1):
        part = parts[i - 1]
        model_reply = classify_reference_list_section(subsection=part)
        try:
            reply = json.loads(model_reply)
            print(reply)
        except json.decoder.JSONDecodeError as e:
            print(f"Model did something weird when checking subsection is a reference list!\n{model_reply}")
            print("Trying again for now...")
            reply = json.loads(classify_reference_list_section(subsection=part))
        if reply['exists'] == 'true':
            reference_list = f"{part} " + reference_list
            reference_start = i - 1

    text = "\n".join(parts[:reference_start])

    return text, reference_list


async def async_split_text_references(full_text: str, semaphore: asyncio.Semaphore) -> tuple[str, str]:
    """
    Splits the text into the text and separate reference list.
    :param full_text: PDF text.
    :param semaphore: Limits number of requests to LLM.
    :return: text and reference list.
    """
    # Try splitting the text on some key words
    full_text = full_text.replace("#", ' ').replace("*", ' ')
    REF_HEADERS = r'(?mi)^\s*(?:\d+\s*[\.\)]?\s*)?\s*(?:references|reference\s+list|reference|bibliography|works\s+cited|appendix|appendices|glossary)\s*[\.:]?\s*$'
    parts = re.split(REF_HEADERS, full_text, flags=re.IGNORECASE)
    # If there's exactly two parts, then this is likely the text and reference list
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    # If there is only one part, then the split didn't work.
    elif len(parts) == 1:
        raise Exception(f"Splitting text and reference list went wrong! Did not split!\n{parts}")

    client, model = async_client()

    async def class_section(subsection, sem):
        async with sem:
            return await async_classify_reference_list_section(subsection=subsection, client=client, MODEL=model)

    tasks = []
    for i in range(len(parts)):
        string_format = f"TEXT_ID_{i}::: [{parts[i]}]"
        tasks.append(asyncio.create_task(class_section(subsection=string_format, sem=semaphore)))

    responses = await asyncio.gather(*tasks)

    text_parts = []
    reference_parts = []
    for reply in responses:
        model_reply = json.loads(reply.replace("TEXT_ID_", ''))
        part = int(model_reply.get("text_id"))
        # If true => This part is part of the reference list
        if model_reply.get("exists") == 'true':
            reference_parts.append(part)
        else:
            text_parts.append(part)

    text_parts.sort()
    reference_parts.sort()

    text = "\n".join([parts[i] for i in text_parts])
    reference_list = "\n".join([parts[i] for i in reference_parts])

    return text, reference_list


def find_citations(sentences: list[str]) -> list[tuple[str, str]]:
    """
    Finds all citations within the list of sentences. Could use some improvemenets, but it's fine.
    :param sentences: list of tokenized sentences.
    :return: List of tuples like (in-text citation, sentence)
    """
    results = []

    # Narrative citations: Smith (1990), Smith et al. (2012)
    narrative_pattern = re.compile(r'\b([a-z]+(?:\s+(?:and|&)\s+[a-z]+|\s+et al\.?)?)\s*\((\d{4})\)')
    # Parenthetical citations: (Smith, 1990; April, 2005)
    parenthetical_pattern = re.compile(r'\((.*?)\)')
    # Numerical citations: [1], [5-10]
    bracket_pattern = re.compile(r'\[\d+(?:\s*-\s*\d+)?(?:\s*,\s*\d+(?:\s*-\s*\d+)?)*\]')

    for sentence in sentences:
        for match in narrative_pattern.findall(sentence):
            # formatted = format_citation(f"({(match[0].replace('&', 'and')).lower()}, {match[1]})")
            formatted = create_author_in_text_citation(author=[(match[0].replace('&', 'and')).lower()], year=match[1])
            results.append((formatted, sentence))

        for group in parenthetical_pattern.findall(sentence):
            # Split multiple citations separated by ;
            parts = group.split(';')
            for part in parts:
                match = re.search(r"\b([a-zA-Z.'-]+(?:\s*(?:&|and)\s*[a-zA-Z]+)*(?:\s*et\s+al\.?)?)\s*,\s*(\d{4})\b", part.strip(), re.IGNORECASE)
                if match:
                    # results.append(match.groups())
                    match = match.groups()
                    # formatted = format_citation(f"({(match[0].replace('&', 'and')).lower()}, {match[1]})")
                    formatted = create_author_in_text_citation(author=[(match[0].replace('&', 'and')).lower()],year=match[1])
                    results.append((formatted, sentence))

        # Numerical stuff
        for match in bracket_pattern.findall(sentence):
            match = match.replace('[', '').replace(']', '')
            parts = match.split(',')

            for part in parts:
                part = part.strip()

                # Case 1: range like 61-63
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    if start < end:
                        for num in range(start, end + 1):
                            results.append((f"[{num}]", sentence))
                # Case 2: single number
                elif part.isdigit():
                    results.append((f"[{part}]", sentence))

    return list(set(results))


def extract_facts_LLM_code(subset: dict, facts: dict) -> dict[str, tuple[str, str, str]]:
    """
    Code for calling the LLM to extract facts. The initial round and retries should work the same way, hence use this function.
    :param subset: Subset of citations.
    :param facts: All current facts that have been extracted.
    :return: Updated dictionary of facts.
    """
    # Create string format to pass to the LLM
    # try:
    # string_format = "\n".join([f"ID_{k}:: [{subset.get(k)[1]}]" for k in subset.keys()])
    subset_sentences = []
    keys = []
    for k in subset.keys():
        sent = subset.get(k)
        if sent is None:
            print(sent)
            continue
        else:
            sent = sent[1]
        subset_sentences.append(sent)
        keys.append(k)
    # Something went wrong and there are a bunch of None types instead of sentences.
    if len(keys) == 0:
        print(f"No keys found\n{subset}")
        return facts
    strings = [f"ID_{keys[i]}::[{subset_sentences[i]}" for i in range(len(keys))]
    string_format = "\n".join(strings)
    # except TypeError as e:
    #     print(f"{subset}\n{e}")
    # TODO: Async this!
    f = extract_fact(string_format)
    split = f.replace("ID_", "")
    parsed = json.loads(split)

    for p in parsed:
        # If fact was found, check the ID doesn't already exist. If not, add it to the dictionary of facts
        try:
            if p['status'] == 'ok':
                if int(p['id']) not in subset.keys():
                    print(f"ID not present: {p['id']}\n{p}\nSubset keys: {subset.keys()}")
                    continue
                citation_id, sentence = subset.get(int(p['id']))
                if p['id'] in facts.keys():
                    print(
                        f"ERROR! ID: {p['id']} ALREADY EXISTS!\n{p['fact']} ALREADY STORED: {subset.get(int(p['id']))[1]}")
                if p['fact'] is None:
                    print(f"No fact present:{p}")
                facts[p['id']] = (citation_id, p['fact'], sentence)
        except Exception as e:
            print(f"Something went wrong in extracting facts: {p}\n{e}")
        # else:
        # Do nothing if there was an error
        # print(f"ID: {p['id']} NO FACT FOUND: {subset.get(int(p['id']))[1]}")
        # facts[p['id']] = (p['id'], dic.get(p['id']))
    return facts


def extract_facts(citations: list[tuple[str, str]], BATCH: int = 25, RETRY_LIMIT: int = 2) -> tuple[
    dict[str, tuple[str, str, str]], list]:
    """
    Extract the facts from the citations.
    :param citations: List of citations => [(id, Sentence)] or [((Author, Year), Sentence)] => [("in-text citation","sentence")]
    :param BATCH: How many citations to use in one batch.
    :param RETRY_LIMIT: number of times to retry finding a fact from the initial failure.
    :return:Facts dictionary {"made up id": ("citation", "fact sentence", "original sentence")
    """
    # TODO: NO PARSING!
    print("NOT parsing for facts!")
    no_parsing = {}
    for i in range(len(citations)):
        citation = citations[i]
        no_parsing[i] = (citation[0].lower(), citation[1], citation[1])
    return no_parsing, []



def old_extract_references(filename: str, reference_string: str) -> tuple[
    dict[str, str], list[dict], dict[str, str]]:
    # references = extract_references_from_string("\n".join(reference_string))
    references = extract_references_from_string(reference_string)
    # references = []
    # for potential_reference_string in reference_strings:
    #     potential_reference_string = potential_reference_string.lower().replace('-', '')
    #     found_references = extract_references_from_string(potential_reference_string)
    #     for found_ref in found_references:
    #         if 'year' in found_ref.keys() and ('linemarker' in found_ref.keys() or 'author' in found_ref.keys()):
    #             references.append(found_ref)

    # references = extract_references_from_string(reference_list_text)
    # print(f"Found references!\n{references}")
    all_refs = {}
    title = {}
    invalid = []
    for ref in references:
        year = "Unknown"
        in_text_citation = ""
        if 'year' in ref.keys():
            year = ref['year'][0]
        if 'linemarker' not in ref.keys():
            bracket_pattern = re.compile(r'\[([^\]]+)\]')
            matches = bracket_pattern.findall(ref['raw_ref'][0])
            if len(matches) == 1:
                number = matches[0]
                in_text_citation = f"[{number}]"
            else:
                author = ref['author']
                string_author = (" ".join(author)).lower().replace(".", "").replace(", ", " ")
                only_last_names = re.sub(r'\b[a-zA-Z]\b', '', string_author)
                list_authors = [x for x in only_last_names.split(" ") if x != '']
                if len(list_authors) == 1:
                    in_text_citation = f"({list_authors[0]}, {year})"
                elif len(list_authors) == 2:
                    in_text_citation = f"({list_authors[0]} and {list_authors[1]}, {year})"
                else:
                    in_text_citation = f"({list_authors[0]} et al., {year})"
        else:
            number = ref['linemarker'][0].replace(".", '').replace("-", '').replace('#', '').replace('[', '').replace(']',
                                            '').replace('(', '').replace(')', '')
            try:
                number = int(number)
            except Exception:
                print(f"Number: {number} not a number!")
                invalid.append(ref)
            in_text_citation = f"[{number}]"

        all_refs[in_text_citation] = ref['raw_ref'][0]

        if 'title' in ref.keys():
            title[in_text_citation] = ref['title'][0]
        elif 'misc' in ref.keys():
            # title[in_text_citation] = ref['misc'][0]
            # Remove everything detected from the raw ref
            raw = ref['raw_ref'][0]
            for key in ref.keys():
                if key != 'raw_ref':
                    for x in ref[key]:
                        raw = raw.replace(x,'')
            if raw == '' or len(raw.replace(' ', '')) == 0:
                title[in_text_citation] = ref['misc'][0]
            title[in_text_citation] = raw
        else:
            title[in_text_citation] = None

    return all_refs, invalid, title




def extract_references(reference_string: str) -> tuple[dict[str, str], list[dict], dict[str, str]]:
    # references = extract_references_from_string("\n".join(reference_string))
    reference_string = reference_string.replace("- ", "")
    # references = extract_references_from_string(reference_string)
    references = []
    for potential_reference_string in reference_string.split("\n"):
        potential_reference_string = potential_reference_string.lower().replace('-', '')
        found_references = extract_references_from_string(potential_reference_string)
        for found_ref in found_references:
            found_author = ""
            if "author" not in found_ref.keys():
                year = ""
                if "year" not in found_ref.keys():
                    year = re.findall(r'\((\d{4})\)', found_ref["raw_ref"][0])
                    if type(year) == list and len(year) > 0:
                        year = year[0]
                else:
                    year = found_ref['year'][0]
                if year != "" and year != []:
                    author = found_ref["raw_ref"][0].split(f"({year})")[0]
                    # found_author = find_author([author])
                    # Parse out one letter and .
                    found_author = author
                    found_ref["author"] = [found_author]
                    if "year" not in found_ref.keys():
                        found_ref["year"] = [year]
            if 'year' in found_ref.keys() and (
                    'linemarker' in found_ref.keys() or 'author' in found_ref.keys() or found_author != ""):
                references.append(found_ref)

    # references = extract_references_from_string(reference_list_text)
    # print(f"Found references!\n{references}")
    all_refs = {}
    title = {}
    invalid = []
    for ref in references:
        try:
            year = "Unknown"
            in_text_citation = ""
            if 'year' in ref.keys():
                year = ref['year'][0]
            if 'linemarker' not in ref.keys():
                bracket_pattern = re.compile(r'\[([^\]]+)\]')
                matches = bracket_pattern.findall(ref['raw_ref'][0])
                if len(matches) == 1:
                    number = matches[0]
                    in_text_citation = f"[{number}]"
                else:
                    author = ref['author']
                    in_text_citation = create_author_in_text_citation(author=author, year=year)
            else:
                number = ref['linemarker'][0].replace(".", '').replace("-", '').replace('#', '').replace('[',
                                                                                                         '').replace(
                    ']',
                    '').replace('(', '').replace(')', '')
                try:
                    number = int(number)
                except Exception:
                    print(f"Number: {number} not a number!")
                    invalid.append(ref)
                in_text_citation = f"[{number}]"

            all_refs[in_text_citation] = ref['raw_ref'][0]

            if 'title' in ref.keys():
                title[in_text_citation] = ref['title'][0]
            elif 'misc' in ref.keys():
                # title[in_text_citation] = ref['misc'][0]
                # Remove everything detected from the raw ref
                raw = ref['raw_ref'][0]
                for key in ref.keys():
                    if key != 'raw_ref':
                        for x in ref[key]:
                            remove_me = x
                            if "doi:" in x:
                                remove_me = remove_me.replace("doi:", "")
                            raw = raw.replace(remove_me, '')
                if raw == '' or len(raw.replace(' ', '')) == 0:
                    title[in_text_citation] = ref['misc'][0]
                raw=raw.replace("()","")
                title[in_text_citation] = raw
            else:
                title[in_text_citation] = None
        except Exception as e:
            print(f"Error extracting a reference: {ref}\n{e}")
            continue

    return all_refs, invalid, title


async def validate_in_text_style(references: dict[str, str], facts: dict[str, tuple[str, str, str]],
                                 semaphore: asyncio.Semaphore):
    """
    Validate the in-text citations are actually in the correct style.
    :param references: dictionary of references {"in-text citation": "reference from reference list"}
    :param facts: Facts dictionary (should contain tuple with original sentences)
    :param semaphore: Semaphore to limit tasks.
    :return: Reference list with corrected in-text citations. {"in-text citation": "reference from reference list"}
    """
    example_in_text_citations = " ".join([facts.get(key)[0] for key in facts.keys()])

    client, model = async_client()

    async def run_validation(ref, string_ref, string_ref_format, semaphore):
        async with semaphore:
            return (ref, await validate_in_text_citations(citation=string_ref, reference=string_ref_format,
                                                          example_in_text_citations=example_in_text_citations,
                                                          client=client, MODEL=model))

    tasks = []
    for ref in references.keys():
        string_citation_format = f"previous_citation: {ref}"
        string_ref_format = str(references.get(ref))
        tasks.append(asyncio.create_task(run_validation(ref, string_citation_format, string_ref_format, semaphore)))

    responses = await asyncio.gather(*tasks)

    for tup in responses:
        original_ref = tup[0].lower()
        model_reply = tup[1]
        try:
            json_text = json.loads(model_reply.lower())
            new_key = json_text.get("new_key")
            # If new_key is the same as the original, then it's fine.
            # Else, copy the contents of the original reference and then delete the original.
            if new_key != original_ref:
                references[new_key] = references.get(original_ref)
                del references[original_ref]
        except json.decoder.JSONDecodeError:
            print(f"Something went wrong with validating!\nOriginal Reference: {original_ref}\n{model_reply}")
            # If something went wrong, just continue as normal
            continue
    return references


def group_facts(facts: dict, references: dict):
    """
    Groups facts with the same in-text citation.
    :param facts: Dictionary of facts.
    :param references: Reference list.
    :return: Dictionary of {"in-text citation": ['fact_id']]}
    """
    # Go through the facts dictionary and group them
    facts_groups = {}
    for key in facts.keys():
        tup = facts.get(key)
        # ref_id = format_citation(citation=tup[0])
        ref_id = tup[0]

        if ref_id not in facts_groups.keys():
            facts_groups[ref_id] = [key]
        else:
            facts_groups[ref_id].append(key)

    # Filter all missing reference ids
    print(f"Length of facts: {len(facts)}")
    print(f"Length of facts groups: {len(facts_groups)}")
    print(f"Length of references: {len(references)}")
    missing = [ref_id for ref_id in references.keys() if ref_id not in facts_groups.keys()]

    # Normalise the missing reference ids and check if they match with s reference key
    unmatched = []
    for m in missing:
        norm = normalise(m)
        match = process.extractOne(norm, facts_groups.keys())
        # If it's a 90% match, use the reference key as the id
        if match:
            if "(" in m and "(" in match[0]:
                m_year = get_year(m)
                match_year = get_year(match[0])
                if m_year != match_year:
                    continue
        if match[1] > 90:
            fact_list = facts_groups.get(m)
            if fact_list:
                if match[0] in facts_groups.keys():
                    facts_groups[match[0]].extend(fact_list)
                    del facts_groups[m]
                else:
                    facts_groups[m] = fact_list
        else:
            # Otherwise, add it to the unmatched pile
            unmatched.append(m)

    # Use an LLM to match the rest of the IDs. Take one sentence from the group and the reference keys as input.
    # Add ids to the unmatched ones
    # Ask LLM to match the unmatched exactly using the sentence as reference. (It can search the sentence for the correct id)
    # unmatched_dic = {}
    # unmatched_set = []
    # sentences = {}
    # for i in range(len(unmatched)):
    #     u = unmatched[i]
    #     ids = facts_groups.get(u)
    #     row = facts.get(ids[0])
    #     citation = row[0]
    #     # f = row[1]
    #     sentence = row[-1]
    #     sentences[citation] = sentence
    #     # Save the citation to the ID we'll use
    #     unmatched_dic[i] = citation
    #     unmatched_set.append(f"UNMATCHED_REFERENCE_ID_{i}::[{citation}]::SENTENCE::[{sentence}]")
    #
    # ref_keys = list(references.keys())
    # known_refs_dic = {}
    # known_refs_set = []
    # for j in range(len(ref_keys)):
    #     known_refs_dic[j] = ref_keys[j]
    #     known_refs_set.append(f"KNOWN_REFERENCE_ID_{j}::[{ref_keys[j]}]")
    #
    # unmatched_string_format = "\n".join(unmatched_set)
    # known_refs_string_format = "\n".join(known_refs_set)
    #
    # model_reply = match_references(unmatched=unmatched_string_format, known_set=known_refs_string_format)
    # matches = json.loads(model_reply.replace('UNMATCHED_REFERENCE_ID_', '').replace("KNOWN_REFERENCE_ID_", ''))
    #
    problems = []
    # for m in matches:
    #     citation_id = unmatched_dic.get(int(m['unmatched_id']))
    #     if m['matched_known_id'] != 'NO_MATCH':
    #         matched_known_id = m['matched_known_id']
    #         if '_' in matched_known_id:
    #             matched_known_id = int(matched_known_id.split('_')[-1])
    #         reference_id = known_refs_dic.get(matched_known_id)
    #         citation = format_citation(citation=citation_id)
    #
    #         facts_list = facts_groups.get(citation)
    #         if reference_id in facts_groups.keys():
    #             facts_groups[reference_id].extend(facts_list)
    #             del facts_groups[citation]
    #         else:
    #             facts_groups[reference_id] = facts_list
    #     else:
    #         problems.append((citation_id, sentences.get(citation_id)))

    return facts_groups, problems

async def extract_reference_paper_text(reference_pdfs_path: str) -> dict[str, str]:
    """
    Search provided path to a folder containing PDFs for the referenced paper.
    :param reference_pdfs_path: Path to folder containing referenced PDFs.
    :return: Dictionary like {"filename": "PDF text"}
    """
    papers = {}
    for file in os.listdir(reference_pdfs_path):
        if file.endswith(".pdf"):
            text, _ = extract_text(filename=f"{reference_pdfs_path}/{file}")
            papers[file.replace(".pdf", "")] = text
    return papers


async def match_input_ref_papers(papers: dict[str, str], reference_list: dict[str, str], titles: dict[str, str],
                                 semaphore: asyncio.Semaphore) -> tuple[dict, dict[str, str], list[int]]:
    """
    Match the input reference papers to the references in the reference list.
    :param papers: Input reference papers {"filename" : "text"}
    :param reference_list: Dictionary of references {"in_text_style": "full reference"}
    :param titles: Dictionary of titles {"in_text_style": "reference title"} may or may not actually contain title.
    :param semaphore: Semaphore to limit number of calls.
    :return: Matches {in_text_style: paper text}
    """
    matches = {}  # In-text citation as key, paper text as value
    in_text_citation_to_paper_filename = {}  # in_text_citation as key, filename as value

    # Lower case everything
    papers = {k.lower(): v for k, v in papers.items()}
    titles = {k: v.lower() for k, v in titles.items() if v is not None}

    # Remove whitespace
    reversed_titles = {v: k for k, v in titles.items()}
    for unmatched_filename in papers.keys():
        filename_no_whitespace = unmatched_filename.replace(" ", '')
        filename_no_whitespace = re.sub(r'[\\/:*?"<>|]', '', filename_no_whitespace)
        # Reference title could be the title itself, or have the author names and publisher etc. in it.
        for reference_title in reversed_titles.keys():
            # Remove characters windows doesn't let you have
            reference_no_whitespace = reference_title.replace(" ", '')
            reference_no_whitespace = re.sub(r'[\\/:*?"<>|]', '', reference_no_whitespace)
            if filename_no_whitespace in reference_no_whitespace:
                matches[reversed_titles.get(reference_title)] = papers[unmatched_filename]
                in_text_citation_to_paper_filename[reversed_titles.get(reference_title)] = unmatched_filename

    titles_subset = {k: v for k, v in titles.items() if k not in matches.keys()}
    for in_text_citation, reference_title in titles_subset.items():
        # Skip if title is missing or empty
        if not reference_title:
            continue

        if reference_title in papers:
            matches[in_text_citation] = papers[reference_title]
            in_text_citation_to_paper_filename[in_text_citation] = reference_title
        else:
            # Find the best fuzzy match with a score >= 90
            best_match = process.extractOne(reference_title, list(papers.keys()), score_cutoff=90)

            if best_match:
                # best_match is a tuple/object: (matched_filename, score, index)
                matched_filename = best_match[0]
                matches[in_text_citation] = papers[matched_filename]
                in_text_citation_to_paper_filename[in_text_citation] = matched_filename

    # Use LLM to find the reference match
    ref_key_list = [key for key in reference_list.keys() if key not in matches.keys()]
    reference_keys = {}
    ref_string_format = ""
    for i in range(len(ref_key_list)):
        # reference[i][0] is the in-text citation, [1] is the actual reference text
        # reference key is in-text citation, value is the actual reference text.
        reference_keys[ref_key_list[i]] = reference_list.get(ref_key_list[i])
        reference_keys[i] = (ref_key_list[i], reference_list.get(ref_key_list[i]))
        ref_string_format += f"REFERENCE_ID_{i}:: [{reference_list.get(ref_key_list[i])}]\n"

    client, model = async_client()

    async def limited_extract(sem, references, reference_text):
        async with sem:
            return await extract_title_from_provided_pdf(references=references, reference_text=reference_text,
                                                         client=client, MODEL=model)

    # Async match references
    tasks = []
    paper_keys = {}
    count = 0
    papers_subset = [key for key in papers.keys() if key not in in_text_citation_to_paper_filename.values()]
    for filename in papers_subset:
        # text = papers.get(filename)
        paper_keys[count] = filename
        # Naive assumption text will be within the first X characters
        string_format = f"PAPER_ID_{count}\nFilename: {filename}"
        count += 1
        tasks.append(asyncio.create_task(
            limited_extract(sem=semaphore, references=ref_string_format, reference_text=string_format)))

    responses = await asyncio.gather(*tasks)

    found_paper_keys = set()
    for model_reply in responses:
        try:
            json_text = json.loads(
                model_reply.replace("REFERENCE_ID_", "").replace("PAPER_ID_", "").replace('null', 'None'))
            if json_text.get('matched_reference_id') is None or json_text.get('paper_id') is None or json_text.get(
                    'matched_reference_id') == 'None' or json_text.get('paper_id') == 'None':
                continue
            # reference = reference_keys.get(json_text.get('matched_reference_id'))
            try:
                matched_id = int(json_text.get('matched_reference_id'))
                in_text_citation = reference_keys.get(matched_id)[0]
                paper_id = int(json_text.get('paper_id'))
                paper_filename = paper_keys.get(paper_id)  # Without .pdf at the end
                paper = papers.get(paper_filename)  # Gets text
            except Exception as e:
                print(f"Matching stuff went wrong... again...\n{json_text}")
                continue
            # if in-text reference hasn't been matched, add it to the matches
            if paper_id not in found_paper_keys and in_text_citation not in matches.keys():
                matches[in_text_citation] = paper
                found_paper_keys.add(paper_id)
                in_text_citation_to_paper_filename[in_text_citation] = f"{paper_filename}.pdf"
            else:
                print(
                    f"######Duplicate match found!\n{json_text}\nIn-text Citation: {in_text_citation}\nPaper Filename: {paper_filename}\nMatched keys: {matches.keys()}\n")
                # TODO: Add another iteration for the duplicates! Remove the already matched ones first!
        except json.decoder.JSONDecodeError as e:
            print(
                f"Failed to JSONify model reply in matching input paper to reference list\nModel Reply: {model_reply}\n{e}")

    leftover_references = [ref for ref in range(len(ref_key_list)) if ref not in found_paper_keys]
    return matches, in_text_citation_to_paper_filename, leftover_references


async def extract_paragraph_references(paragraphs: list[str], semaphore: asyncio.Semaphore, BATCH_SIZE: int = 15) -> \
        list[tuple[str, str]]:
    """
    Extract sentences from paragraphs with one reference for the whole paragraph.
    :param paragraphs: list of paragraphs.
    :param semaphore: Limit the number of calls to the LLM.
    :param BATCH_SIZE: Number of paragraphs to include at a time.
    :return: list of ("in_text_citation", "sentence")
    """
    client, model = async_client()

    async def limited_extract(paragraph, sem):
        async with sem:
            return await extract_referenced_paragraphs(paragraphs=paragraph, client=client, MODEL=model)

    tasks = []
    string_format = [f"PARAGRAPH_ID_{i}::: [{paragraphs[i]}]" for i in range(len(paragraphs))]
    for i in range(0, len(paragraphs), BATCH_SIZE):
        subset_string_format = "\n".join(string_format[i:i + BATCH_SIZE])
        tasks.append(asyncio.create_task(limited_extract(paragraph=subset_string_format, sem=semaphore)))

    responses = await asyncio.gather(*tasks)

    valid_paragraphs = {}
    for model_reply in responses:
        try:
            json_text = json.loads(model_reply.replace("PARAGRAPH_ID_", ""))  # Is a list of dictionary
            for dic in json_text:
                in_text_citation = dic.get('citation')
                valid_paragraphs[in_text_citation] = paragraphs[int(dic.get("paragraph_id"))]
        except json.decoder.JSONDecodeError:
            print(f"Couldn't JSONify model reply!\nModel Reply: {model_reply}")

    # Reference list of [(in_text_citation, sentence)]
    citation_list = []
    for citation in valid_paragraphs.keys():
        paragraph = valid_paragraphs.get(citation)
        sentences = join_citations(tokenize_sentences(paragraph))
        for s in sentences:
            # Nope, adding citaiton does fuck things
            # Add citation to the sentence, in case leaving it messes something up down the line
            # citation_list.append((citation, f"{s} {citation}"))
            formatted = format_citation(citation.replace("(","").replace(")",""))
            if formatted != "":
                citation_list.append((formatted, s))

    return citation_list
