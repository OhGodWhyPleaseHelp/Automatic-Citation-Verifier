import time
import traceback

from ParseInputPDF import *
from RAGSystem import *
from VectorlessRAG import vectorless_rag_verify_facts
from SOM_Justification import *
from QraftLite_Justification import *
from utils import init_logger, close_logger


async def process_paragraphs(filename: str, semaphore: asyncio.Semaphore) -> list[tuple[str, str]]:
    """
    Asynchronously processes the paragraphs of the PDF.
    :param filename: Input PDF.
    :param semaphore: Limit on calling the LLM.
    :return:
    """
    print("Processing paragraphs!")
    paragraphs, _ = find_paragraphs(filename)
    return await extract_paragraph_references(paragraphs=paragraphs, semaphore=semaphore)


async def process_sentences(full_text: str, semaphore: asyncio.Semaphore) -> tuple[list[tuple[str, str]], str]:
    """
    Processes the extracted text into the citation list and reference list.
    :param full_text: Extracted text of input PDF.
    :param semaphore: Limit on calling the LLM.
    :return: Citation list and reference list.
    """
    print("Processing sentences!")
    text, reference_list = await async_split_text_references(full_text=full_text, semaphore=semaphore)
    sentences = join_citations(tokenize_sentences(text=full_text))
    citation_list = find_citations(sentences=sentences)
    return citation_list, reference_list


def write_to_pdf(verification_type, verdicts, filename, reference_pdfs, output_path, report, previous_time):
    highlight_text(pdf_path=filename, verdicts=verdicts, verification_type=verification_type, output_path=output_path)
    new_time = time.time()
    report.write(f"Time: {new_time - previous_time:.2f}s or {(new_time - previous_time) / 60:.2f} minutes\n\n")


async def run_rag_stuff(verification_type: str, faiss_indexes, chunk_stores, groups, facts, semaphore, filename,
                        reference_pdfs, output_path, citation_to_filename, report, previous_time):
    MAX_CHUNKS = 10
    ENHANCEMENT_CHUNKS = 15
    print(f'{verification_type}!')
    if verification_type == 'Naive RAG':
        verdicts = await naive_RAG_query(faiss_indexes=faiss_indexes, chunk_stores=chunk_stores, groups=groups,
                                         facts=facts, semaphore=semaphore)
    elif verification_type == 'Naive RAG Enhanced':
        verdicts = await naive_RAG_query(faiss_indexes=faiss_indexes, chunk_stores=chunk_stores, groups=groups,
                                         facts=facts, semaphore=semaphore, chunk_evaluation=True,
                                         max_chunks=MAX_CHUNKS, enhancement_chunks=ENHANCEMENT_CHUNKS)
    elif verification_type == "Quotes RAG":
        verdicts = await quotes_check_fact(faiss_indexes=faiss_indexes, chunk_stores=chunk_stores, groups=groups,
                                           facts=facts, semaphore=semaphore)
    elif verification_type == "Quotes RAG Enhanced":
        verdicts = await quotes_check_fact(faiss_indexes=faiss_indexes, chunk_stores=chunk_stores,
                                           groups=groups,
                                           facts=facts, semaphore=semaphore, chunk_evaluation=True,
                                           max_chunks=MAX_CHUNKS, enhancement_chunks=ENHANCEMENT_CHUNKS)
    else:
        verdicts = await vectorless_rag_verify_facts(groups=groups, facts=facts,
                                                     citation_to_paper_names=citation_to_filename,
                                                     reference_pdf_path=reference_pdfs,
                                                     semaphore=semaphore)

    write_to_pdf(verification_type, verdicts, filename, reference_pdfs, output_path, report, previous_time)
    new_time = time.time()

    return verdicts, new_time


async def justifications(justification_type: str, initial_verdicts, faiss_indexes, chunk_stores, groups, semaphore):
    """
    Run Justification Objects.
    :param justification_type: SOM or QraftLite
    :param initial_verdicts: from Naive RAG
    :param faiss_indexes: FAISS Vector Database
    :param chunk_stores: Chunk storage to match FAISS Vector Database
    :param groups: Grouped fact ids.
    :param semaphore: Semaphore to limit async calls.
    :return: New justifications.
    """
    outer_semaphore = asyncio.Semaphore(8)
    if justification_type != "SOM":
        outer_semaphore = asyncio.Semaphore(24)

    async def run_justification(justifier_object, claim, initial_verdict, fact_id, row):
        async with outer_semaphore:
            try:
                result = await justifier_object.verify_citation(claim=claim, initial_verdict=initial_verdict)
                return (fact_id, row, result)
            except Exception as e:
                print(f"[TASK ERROR] {fact_id}: {e}")
                raise e

    tasks = []
    for paper_id in faiss_indexes.keys():
        fact_group = groups.get(paper_id)
        if fact_group is None:
            continue

        for fact_id in fact_group:
            if justification_type == "SOM":
                justifier = SOM_Justifier(faiss=faiss_indexes, chunks=chunk_stores, paper_id=paper_id,
                                          semaphore=semaphore)
            else:
                justifier = QraftLite(faiss=faiss_indexes, chunks=chunk_stores, paper_id=paper_id, semaphore=semaphore)
            justifier.set_paper_id(paper_id=paper_id)
            row = initial_verdicts.get(fact_id)
            tasks.append(asyncio.create_task(
                run_justification(justifier_object=justifier, claim=row[2], initial_verdict=row[3], fact_id=fact_id,
                                  row=row)))

    results = await asyncio.gather(*tasks)
    verdicts = {}
    # Process results
    for tup in results:
        fact_id = tup[0]
        row = tup[1]
        justification = tup[2]
        if isinstance(justification, Exception):
            print(f"[ERROR] Failed to verify {fact_id}: {justification}")
            continue
        new_row = (justification.get("verdict", row[0]), row[1], row[2], row[3], justification.get('justification'))
        verdicts[fact_id] = new_row

    return verdicts


async def get_reference_list(filename:str, semaphore):
    _, ref = await async_split_text_references(evaluation_extract_texts(filename=filename), semaphore)
    return ref

async def preprocessing(semaphore, filename, reference_pdfs, output_path, report, start_time):
    print("Handling reference papers...")
    handle_reference_paper_task = asyncio.create_task(extract_reference_paper_text(reference_pdfs))

    # Extract text
    print("Getting full text...")
    full_text, _ = await asyncio.to_thread(extract_text, filename=filename, find_reference_strings=False)

    handle_paragraph_task = asyncio.create_task(process_paragraphs(filename=filename, semaphore=semaphore))

    handle_sentence_task = asyncio.create_task(process_sentences(full_text=full_text, semaphore=semaphore))

    papers = await handle_reference_paper_task
    paragraph_citation_list, (citation_list, reference_string) = await asyncio.gather(
        handle_paragraph_task,
        handle_sentence_task
    )
    citation_list.extend(paragraph_citation_list)
    # reference_string = await get_reference_list(filename=filename, semaphore=semaphore)

    print("Extracting facts...")
    facts, problems = extract_facts(citations=citation_list, BATCH=5)
    report.write(f"Number of facts found: {len(facts.keys())}\n")

    print("Extracting references...")
    references_list, invalid_references, titles = extract_references(reference_string=reference_string)

    print("Grouping facts!")
    groups, group_problems = group_facts(facts=facts, references=references_list)
    report.write(f"Number of groups found: {len(groups.keys())}\n")
    report.write(f"Number of ungrouped facts found: {len(group_problems)}\n")
    print("Parsing out 'None' groups")
    before = len(groups)
    with open(f"{output_path}/groups.json", 'a') as f:
        json.dump(groups, f)
    groups = {key: groups.get(key) for key in groups.keys() if groups.get(key)}
    print(f"Removed: {before - len(groups)} groups")
    report.write(f"Number of groups removed: {before - len(groups)}\n")

    print("Matching reference papers to reference list!")
    references, citation_to_filename, unmatched = await match_input_ref_papers(papers=papers,
                                                                               reference_list=references_list,
                                                                               titles=titles, semaphore=semaphore)
    report.write(f"Number of references found: {len(references)}\n")
    print("Removing references with no text...")
    before = len(references)
    with open(f"{output_path}/references.json", 'a') as f:
        json.dump(references, f)
    references = {key: references.get(key) for key in references.keys() if references.get(key)}
    print(f"Removed: {before - len(references)} references")
    report.write(f"Number of references removed: {before - len(references)}\n")

    report.write(f"Number of unmatched papers found: {len(unmatched)}\n")

    end_of_preprocessing_time = time.time()
    end_of_preprocessing_time = end_of_preprocessing_time - start_time
    report.write(
        f"End of preprocessing time: {end_of_preprocessing_time:.2f}s or {end_of_preprocessing_time / 60:.2f} minutes\n\n")
    return references, groups, facts, citation_to_filename


def write_token_report(methods, report):
    for method in methods.keys():
        report.write(
            f"{method}\nTotal tokens: {methods[method]['total']}\nInput tokens: {methods[method]['input']}\nOutput tokens: {methods[method]['output']}\n")


async def start_justification_types(sem, filename="Research_Topics.pdf", reference_pdfs="ReferencePDFs",
                                    output_path="Output"):
    refresh_output_folder(output_path=output_path)
    report = open(f"{output_path}\\report.txt", 'w')
    await init_logger(output_filename=f"{output_path}\\token_usage.log")
    startTime = time.time()
    semaphore = sem

    references, groups, facts, citation_to_filename = await preprocessing(semaphore, filename, reference_pdfs,
                                                                          output_path, report, startTime)
    faiss_indexes, chunk_stores = create_FAISS_indices(references=references)

    verdict_start_time = time.time()
    initial_verdicts, previous_time = await run_rag_stuff('Naive RAG Enhanced', faiss_indexes, chunk_stores, groups, facts,
                                                          semaphore,
                                                          filename, reference_pdfs, output_path, citation_to_filename,
                                                          report, verdict_start_time)

    with open(f"{output_path}\\naive_rag_enhanced_verdicts.json", 'w') as f:
        json.dump(initial_verdicts, f)

    print("Starting SOM Justifications!")
    som_verdicts = await justifications("SOM", initial_verdicts=initial_verdicts, faiss_indexes=faiss_indexes,
                                        chunk_stores=chunk_stores, groups=groups, semaphore=semaphore)
    report.write(f"SOM completed.")
    with open(f"{output_path}\\som_verdicts.json", 'w') as f:
        json.dump(som_verdicts, f)
    write_to_pdf("SOM Justification", som_verdicts, filename, reference_pdfs, output_path, report,
                 previous_time)
    previous_time = time.time()
    print("Starting QraftLite Justifications!")
    ta_justifier = await justifications("TA", initial_verdicts=initial_verdicts, faiss_indexes=faiss_indexes,
                                        chunk_stores=chunk_stores, groups=groups, semaphore=semaphore)
    report.write(f"QraftLite completed.")
    with open(f"{output_path}\\ta_verdicts.json", 'w') as f:
        json.dump(ta_justifier, f)
    write_to_pdf("QraftLite Justification", ta_justifier, filename, reference_pdfs, output_path,
                 report,
                 previous_time)

    report.write("######\nEnd of metrics\n\n")
    close_logger()
    total_tokens, input_tokens, output_tokens, methods = justifications_count_tokens(
        token_filepath=f"{output_path}\\token_usage.log")
    create_token_graph(methods=methods, output_filepath=output_path)
    print(f"Total tokens: {total_tokens}\nTotal input tokens: {input_tokens}\nTotal output tokens: {output_tokens}\n")
    write_token_report(methods=methods, report=report)

    endTime = time.time()
    print(f"\nTotal time: {endTime - startTime:.2f}s")
    print(f"Total time: {(endTime - startTime) / 60:.2f} minutes")
    report.write(f"Total time: {endTime - startTime:.2f}s\n")
    report.write(f"Total time: {(endTime - startTime) / 60:.2f} minutes")
    report.close()


async def run_justification_types():
    sem = asyncio.Semaphore(24)

    evals = [4,5,6,7,8]

    for eval in evals:
        print(f"# *** *** Starting: {eval} *** *** #")
        try:
            await start_justification_types(sem, filename=f"Human Evaluation\\{eval}\\input.pdf",
                                            reference_pdfs=f"Human Evaluation\\{eval}\\References",
                                            output_path=f"Human Evaluation\\{eval}\\Rerun")
        except Exception as e:
            print(f"* ===================== Evaluation for {eval} failed! ===================== *\n{e}")
            traceback.print_exc()
            close_logger()
            raise e


if __name__ == "__main__":
    print("Running!")
    asyncio.run(run_justification_types())
