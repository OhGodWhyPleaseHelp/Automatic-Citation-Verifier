import time
from fileinput import filename
import traceback
import shutil

from ParseInputPDF import *
from RAGSystem import *
from VectorlessRAG import vectorless_rag_verify_facts
from SOM_Justification import *
from QraftLite_Justification import *
from utils import init_logger, close_logger

all_metrics = {}
eval_metrics = {}


async def process_paragraphs(filename: str, semaphore: asyncio.Semaphore) -> list[tuple[str, str]]:
    """
    Asynchronously processes the paragraphs of the PDF.
    :param filename: Input PDF.
    :param semaphore: Limit on calling the LLM.
    :return:
    """
    print("Processing paragraphs!")
    paragraphs, _ = find_paragraphs(filename)
    paragraphs = [paragraph.replace("_","") for paragraph in paragraphs]
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
    sentences = join_citations(tokenize_sentences(text=text))
    citation_list = find_citations(sentences=sentences)
    return citation_list, reference_list


def write_to_pdf(verification_type, verdicts, filename, reference_pdfs, output_path, report, previous_time):
    global all_metrics
    global eval_metrics
    evaluation_metrics, all_citation_metrics = evaluate(verdicts=verdicts, verification_type=verification_type,
                                                        reference_pdf_path=reference_pdfs, output_path=output_path)
    eval_metrics[verification_type] = evaluation_metrics
    all_metrics[verification_type] = all_citation_metrics

    highlight_text(pdf_path=filename, verdicts=verdicts, verification_type=verification_type, output_path=output_path)
    new_time = time.time()
    report.write(
        f"{verification_type} - Evaluation citations only:\nAccuracy: {evaluation_metrics[0]:.2f}\nPrecision: {evaluation_metrics[1]:.2f}\nRecall: {evaluation_metrics[2]:.2f}\nF1-Score: {evaluation_metrics[3]:.2f}\n")
    report.write(
        f"{verification_type} - All citations:\nAccuracy: {all_citation_metrics[0]:.2f}\nPrecision: {all_citation_metrics[1]:.2f}\nRecall: {all_citation_metrics[2]:.2f}\nF1-Score: {all_citation_metrics[3]:.2f}\n")
    report.write(f"Time: {new_time - previous_time:.2f}s or {(new_time - previous_time) / 60:.2f} minutes\n\n")


async def run_rag_stuff(verification_type: str, faiss_indexes, chunk_stores, groups, facts, semaphore, filename,
                        reference_pdfs, output_path, citation_to_filename, report, previous_time):
    MAX_CHUNKS = 20
    ENHANCEMENT_CHUNKS = 15
    print(f'{verification_type}!')
    if verification_type == 'Naive RAG':
        verdicts = await naive_RAG_query(faiss_indexes=faiss_indexes, chunk_stores=chunk_stores, groups=groups,
                                         facts=facts, semaphore=semaphore, max_chunks=MAX_CHUNKS)
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
        # set_token_logger(log_usage)
        verdicts = await vectorless_rag_verify_facts(groups=groups, facts=facts,
                                                     citation_to_paper_names=citation_to_filename,
                                                     reference_pdf_path=reference_pdfs,
                                                     semaphore=semaphore)
        # shutdown_token_logger()
    with open(f"{output_path}/{verification_type}_verdicts.json", "w") as f:
        json.dump(verdicts, f)
    write_to_pdf(verification_type, verdicts, filename, reference_pdfs, output_path, report, previous_time)
    # evaluation_metrics, all_citation_metrics = evaluate(verdicts=verdicts, verification_type=verification_type,
    #                                                     reference_pdf_path=reference_pdfs, output_path=output_path)
    # highlight_text(pdf_path=filename, verdicts=verdicts, verification_type=verification_type, output_path=output_path)
    new_time = time.time()
    # report.write(
    #     f"{verification_type} - Evaluation citations only:\nAccuracy: {evaluation_metrics[0]:.2f}\nPrecision: {evaluation_metrics[1]:.2f}\nRecall: {evaluation_metrics[2]:.2f}\nF1-Score: {evaluation_metrics[3]:.2f}\n")
    # report.write(
    #     f"{verification_type} - All citations:\nAccuracy: {all_citation_metrics[0]:.2f}\nPrecision: {all_citation_metrics[1]:.2f}\nRecall: {all_citation_metrics[2]:.2f}\nF1-Score: {all_citation_metrics[3]:.2f}\n")
    # report.write(f"Time: {new_time - previous_time:.2f}s or {(new_time - previous_time) / 60:.2f} minutes\n\n")

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
        # print("Task started!")
        async with outer_semaphore:
            # print("Semaphore Acquired!")
            try:
                result = await justifier_object.verify_citation(claim=claim, initial_verdict=initial_verdict)
                # print(f"[TASK DONE] {fact_id}")
                return (fact_id, row, result)
            except Exception as e:
                print(f"[TASK ERROR] {fact_id}: {e}")
                raise e

    tasks = []
    for paper_id in faiss_indexes.keys():
        fact_group = groups.get(paper_id)

        if len(fact_group) < 5:
            continue

        # Create tasks for all facts in this paper
        for fact_id in fact_group:
            if justification_type == "SOM":
                # print("SOM Justifier selected!")
                # outer_semaphore = asyncio.Semaphore(7)
                justifier = SOM_Justifier(faiss=faiss_indexes, chunks=chunk_stores, paper_id=paper_id,
                                          semaphore=semaphore)
            else:
                # print("QraftLite Justifier selected!")
                # outer_semaphore = semaphore
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
        new_row = (row[0], row[1], row[2], row[3], justification.get('justification'))
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
    reference_string = await get_reference_list(filename=filename, semaphore=semaphore)

    # with open(f"{output_path}\\citations.txt", "w", encoding='utf-8') as f:
    #     f.write(str(citation_list))

    # Extract facts from the sentences.
    # If no fact could be deduced, these sentences are returned in the 'problems' dictionary and labelled as ambiguous.
    # 'problems' structure -> {'generated_id': ('citation_id', 'sentence', 'AMBIGUOUS SENTENCE')
    print("Extracting facts...")

    facts, problems = extract_facts(citations=citation_list, BATCH=5)
    report.write(f"Number of facts found: {len(facts.keys())}\n")

    # Reformat reference list into a dictionary -> contains {"reference_id": "reference"}
    # reference_id should be in-text citation
    # Invalid references have the same format, but it's from a website or something
    print("Extracting references...")
    references_list, invalid_references, titles = old_extract_references(filename=filename,reference_string=reference_string)


    # Match citations to the reference list -> {paper: [citation ids ]}
    print("Grouping facts!")
    groups, group_problems = group_facts(facts=facts, references=references_list)
    report.write(f"Number of ungrouped facts found: {len(group_problems)}\n")

    # Citation to filename is a dict like: {"in_text_citation": "paper_filename.odf"} -> Need to add path to pdf too (reference_pdfs)
    print("Matching reference papers to reference list!")
    references, citation_to_filename, unmatched = await match_input_ref_papers(papers=papers,
                                                                               reference_list=references_list,
                                                                               titles=titles, semaphore=semaphore)

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


async def start(semaphore, filename="Research_Topics.pdf", reference_pdfs="ReferencePDFs", output_path="Output"):
    """
    Call this function to begin parsing the input PDF.
    :param filename: Path to the PDF.
    :param reference_pdfs: Path to the reference article PDFs.
    :return: TBD.
    """
    global all_metrics
    global eval_metrics
    all_metrics = {}
    eval_metrics = {}
    refresh_output_folder(output_path=output_path)
    # Semaphore to limit the number of tasks the LLM can handle
    # Set to 24 as some sequential calls aren't going to use the semaphore
    report = open(f"{output_path}\\report.txt", 'w')
    await init_logger(output_filename=f"{output_path}\\token_usage.log")
    startTime = time.time()

    references, groups, facts, citation_to_filename = await preprocessing(semaphore, filename, reference_pdfs,
                                                                          output_path, report, startTime)

    # RAG stuff
    print("Boring RAG stuff!")
    faiss_indexes, chunk_stores = create_FAISS_indices(references=references)

    previous_time = time.time()
    for verification_type in ['Vectorless RAG']:
        _, previous_time = await run_rag_stuff(verification_type, faiss_indexes, chunk_stores, groups, facts, semaphore,
                                               filename,
                                               reference_pdfs, output_path, citation_to_filename, report, previous_time)
    # NAIVE RAG ONLY
    # for verification_type in ['Vectorless RAG']:
    #     previous_time = await run_rag_stuff(verification_type, faiss_indexes, chunk_stores, groups, facts, semaphore,
    #                                         filename,
    #                                         reference_pdfs, output_path, citation_to_filename, report, previous_time)

    # End of metrics
    report.write("######\nEnd of metrics\n\n")
    # total_tokens, input_tokens, output_tokens = read_token_log(output_filename=f"{output_path}\\token_usage.log")
    total_tokens, input_tokens, output_tokens, methods = all_evaluations_count_tokens(
        token_filepath=f"{output_path}\\token_usage.log")
    create_token_graph(methods=methods, output_filepath=output_path)
    print(f"Total tokens: {total_tokens}\nTotal input tokens: {input_tokens}\nTotal output tokens: {output_tokens}\n\n")
    write_token_report(methods=methods, report=report)
    create_evaluation_metrics_graph(data=eval_metrics, output_filepath=f"{output_path}/eval_evaluation_metrics.png")
    # create_evaluation_metrics_graph(data=all_metrics, output_filepath=f"{output_path}/all_evaluation_metrics.png")

    endTime = time.time()
    print(f"\nTotal time: {endTime - startTime:.2f}s")
    print(f"Total time: {(endTime - startTime) / 60:.2f} minutes")
    report.write(f"Total time: {endTime - startTime:.2f}s\n")
    report.write(f"Total time: {(endTime - startTime) / 60:.2f} minutes")
    report.close()


async def run_evaluation():
    sem = asyncio.Semaphore(10)
    try:
        print("Starting Psychology")
        await start(sem, filename="Evaluation set\\Psychology\\input.pdf",
                    reference_pdfs="Evaluation set\\Psychology\\References", output_path="Output\\Chunking\\20 CHUNKS")
    except Exception as e:
        print(f"* ===================== Evaluation for Psychology failed! ===================== *\n{e}")
        traceback.print_exc()
    # try:
    #     print("Starting Biology")
    #     await start(sem, filename="Evaluation set\\Biology\\input.pdf",
    #                 reference_pdfs="Evaluation set\\Biology\\References", output_path="Output\\VectorlessRAG\\Bio\\Cache")
    # except Exception as e:
    #     print(f"* ===================== Evaluation for Biology failed! ===================== *\n{e}")
    #     traceback.print_exc()
    # try:
    #     print("Starting Computer Science")
    #     await start(sem, filename="Evaluation set\\Computer Science\\input.pdf",
    #                 reference_pdfs="Evaluation set\\Computer Science\\References", output_path="Output\\VectorlessRAG\\CS\\Cache")
    # except Exception as e:
    #     print(f"* ===================== Evaluation for Computer Science failed! ===================== *\n{e}")
    #     traceback.print_exc()


if __name__ == "__main__":
    print("Running!")
    asyncio.run(run_evaluation())
    # asyncio.run(run_justification_types())
