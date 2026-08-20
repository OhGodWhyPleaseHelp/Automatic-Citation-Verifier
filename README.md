# Automatic-Citation-Verifier

This is the repository for the Automatic Citation Verification System.

The repository here contains both the code used and the evaluation set.

The code here was not initially intended for public release, so the code presented as is, which may or may not have sufficient documentation to understand each function. 

The file RunEvaluations.py is intended to run the evaluation set, whereas RunNewDocument.py runs all justifications for a given input document and references. You will need to change the path to the input and reference PDFs for this folder.
In Example, you can see an example of how the input and reference documents should be organised. Please note: the reference PDF filenames should match the titles of the reference list in the input PDF.


#### Requirements

To run the code, you will need to install the FAISS vector database package, torch and run the following command in your project environment:

```bash
pip install -r requirements.txt
```