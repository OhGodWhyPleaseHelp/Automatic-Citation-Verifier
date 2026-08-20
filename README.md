# Automatic-Citation-Verifier

This is the repository for the Automatic Citation Verification System.

The repository here contains both the code used and the evaluation set.

The code here was not initially intended for public release, so the code presented as is, which may or may not have sufficient documentation to understand each function. 


#### Requirements

To run the code, you will need to install the [FAISS vector database](https://github.com/facebookresearch/faiss) package, [torch](https://pytorch.org/get-started/locally/) and run the following command in your project environment:

```bash
pip install -r requirements.txt
```

#### Run

The file [RunEvaluations](https://github.com/OhGodWhyPleaseHelp/Automatic-Citation-Verifier/blob/main/RunEvaluations.py) is intended to run the evaluation set, whereas [RunNewDocument](https://github.com/OhGodWhyPleaseHelp/Automatic-Citation-Verifier/blob/main/RunNewDocument.py) runs all justifications for a given input document and references. You will need to change the path to the input and reference PDFs for this folder.

In [Example](https://github.com/OhGodWhyPleaseHelp/Automatic-Citation-Verifier/tree/main/Example), you can see an example of how the input and reference documents should be organised. Please note: the reference PDF filenames should match the titles of the reference list in the input PDF.
