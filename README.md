# Multimodal Agentic RAG

### Table of contents
* [Overview](###Overview)
* [Features](###Features)
* [Installation](###Installation)
* [Usage](###Usage)
* [Perspectives](###Perspectives)

### Overview
Retrieval-Augmented Generation (RAG) is an advanced AI framework that combines the retrieval capabilities of information retrieval systems (e.g., encoders and vector databases) with the generative power of large language models (LLMs). By leveraging external knowledge sources, such as user-provided documents, RAG delivers accurate, context-aware answers and generates factual, coherent responses.

Traditional RAG systems primarily rely on static chunking and retrieval mechanisms, which may struggle to adapt dynamically to complex, multimodal data. To address this limitation, this project introduces an agentic approach to chunking and retrieval, adding significant value to the RAG process.

### Features
**- chunking (Semantic or Agentic):** \
Semantic chunker split documents into semantically coherent, meaningful chunks. 
Agentic chunker goes further and simulates human judgment of text segmentation: start at the beginning of a document, group sentences based on context and topic, and continue this process iteratively until the entire document is segmented. \
(For more info: [Agentic Chunking: Enhancing RAG Answers for Completeness and Accuracy](https://gleen.ai/blog/agentic-chunking-enhancing-rag-answers-for-completeness-and-accuracy/)).\
**- Image and table detection:** \
Detecting images and tables using PyMuPDF and img2table respectively.\
**- Summarizing images and tables:** \
Using a multimodal LLM (eg. gemini-1.5-flash), create a text description of each image and table.\
**- Embedding:** \
Embed chunks, images and tables summaries using "text-embedding-004" model.\
**- Retrieval (Semantic or Agentic):** \
For a given query: semantic retrieval focuses on embedding-based similarity searches to retrieve information. Agentic retrieval includes 4 steps, following ReAct process: \
(1). Query rephrasing, with regards to chat history \
(2). semantic retrieval \
(3). Assess whether the retrieved documents are relevant and sufficient to answer the query \
(4). Accordingly, either use the retrieved documents or web search engine to generate a relevant, sufficient and factual answer.  

### Installation
To run the app locally, the following steps are necessary:
- Clone The repo:
```bash
git clone https://github.com/AhmedAl93/multimodal-agentic-RAG.git
cd multimodal-agentic-RAG/
```
- Install the required python packages:
```bash
pip install -r requirements.txt
```
- Set up the environment variables in the .env file:
```bash
LLAMA_CLOUD_API_KEY=<Your LLAMACLOUD API KEY>
GOOGLE_API_KEY=<Your Google API KEY>
TAVILY_API_KEY=<Your Tavily API KEY>
```

### Usage
1. Process input document(s):

Run the following command:
```bash
python main.py --InputPath <path> --parser_name <parser> --chunking_strategy <chunking> --retrieval_strategy <retrieval>
```
Here are more details about the inputs:
```bash
--InputPath: 'Directory path containing files to be processed, or a single file path'
--parser_name: 'Specify the name of the parser to use for document processing. Possible values: ["LlamaParse", "pymupdf4llm"]'
--chunking_strategy: 'Define the chunking strategy to apply when processing documents. Possible values: ["semantic", "agentic"]'
--retrieval_strategy: 'Specify the retrieval strategy for querying indexed documents. Possible values:["semantic", "agentic"]'
```
Currently, only PDF files are supported. So if the input directory contains x PDFs, x files will be processed.

2. Provide queries:
In the terminal, you can provide multiple queries and get relevant answers.

### Perspectives
In the near future, I plan to work on the following features:
- Support other file types than PDF
- Performance Evaluation for different chunking and retrieval strategies
- Support open-source LLMs
- Support other Vector DBs providers
- Assess and test new concepts: GraphRAG, ... 
- Cloud deployment