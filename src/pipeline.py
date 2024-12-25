from .chunkers import Chunker
from .docparser import DocParser
from .imageprocessing import ImageProcessor
from glob import glob
from pathlib import Path
from .doc_qa import QA, AgenticQA, indexing


def list_supported_files(inputPath, supported_extensions= [".pdf"]):
    """
    Lists all supported files in the given input path.
    
    Args:
        inputPath (str): The path where files are located.

    Returns:
        List[str]: A list of file paths with supported extensions.
    """
    # Retrieve all files matching the input path and filter by supported extensions
    file_list = glob(inputPath)
    return [f for f in file_list if Path(f).suffix in supported_extensions]


def pipeline(inputPath, 
             parser_name, 
             chunking_strategy, 
             retrieval_strategy):

    parser= DocParser(parser_name= parser_name)
    chunker= Chunker(chunking_strategy)
    image_processor= ImageProcessor()

    files_list= list_supported_files(inputPath)
    chunks, image_documents= [], []

    for file_path in files_list:
        print("processing started ...")

        text_docs= parser.parsing_function(file_path)
        parser.extract_tables(file_path)

        chunks.extend(chunker.build_chunks(text_docs, source= file_path))
        image_documents.extend(image_processor.get_image_documents())

    doc_indexing= indexing()
    retriever= doc_indexing.index_documents(chunks + image_documents)

    if retrieval_strategy == "agentic":
        agentic_qa= AgenticQA()
        agentic_qa.run(retriever)
        agentic_qa.query()
    else:
        qa = QA(retriever)
        qa.query()
