from .chunkers import Chunker
from .docparser import DocParser
from .imageprocessing import ImageProcessor
from glob import glob
from pathlib import Path

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

    for file_path in files_list:
        print("processing started ...")

        text_docs= parser.parsing_function(file_path)
        parser.extract_tables(file_path)

        chunks= chunker.build_chunks(text_docs, source= file_path)
        image_documents= image_processor.get_image_documents()