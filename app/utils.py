from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from .config import  CHUNK_SIZE, CHUNK_OVERLAP
import logging

# Setup logger for this module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_line_numbered_chunks(code: str, language: Language, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    lines = code.splitlines(keepends=True)
    line_offsets = []
    offset = 0
    for i, line in enumerate(lines):
        line_offsets.append((offset, i + 1))
        offset += len(line)

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    documents = splitter.create_documents([code])

    def get_start_line(text_chunk: str) -> int:
        idx = code.find(text_chunk)
        if idx == -1: return -1
        for i in range(len(line_offsets) - 1, -1, -1):
            if line_offsets[i][0] <= idx:
                return line_offsets[i][1]
        return 1

    numbered_chunks = []
    for doc in documents:
        chunk = doc.page_content
        start_line = get_start_line(chunk)
        chunk_lines = chunk.splitlines()
        numbered_code = "\n".join(
            f"{start_line + i:4d} | {line}" for i, line in enumerate(chunk_lines)
        )
        numbered_chunks.append({
            "chunk": numbered_code,
            "start_line": start_line
        })

    return numbered_chunks

def wrap_code_with_line_numbers(code: str) -> list:
    lines = code.splitlines()
    numbered_code = "\n".join(f"{i + 1:4d} | {line}" for i, line in enumerate(lines))

    return [{
        "chunk": numbered_code,
        "start_line": 1
    }]

def print_lines(chunks):
    logger.info(f"Total chunks: {len(chunks)}")
    for chunk_info in chunks:
        logger.info(f"\nChunk starting at line {chunk_info['start_line']}:\n{chunk_info['chunk']}\n{'-' * 50}")

def build_prompt(chunk: dict, include_instruction: bool = True):
    prefix = ""
    if include_instruction:
        prefix = (
            """You are a smart contract code auditor with strict rules from the rulebook.\n
Return only issues in JSON format like this:\n
[
  {
    "line": 21,
    "type": "error",
    "message": "draft as per rulebook"
  }
]
Each line is prefixed with its original line number (e.g. 14 |). Double check line number. Pinpoint security flaws and guideline violations. If no issues, return [].\n\n"""
        )
    return prefix + f'"""\n{chunk["chunk"]}\n"""'
