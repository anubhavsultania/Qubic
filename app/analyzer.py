import os
import re
import json
import asyncio
from typing import List
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.text_splitter import Language
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from .utils import add_line_numbered_chunks, build_prompt, print_lines, wrap_code_with_line_numbers
from app.config import GOOGLE_API_KEY
import logging
logger = logging.getLogger(__name__)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.15,
    check_every_n_seconds=0.1,
    max_bucket_size=10,
)

def parse_model_response(raw_response):
    clean = re.sub(r"^```json|```$", "", raw_response.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(clean) if clean else []
    except json.JSONDecodeError:
        return []

async def analyze_code(code: str) -> List[dict]:
    chunks = wrap_code_with_line_numbers(code)
    print_lines(chunks)
    all_issues = []

    for i, chunk in enumerate(chunks):
        query = build_prompt(chunk, include_instruction=(i % 1 == 0))
        await asyncio.sleep(0.1)
        while not rate_limiter.acquire():
            await asyncio.sleep(0.1)

        logger.info(f"Processing chunk {i+1}/{len(chunks)}...")

        response = await qa_chain.arun(query)
        result = parse_model_response(response)
        all_issues.extend(result)

    return all_issues

