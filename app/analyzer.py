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
# from app.config import QDRANT_URL, QDRANT_API_KEY, GOOGLE_API_KEY, QDRANT_COLLECTION_NAME
from app.config import GOOGLE_API_KEY
# from qdrant_client import QdrantClient
# from langchain_community.vectorstores import Qdrant
# from qdrant_client.http.models import Distance, VectorParams
import logging
logger = logging.getLogger(__name__)





# os.environ["GOOGLE_API_KEY"] = 

embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)



# Connect to Qdrant
# qdrant_client = QdrantClient(
#     url=QDRANT_URL,
#     api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
# )
#
# # Create collection if it doesn't exist
# if not qdrant_client.collection_exists(collection_name=QDRANT_COLLECTION_NAME):
#     qdrant_client.recreate_collection(
#         collection_name=QDRANT_COLLECTION_NAME,
#         vectors_config=VectorParams(
#             size=embedding_model.embedding_dimensions,
#             distance=Distance.COSINE,
#         ),
#     )
#
# # Wrap Qdrant with LangChain
# vectorstore = Qdrant(
#     client=qdrant_client,
#     collection_name=QDRANT_COLLECTION_NAME,
#     embeddings=embedding_model,
# )





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



# import os
# import re
# import json
# import asyncio
# from typing import List
# from langchain_core.rate_limiters import InMemoryRateLimiter
# from langchain.text_splitter import Language
# from langchain_core.documents import Document
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from .utils import add_line_numbered_chunks, build_prompt, print_lines, wrap_code_with_line_numbers
# from app.config import GOOGLE_API_KEY

# # Load models
# try:
#     embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#     vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
#     llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
#     print("✅ Models loaded successfully.")
# except Exception as e:
#     print("❌ Failed to initialize embedding model, vectorstore, or LLM.")
#     print(e)
#     raise

# # Build retriever and QA chain
# try:
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
#     qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
#     print("✅ Retriever and QA chain initialized.")
# except Exception as e:
#     print("❌ Failed to initialize retriever or QA chain.")
#     print(e)
#     raise

# # Rate limiter
# rate_limiter = InMemoryRateLimiter(
#     requests_per_second=0.15,
#     check_every_n_seconds=0.1,
#     max_bucket_size=10,
# )

# # Parser for model response
# def parse_model_response(raw_response: str):
#     clean = re.sub(r"^```json|```$", "", raw_response.strip(), flags=re.MULTILINE).strip()
#     try:
#         parsed = json.loads(clean) if clean else []
#         return parsed
#     except json.JSONDecodeError as e:
#         print("❌ JSON decode error:", e)
#         print("↪ Raw response that caused the error:")
#         print(raw_response)
#         return []

# # Analyzer function
# async def analyze_code(code: str) -> List[dict]:
#     chunks = wrap_code_with_line_numbers(code)
#     print_lines(chunks)  # Print each code chunk

#     all_issues = []

#     for i, chunk in enumerate(chunks):
#         query = build_prompt(chunk, include_instruction=(i % 1 == 0))
#         await asyncio.sleep(0.1)
#         while not rate_limiter.acquire():
#             await asyncio.sleep(0.1)

#         print(f"\n🚧 Processing chunk {i + 1}/{len(chunks)}")

#         try:
#             response = await qa_chain.arun(query)
#             print(f"✅ Response received for chunk {i + 1}:")
#             print(response)
#         except Exception as e:
#             print(f"❌ Error while running LLM on chunk {i + 1}:")
#             print(e)
#             continue

#         try:
#             result = parse_model_response(response)
#             print(f"✅ Parsed result for chunk {i + 1}:")
#             print(result)
#             all_issues.extend(result)
#         except Exception as e:
#             print(f"❌ Failed to parse model response for chunk {i + 1}")
#             print(e)
#             print("↪ Raw response:")
#             print(response)

#     return all_issues

