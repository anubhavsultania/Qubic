# import os
#
# # Embed your API key securely here or use .env in real deployment
# GOOGLE_API_KEY = os.getenv("AIzaSyBaZiaQDG-3gkY1RxhQigSY4YjtEuW4z0g")
# CHUNK_SIZE = 2000
# CHUNK_OVERLAP = 100
#
# QDRANT_URL = os.getenv("QDRANT_URL", "https://your-qdrant-url.com")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "your-secret-api-key")  # Leave blank if none
# QDRANT_COLLECTION_NAME = "cpp_code_chunks"
#


# app/config.py
from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env into environment

# Now read values into constants
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_COLLECTION_NAME = "cpp_code_chunks"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100
