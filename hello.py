from qdrant_client import QdrantClient

client = QdrantClient(
    url="https://6d764c08-d800-45f8-85b3-e71765344577.us-west-1-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.FSWK9bwsThThJS-hIH3jm4sYDbSvbJxM7jH2ucn-1E8"
)

print(client.get_collections())
