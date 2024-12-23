## Curl command for upload document
```bash
curl -X 'POST' \
  'http://localhost:8000/documents/upload' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/file.pdf'
```

## Curl command for chat
```bash
curl -X 'POST' \
  'http://localhost:8000/chat/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What is the main topic of the document?",
    "chat_history": [],
    "file_id": "6d0305b2-bd79-48d1-b8ab-f9d0018c2a5f"
  }' | jq
```
