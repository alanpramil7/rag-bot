## Curl command for upload document
```bash
 curl -X 'POST' \
  'http://localhost:8000/documents/upload' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample.pdf' | jq
```

## Curl command for chat
```bash
 curl -X 'POST' \
  'http://localhost:8000/chat/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "Can you help me write a sample paper accroding to this instructions?",
    "session_id": "e6e7529e-cc64-4c01-b37c-6dd2606f86a5"
  }' | jq


```

## Curl command to get chat hisotry
```bash
curl -X 'POST' \
  'http://localhost:8000/chat/history?session_id=e6e7529e-cc64-4c01-b37c-6dd2606f86a5' \
  -H 'accept: application/json'
```
