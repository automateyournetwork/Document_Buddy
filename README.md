# Document Buddy
Chat with various document types - XLSX, PPTX, DOCX, PDF, CSV, TXT - using chatGPT 4 Turbo and Langchain

## Getting started

Clone the repo

Make a .env file inside the document_buddy subfolder (where the document_buddy.py file is located; /document_buddy/document_buddy)

put this in the file:

```console
OPENAI_API_KEY="<your openapi api key>"
ANTHROPIC_API_KEY="<your Anthropic api key>"
```

If you need an OpenAI key visit https://platform.openai.com to get one

If you need an Anthropic key visit https://console.anthropic.com

## Bring up the server
docker-compose up 

## Visit localhost
http://localhost:8510

## Thank you