# Document Buddy
Chat with various document types - XLSX, PPTX, DOCX, PDF, CSV, TXT - using chatGPT 4 Turbo and Langchain

## Getting started

1. Clone the repo

2. Copy the `document_buddy/.env.dist` file to `document_buddy/.env`:

   ```shell
   cp document_buddy/.env.dist document_buddy/.env
   ```

3. Edit the `document_buddy/.env` file and add your API key(s). You can add either an OpenAI key, an Anthropic key, or both if you would like to get answers from both services:

   - If you need an OpenAI key, visit https://platform.openai.com

   - If you need an Anthropic key, visit https://console.anthropic.com

## Bring up the server
docker compose up 

## Visit localhost
http://localhost:8510

## Thank you