# Custom Research Assistant

## Project Structure
| Layer     | Description                                                                             |
| --------- | --------------------------------------------------------------------------------------- |
| Frontend  | Streamlit UI for user interaction and QA input                                          |
| Backend   | Handles document loading, chunking, embedding, vector store operations, and QA pipeline |
| Vector DB | FAISS (local, in-memory or persisted to disk)                                           |
| LLM       | OpenAI (via `langchain` integration)                                                    |

## Tech Stack
| Component       | Technology Used                  |
| --------------- |----------------------------------|
| Frontend        | Streamlit                        |
| Backend         | LangChain + Python               |
| Embeddings      | `OpenAIEmbeddings`               |
| Vector Store    | `FAISS`, `langchain_community`   |
| LLM Integration | `OpenAI`, LangChain LLM wrappers |

## Folder Structure
```commandline
custom_research_assistant/
├── backend/
│   ├── content_processor.py         
│   ├── document_loader.py           
│   ├── qn_answer_pipline.py         
│   └── .env                         
│
├── frontend/
│   └── app.py                       
│
├── vector_store/ (Auto-generated)
│   ├── index.faiss                 
│   ├── documents.pkl               
│   └── index_map.pkl               
│
├── requirements.txt                
└── README.md                       
```

## Setup Instructions

1. **Clone the repository**:
```bash
    git clone <repo>
    cd custom_research_assistant
```
2. **Install dependencies**:
```commandline
    pip install -r requirements.txt
```
3. **Create environment file for API key**:
```bash
    touch .env
```
    **Then add:**
```
    OPENAI_API_KEY=your_openai_api_key_here
```
**4.Run Streamlit App**:
```commandline
    streamlit run frontend/app.py
```
