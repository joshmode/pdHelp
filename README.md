### `pdHelp` 
```markdown
# 🧠 pdHelp: Localized RAG AI Agent

> A privacy-first, local AI assistant built in Python that leverages Retrieval-Augmented Generation (RAG) to securely parse and analyze domain-specific PDF data.

## 🏗️ RAG Pipeline Architecture

```text
[ Local PDF Document ] ──► ( Document Parser )
                                │
                                ▼
[ User Query ] ──────────► ( Vector Embedding & Context Retrieval )
                                │
                                ▼
                           ( Local LLM ) ──► Generates Grounded Response
