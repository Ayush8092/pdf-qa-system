# Simple PDF Question-Answering System

This project provides a straightforward PDF question-answering pipeline:
- Extract text from PDF pages
- Chunk text semantically
- Index with TF-IDF and sentence embeddings
- Retrieve and return best chunks for user queries

## Problem Statement

Manually searching large PDF documents for specific information is slow and error-prone. This system automates:
1. PDF text extraction
2. Semantic chunking
3. Hybrid lexical + semantic indexing
4. Natural-language question retrieval

## Features

- Supports single PDF upload
- TF-IDF and FAISS embedding index
- Fast retrieval (<5s) on CPU
- Returns most relevant text chunks
