# DeepSeek R1 PDF Chat with Ollama

A Streamlit application that allows users to chat with their PDF documents using DeepSeek R1 1.5B model running locally through Ollama.



## Features

- **Local Document Processing**: Processes PDF files without sending data to external services
- **Vector Embedding**: Creates embeddings from documents for semantic search
- **Interactive Chat**: Ask questions about your documents and get contextual answers
- **Efficient Processing**: Uses chunking to handle large documents effectively
- **Response Time Tracking**: Shows how long each query takes to process

## Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) installed locally with DeepSeek R1 model
- Required Python packages (see installation section)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/username/deepseek-r1-pdf-chat.git
cd deepseek-r1-pdf-chat
```

2. Install the required packages:
```bash
pip install streamlit langchain langchain-community faiss-cpu
```

3. Install Ollama and pull the DeepSeek R1 model:
```bash
# Install Ollama from https://ollama.ai/
ollama pull deepseek-r1:1.5b
```

## Usage

1. Create a directory for your PDF files:
```bash
mkdir -p docus
```

2. Place your PDF files in the `docus` directory.

3. Run the Streamlit app:
```bash
streamlit run app.py
```

4. In the web interface:
   - Click "Documents Embedding" to process the PDF files
   - Enter your question in the text input field
   - View the response with relevant document sections

## How It Works

1. **Document Loading**: PDF files from the `docus` directory are loaded
2. **Text Splitting**: Documents are split into manageable chunks for processing
3. **Embedding Generation**: DeepSeek R1 generates vector embeddings for each chunk
4. **Vector Store Creation**: Embeddings are stored in a FAISS vector database
5. **Query Processing**: When a question is asked, relevant chunks are retrieved
6. **Response Generation**: The LLM generates answers based on the relevant document portions

## Customization

You can modify the following parameters in the code:
- `chunk_size`: Size of document chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- Adjust the prompt template for different response styles

## Limitations

- The application currently only processes PDF files
- Processing large documents may require significant memory
- Response quality depends on the DeepSeek R1 1.5B model capabilities
- Running locally means performance depends on your hardware

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the document processing framework
- [Ollama](https://ollama.ai/) for local LLM serving
- [Streamlit](https://streamlit.io/) for the web interface
- [DeepSeek](https://github.com/deepseek-ai/deepseek-coder) for the language model
