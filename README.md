# Ollama + LangChain: Complete Learning Guide

A comprehensive, hands-on guide to building AI applications with Ollama and LangChain. This repository contains everything you need to understand and implement local LLMs, RAG applications, and AI agents.

## 📚 What You'll Learn

This guide is divided into three progressive modules:

1. **Ollama Python Basics** - Run LLMs locally on your machine
2. **RAG Applications** - Build intelligent document Q&A systems
3. **Tools & Agents** - Create autonomous AI agents with external tools

## 🎯 Who Is This For?

- Developers wanting to run LLMs locally without cloud dependencies
- Anyone interested in building RAG (Retrieval-Augmented Generation) applications
- AI enthusiasts learning about agents and tool-calling
- Teams building privacy-focused AI solutions

## 📖 How to Use This Repository

Each topic has **two learning resources**:

### 1. **Workflow Diagrams** (Start Here!)

ASCII diagrams that explain concepts visually - perfect for understanding before coding.

### 2. **Jupyter Notebooks**

Hands-on code examples you can run and experiment with.

## 🗺️ Learning Path

### Option A: Visual Learner (Recommended for Beginners)

```
1. Read workflow diagram → 2. Run notebook → 3. Experiment
```

### Option B: Hands-On Learner

```
1. Run notebook → 2. Reference workflow diagram when confused
```

---

## 📂 Repository Structure

```
Ollama-Langachain-RAG-Agents/
│
├── README.md                          # You are here!
       
# Sample data for RAG examples
│
├── notebooks/                         # Interactive Jupyter notebooks
│   ├── 1-Ollama-Python-Basics.ipynb
│   ├── 2-RAG-Application-Ollama-Langchain.ipynb
│   └── 3-Tools-and-Agents-Ollama-Langchain.ipynb
        LangchainRetrieval.txt 
│
└── workflows/                         # Visual ASCII workflow diagrams
    ├── 1-Ollama-Python-Basics-Workflow.txt
    ├── 2-RAG-Application-Workflow.txt
    └── 3-Tools-and-Agents-Workflow.txt
```

---

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama**

   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.com/install.sh | sh

   # Or download from: https://ollama.com/download
   ```
2. **Python 3.9+**

   ```bash
   python --version  # Should be 3.9 or higher
   ```
3. **Install Python Dependencies**

   ```bash
   pip install ollama langchain langchain-core langchain-ollama langchain-community langchain-chroma ddgs wikipedia
   ```

### Download a Model

```bash
# Download Llama 3.1 8B model (recommended)
ollama pull llama3.1:8b

# Download embedding model for RAG
ollama pull nomic-embed-text
```

### Verify Installation

```bash
# Check Ollama is running
ollama list

# Test with a simple query
ollama run llama3.1:8b "Hello, how are you?"
```

---

## 📘 Module 1: Ollama Python Basics

**Learn:** How to use Ollama with Python to run local LLMs

### What's Covered

- Installing and configuring Ollama
- Basic model interactions (`generate()` vs `chat()`)
- Creating custom models with system prompts
- Using Ollama REST API
- OpenAI compatibility mode

### Learning Resources

| Resource                   | Path                                              | Best For               |
| -------------------------- | ------------------------------------------------- | ---------------------- |
| **Workflow Diagram** | `workflows/1-Ollama-Python-Basics-Workflow.txt` | Understanding concepts |
| **Notebook**         | `notebooks/1-Ollama-Python-Basics.ipynb`        | Hands-on practice      |

### Key Concepts

```
User Question
     ↓
Ollama (Local LLM)
     ↓
Response (No API costs, Full privacy)
```

### Quick Example

```python
import ollama

# Simple generation
response = ollama.generate(
    model='llama3.1:8b',
    prompt='Explain quantum computing in simple terms'
)
print(response['response'])
```

---

## 📗 Module 2: RAG Applications

**Learn:** Build document Q&A systems that answer questions based on your own data

### What's Covered

- Document loading and text splitting
- Creating embeddings for semantic search
- Vector databases (ChromaDB)
- Building RAG chains with LangChain
- Querying your documents with natural language

### Learning Resources

| Resource                   | Path                                                   | Best For                       |
| -------------------------- | ------------------------------------------------------ | ------------------------------ |
| **Workflow Diagram** | `workflows/2-RAG-Application-Workflow.txt`           | Understanding RAG architecture |
| **Notebook**         | `notebooks/2-RAG-Application-Ollama-Langchain.ipynb` | Building your first RAG app    |
| **Sample Data**      | `LangchainRetrieval.txt`                             | Practice dataset               |

### Key Concepts

```
Your Documents
     ↓
Split into Chunks
     ↓
Create Embeddings (Vector Representations)
     ↓
Store in Vector DB (ChromaDB)
     ↓
User Question → Retrieve Relevant Chunks → LLM + Context → Answer
```

### Quick Example

```python
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

# Load and embed documents
docs = TextLoader("LangchainRetrieval.txt").load()
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma.from_documents(docs, embeddings)

# Ask questions
retriever = db.as_retriever()
# ... build chain and query
```

### Why RAG?

✅ **Up-to-date information** - Add new documents without retraining
✅ **Source attribution** - Know where answers come from
✅ **Domain-specific** - Answer questions on your custom data
✅ **Reduced hallucinations** - Grounded in actual documents
✅ **Cost-effective** - No need to fine-tune models

---

## 📙 Module 3: Tools & Agents

**Learn:** Create AI agents that can use external tools to solve complex problems

### What's Covered

- Understanding AI agents
- Creating tools (DuckDuckGo Search, Wikipedia)
- Building tool-calling agents
- Multi-step reasoning and planning
- Agent decision-making process

### Learning Resources

| Resource                   | Path                                                    | Best For                         |
| -------------------------- | ------------------------------------------------------- | -------------------------------- |
| **Workflow Diagram** | `workflows/3-Tools-and-Agents-Workflow.txt`           | Understanding agent architecture |
| **Notebook**         | `notebooks/3-Tools-and-Agents-Ollama-Langchain.ipynb` | Building your first agent        |

### Key Concepts

```
User Question
     ↓
Agent (LLM Brain)
     ↓
Decide Which Tool to Use
     ↓
Execute Tool (Search Web, Query Wikipedia, etc.)
     ↓
Synthesize Final Answer
```

### Quick Example

```python
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_ollama import ChatOllama

# Create tools
search = DuckDuckGoSearchResults()
tools = [search]

# Create agent
llm = ChatOllama(model="llama3.1:8b")
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Ask complex questions
agent_executor.invoke({"input": "What are the latest developments in AI?"})
```

### Agent Capabilities

✅ **Autonomous decision-making** - Chooses best tool for each task
✅ **Multi-step reasoning** - Chains multiple actions together
✅ **Real-time information** - Access to current web data
✅ **Extensible** - Easy to add custom tools
✅ **Transparent** - Verbose mode shows reasoning

---

## 🎓 Detailed Module Breakdown

### Module 1: Ollama Python Basics

#### Topics Covered

1. **Installation & Setup**

   - Installing Ollama and Python SDK
   - Downloading models
   - Verifying installation
2. **Basic Interactions**

   - `ollama.generate()` - Simple text generation
   - `ollama.chat()` - Conversational mode
   - Understanding response metadata
3. **Model Management**

   - Listing available models
   - Pulling new models
   - Deleting models
   - Creating custom models with Modelfile
4. **Advanced Usage**

   - Using the Ollama Client for REST API
   - System prompts for behavior customization
   - OpenAI compatibility mode
   - Temperature and parameter tuning

#### Hands-On Exercises

- [ ] Pull and run llama3.1:8b model
- [ ] Create a custom model with a unique personality
- [ ] Build a simple chatbot using `ollama.chat()`
- [ ] Compare responses with different temperature settings

---

### Module 2: RAG Applications

#### Topics Covered

1. **LangChain Fundamentals**

   - Prompt templates
   - ChatOllama integration
   - Output parsers
   - Chains and LCEL (LangChain Expression Language)
2. **Document Processing**

   - Loading documents (TextLoader, PDFLoader, etc.)
   - Text splitting strategies
   - Chunk size and overlap considerations
3. **Embeddings & Vector Storage**

   - Understanding text embeddings
   - Using OllamaEmbeddings
   - ChromaDB for vector storage
   - Similarity search
4. **Building RAG Pipelines**

   - Creating retrievers
   - Prompt engineering for RAG
   - Chain composition
   - Query execution
5. **Advanced RAG Techniques**

   - Contextual compression
   - Hybrid search
   - Re-ranking
   - Query transformation

#### Hands-On Exercises

- [ ] Load and split the LangchainRetrieval.txt file
- [ ] Create embeddings and store in ChromaDB
- [ ] Build a complete RAG chain
- [ ] Ask questions about LangChain features
- [ ] Experiment with chunk size and retrieval parameters

#### RAG Architecture Deep Dive

**Indexing Phase (One-time setup):**

```
Documents → Text Splitter → Embeddings → Vector Store
```

**Query Phase (Every user question):**

```
Question → Embed Query → Search Vectors → Retrieve Docs → LLM + Context → Answer
```

---

### Module 3: Tools & Agents

#### Topics Covered

1. **Agent Fundamentals**

   - What are AI agents?
   - Agent vs traditional LLMs
   - Agent architecture components
2. **Tool Creation**

   - Built-in tools (DuckDuckGo, Wikipedia)
   - Custom tool development
   - Tool descriptions and schemas
3. **Agent Building**

   - Prompt templates for agents
   - Tool-calling agents
   - AgentExecutor configuration
4. **Agent Execution**

   - Decision-making process
   - Multi-step reasoning
   - Agent scratchpad
   - Verbose output analysis
5. **Advanced Patterns**

   - Combining multiple tools
   - Error handling and retries
   - Agent memory
   - Conditional tool usage

#### Hands-On Exercises

- [ ] Create an agent with DuckDuckGo search
- [ ] Add Wikipedia as a second tool
- [ ] Build a custom calculator tool
- [ ] Observe agent reasoning with verbose mode
- [ ] Create a research assistant agent

#### Agent Decision Flow

```
1. Analyze user question
2. Determine if external information is needed
3. Select appropriate tool(s)
4. Execute tool with parameters
5. Evaluate results
6. Decide: Need more info? → Repeat | Have enough? → Generate answer
7. Synthesize final response
```

---

## 💡 Common Use Cases

### RAG Applications

| Use Case                     | Description                         | Example                             |
| ---------------------------- | ----------------------------------- | ----------------------------------- |
| **Document Q&A**       | Answer questions about company docs | "What's our vacation policy?"       |
| **Code Documentation** | Explain codebases                   | "How does the auth system work?"    |
| **Research Assistant** | Analyze research papers             | "Summarize findings on topic X"     |
| **Customer Support**   | Answer based on knowledge base      | "How do I reset my password?"       |
| **Legal Review**       | Analyze contracts                   | "What are the termination clauses?" |

### Agent Applications

| Use Case                   | Description                              | Tools Needed                                |
| -------------------------- | ---------------------------------------- | ------------------------------------------- |
| **Research Agent**   | Gather information from multiple sources | Web Search, Wikipedia, Academic APIs        |
| **Data Analyst**     | Query databases and generate reports     | SQL Tool, Calculator, Chart Generator       |
| **Customer Service** | Handle support tickets                   | Knowledge Base, Ticketing API, Email        |
| **Code Assistant**   | Help with programming tasks              | Documentation Search, Code Executor, GitHub |
| **Travel Planner**   | Research and book travel                 | Flight APIs, Hotel APIs, Weather API        |

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Ollama Connection Errors

```python
# Error: Cannot connect to Ollama
```

**Solution:**

```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve
```

#### 2. Model Not Found

```python
# Error: model 'llama3.1:8b' not found
```

**Solution:**

```bash
# Pull the model
ollama pull llama3.1:8b
```

#### 3. Import Errors

```python
# Error: No module named 'langchain_ollama'
```

**Solution:**

```bash
pip install langchain-ollama langchain-community
```

#### 4. Embedding Model Missing

```python
# Error: model 'nomic-embed-text' not found
```

**Solution:**

```bash
ollama pull nomic-embed-text
```

#### 5. ChromaDB Persistence Issues

If you want persistent vector storage:

```python
# Instead of:
db = Chroma.from_documents(documents, embeddings)

# Use:
db = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory="./chroma_db"
)
```

---

## 📊 Performance Tips

### Model Selection

| Model            | Size  | Speed     | Quality   | Best For                  |
| ---------------- | ----- | --------- | --------- | ------------------------- |
| llama3.1:8b      | 4.7GB | Fast      | Good      | General use, development  |
| llama3.1:70b     | 40GB  | Slow      | Excellent | Production, complex tasks |
| deepseek-r1:1.5b | 1.1GB | Very Fast | Fair      | Testing, quick prototypes |

### RAG Optimization

```python
# Optimal chunk size depends on your content
# Technical docs: 300-500 characters
# Narrative text: 500-1000 characters
# Code: 200-400 characters

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # Adjust based on content
    chunk_overlap=20     # 10-20% of chunk_size
)

# Retrieval optimization
retriever = db.as_retriever(
    search_kwargs={"k": 4}  # Retrieve top 4 chunks (adjust 2-10)
)
```

### Agent Performance

```python
# Use temperature=0 for consistent agent decisions
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0  # Deterministic
)

# Use verbose=True during development
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True  # See reasoning steps
)
```

---

## 🔒 Privacy & Security

### Why Local LLMs?

✅ **Full Privacy** - No data sent to external APIs
✅ **No Cost** - No API charges
✅ **Offline Capable** - Works without internet (after model download)
✅ **Compliance** - Easier GDPR/HIPAA compliance
✅ **Customization** - Full control over models

### Best Practices

1. **Sensitive Data**: Always use local models for confidential information
2. **API Keys**: If using external tools (search), store API keys in environment variables
3. **Data Retention**: ChromaDB stores embeddings locally - manage lifecycle appropriately
4. **Model Updates**: Regularly update models for security patches

---

## 🎯 Learning Objectives Checklist

### After Module 1, you should be able to:

- [ ] Install and configure Ollama
- [ ] Download and run local LLM models
- [ ] Use both `generate()` and `chat()` methods
- [ ] Create custom models with system prompts
- [ ] Understand model parameters (temperature, context size)

### After Module 2, you should be able to:

- [ ] Explain what RAG is and why it's useful
- [ ] Load and process documents for RAG
- [ ] Create embeddings and store in vector databases
- [ ] Build complete RAG chains with LangChain
- [ ] Query your documents using natural language
- [ ] Optimize chunk size and retrieval parameters

### After Module 3, you should be able to:

- [ ] Explain how AI agents work
- [ ] Create tools for agents to use
- [ ] Build tool-calling agents with LangChain
- [ ] Understand agent decision-making process
- [ ] Debug agents using verbose mode
- [ ] Design multi-step agent workflows

---

## 📚 Additional Resources

### Official Documentation

- [Ollama Documentation](https://github.com/ollama/ollama)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

### Community & Support

- [Ollama Discord](https://discord.gg/ollama)
- [LangChain Discord](https://discord.gg/langchain)
- [GitHub Discussions](https://github.com/ollama/ollama/discussions)

### Video Tutorials

- Search YouTube for "Ollama tutorial"
- Search YouTube for "LangChain RAG tutorial"
- Search YouTube for "LangChain agents"

### Books & Courses

- "Building LLM Applications" by various authors
- LangChain courses on Udemy/Coursera
- RAG implementation guides

---

## 🤝 Contributing

Found an issue or want to improve the examples? Contributions are welcome!

### How to Contribute

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Areas for Contribution

- Additional use case examples
- Performance optimization tips
- New tool implementations
- Documentation improvements
- Bug fixes

---

## 🆘 Getting Help

### Stuck on Something?

1. **Check the workflow diagrams** - Often the visual explanation helps
2. **Review the troubleshooting section** - Common issues are documented
3. **Run cells one at a time** - Easier to identify where errors occur
4. **Check Ollama logs** - `ollama logs` can show connection issues
5. **Verify dependencies** - `pip list | grep langchain` shows installed packages

### Still Need Help?

- Open an issue in this repository
- Join the Ollama Discord community
- Check Stack Overflow for similar questions

---

## 📝 License

This project is provided as educational material. Feel free to use, modify, and share.

---

## 🎉 What's Next?

After completing all three modules, consider:

1. **Build a Real Project**

   - Document Q&A system for your organization
   - Personal research assistant
   - Code documentation explorer
2. **Explore Advanced Topics**

   - Multi-agent systems
   - Fine-tuning local models
   - Advanced RAG techniques (HyDE, RAG-Fusion)
   - Memory systems for agents
3. **Optimize for Production**

   - Docker deployment
   - API wrappers
   - Monitoring and logging
   - Scaling strategies

---

## 📅 Suggested Learning Timeline

### Week 1: Foundations

- Day 1-2: Module 1 (Ollama Basics)
- Day 3-4: Practice and experimentation
- Day 5-7: Module 2 (RAG Applications)

### Week 2: Advanced Topics

- Day 1-2: Complete RAG exercises
- Day 3-4: Module 3 (Agents)
- Day 5-7: Build a personal project

### Week 3: Mastery

- Day 1-3: Combine concepts (RAG + Agents)
- Day 4-7: Build production-ready application

---

## 🙏 Acknowledgments

Built with:

- [Ollama](https://ollama.com/) - Run LLMs locally
- [LangChain](https://www.langchain.com/) - LLM application framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Meta&#39;s Llama](https://llama.meta.com/) - Base models

---

## 📧 Contact

Questions, suggestions, or feedback? Feel free to reach out or open an issue!

---

**Happy Learning! 🚀**

Start with the workflow diagrams, experiment with the notebooks, and build amazing AI applications!
