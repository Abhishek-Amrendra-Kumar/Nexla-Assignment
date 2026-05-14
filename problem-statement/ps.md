# Nexla | Software–AI | Take-Home Assignment

> Role: **Software Engineer**  
> Estimated Time: **3–4 hours**  
> Deliverable: **Public GitHub Repository URL**

Provided Documents:
- 4–5 PDF files (data)

---

# 1. About Nexla

Nexla is the leading integration platform built with AI, for AI. It takes a metadata-driven approach to converge diverse integrations across Data, Documents, Agents, Applications, and APIs into a single unified design pattern.

Nexla accelerates the development of solutions for GenAI, Analytics, and inter-company data exchange — making data users and developers up to **10x more productive** through a true blend of:
- no-code
- low-code
- pro-code interfaces

Trusted by leading companies including:
- DoorDash
- LinkedIn
- Johnson & Johnson
- LiveRamp

Nexla is:
- recognized in the Gartner Magic Quadrant for Data Integration Tools
- top-rated by customers on Gartner Peer Insights

As a Software Engineer at Nexla, your job is to:
- embed directly with strategic customers
- understand real-world data challenges
- ship production-grade solutions on top of the Nexla platform

This assignment is designed to simulate that exact workflow:

> Given a messy, real-world problem — build something that works, scales, and is well-reasoned.

---

# 2. Assignment Context

One of the most common challenges Nexla's enterprise customers face is making sense of large volumes of unstructured document data, such as:
- reports
- contracts
- whitepapers
- research papers

These documents often contain:
- valuable structured information
- semi-structured information

However, that information is difficult to:
- query
- search
- analyze programmatically

The **Model Context Protocol (MCP)** is an emerging open standard that enables AI models and agents to interact with external systems through a well-defined tool interface.

Building an MCP server that exposes document intelligence capabilities is considered a meaningful real-world engineering problem and reflects the kind of work expected in this role.

In this assignment, you are provided with **4–5 PDF documents**.

Your task is to:

> Build an MCP server that allows an AI agent (e.g., Claude, ChatGPT, or any MCP-compatible client) to ask natural language questions and receive grounded answers derived from the content of those documents.

---

# 3. The Assignment

## 3.1 Core Task

Build a working MCP server that exposes a **Q&A tool** over the provided PDF documents.

An AI agent connecting to your MCP server should be able to:
- pose natural language questions
- receive accurate, grounded answers
- get answers derived directly from document content

---

## 3.2 Functional Requirements

Your MCP server **must** satisfy the following requirements:

### Document Ingestion
The server must:
- parse all 4–5 provided PDFs
- index them
- do this either:
  - at startup
  - or on demand

### Q&A Tool
Expose at least one MCP tool, for example:
- `query_documents`

The tool must:
- accept a natural language question
- return a grounded answer
- include:
  - source document reference
  - relevant section reference

### Multi-document Awareness
The system must support questions that require:
- retrieving context from multiple documents

### Source Attribution
Every answer must include:
- document name
- and where possible:
  - page number
  - section reference

### MCP Protocol Compliance
The server must:
- be a valid MCP server
- expose tools via the standard MCP tool-calling interface
- run locally

---

## 3.3 Vibe Coding Setup

Nexla wants to understand how you use AI-assisted development tools in your workflow.

You are expected to use a **vibe coding setup**, meaning:
- leverage AI coding assistants during development

Examples include:
- Cursor
- GitHub Copilot
- Claude Code
- Windsurf
- similar tools

### README Requirements

Include a section in your README describing:

#### AI Coding Tools Used
Explain:
- which tools you used
- how you used them

#### Prompting Strategy
Describe:
- how you prompted or directed the AI
- what worked
- what did not work

#### Human vs AI Contribution
Explain:
- where you relied on AI
- where you corrected or overrode AI-generated output

#### Reflection on AI Tooling
Share your perspective on:
- how AI tooling fits into software engineering workflows

There are no “correct” answers here.

The evaluation focuses on:
- self-awareness
- intentionality
- engineering judgment

—not on maximizing AI usage.

---

# 4. Technical Guidelines

## 4.1 Technology Choices

You are free to choose your own stack.

The document provides the following **non-prescriptive suggestions**:

| Component | Suggested Options |
|---|---|
| MCP Framework | Anthropic MCP SDK (Python), FastMCP |
| PDF Parsing | PyMuPDF, pdfplumber, pypdf, LlamaParse |
| Embeddings | OpenAI `text-embedding-3-small`, Cohere, `sentence-transformers` |
| Vector Store | ChromaDB, FAISS, Pinecone, Qdrant, Weaviate |
| LLM for Q&A | OpenAI GPT-5, Anthropic Claude, Ollama (local) |
| Language | Python |

---

## 4.2 What Nexla Is *Not* Looking For

They are **not** expecting:

### Production Deployment
A perfect production-deployed system is **not required**.

A:
- clean
- local
- runnable implementation

is sufficient.

### Specific Technologies
You are free to use:
- any tools
- any frameworks
- any architecture

as long as choices are reasonable and justified.

### Full Production Engineering
You do **not** need:
- CI/CD pipelines
- cloud deployment
- enterprise productionization

---

# 5. Deliverables

Submit:

> A single public GitHub repository URL

The repository must include the following.

---

## 5.1 Working MCP Server Code

Include:

1. All source code needed to run the MCP server locally
2. A working implementation of:
   - `query_documents`
   - or an equivalently named tool
3. Any:
   - ingestion scripts
   - preprocessing steps
   required to index the PDFs

---

## 5.2 README.md (Required — Will Be Read Carefully)

Your README is considered just as important as your code.

It must include:

### Setup Instructions
Step-by-step instructions for:
- installing dependencies
- running the server from scratch

### Architecture Overview
A written explanation or diagram showing:
- system structure
- component interaction

### Tool Documentation
Document:
- each MCP tool
- inputs
- outputs
- example queries

### Vibe Coding Section
Include the AI-tooling discussion described earlier.

---

## 5.3 Example Interaction Log

Include at least **3 sample questions** and their corresponding answers.

You may include these:
- in a Markdown file
- or directly in the README

The examples should demonstrate:
- that the system works
- that source attribution is included

---

# 6. Evaluation Criteria

Your submission will be evaluated across four dimensions.

| Criterion | Weight | What They Look For |
|---|---|---|
| Code Quality & Architecture | 35% | Clean, readable, well-structured code. Good separation of concerns. Thoughtful design choices with clear intent. |
| MCP Protocol Understanding | 25% | Correct MCP server implementation. Properly defined tools and schemas. Demonstrably working Q&A interaction. |
| Vibe Coding Setup | 40% | Evidence of intentional AI-assisted development. Honest reflection on tools used, workflow, and lessons learned. Demonstrates AI engineering mindset. |

---

# 7. Submission Instructions

Before submitting:

1. Ensure the repository is **public**
2. Reply to the email thread from which you received the assignment
3. Use the subject format:

```text
[Software Engineer-AI Take-Home] Your Name
```

Additional note:
- The estimated completion time is **3–4 hours**
- Nexla explicitly asks candidates to stay reasonably close to that estimate

Clarification policy:
- You may reply to the same email thread with questions
- The timer does **not pause** while waiting for responses

---

# 8. A Note from the Team

Nexla explicitly states:

> They are not looking for perfection.

They care more about:
- how you think
- how you reason
- how you communicate trade-offs

A:
- clean
- documented
- thoughtfully explained solution

will score higher than:
- a technically sophisticated implementation with poor explanation

The role is described as operating like:

> “a technical co-founder for each customer engagement”

They want to evaluate whether you can:
- own outcomes
- reason under ambiguity
- communicate engineering decisions clearly

This assignment is specifically designed to surface those abilities.

Final note from the document:

> Good luck — and we look forward to seeing what you build.

— Nexla, Inc. | San Mateo, California | nexla.com
