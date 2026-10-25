# A fine-tuned transformer summarization model-Based Approach to Legal document summarization

**Domain:** Generative AI and Agentic AI  
**Project Type:** Academic Minor Project  
**Application:** AI-Assisted Legal Document Summarization and Analysis

---

## 1. Abstract

Legal documents such as contracts, agreements, terms and conditions, and other legal records often contain lengthy paragraphs, complex terminology, and important clauses that require careful interpretation. Manually reviewing these documents can be time-consuming and may make it difficult to identify key obligations, conditions, exceptions, and other relevant information.

This project, **“A fine-tuned transformer summarization model-Based Approach to Legal document summarization,”** aims to develop an AI-powered application that assists users in understanding lengthy legal documents through automated text summarization and supporting document-analysis capabilities.

The project belongs to the domain of **Generative AI and Agentic AI**. Generative AI is applied through a fine-tuned Longformer Encoder-Decoder (LED) transformer model intended to generate concise summaries from legal text. The application is developed using Python, PyTorch, Hugging Face Transformers, and Streamlit to provide an interactive interface for document upload, processing, and review.

In addition to summarization, the application provides dedicated workspaces for evidence search, claim review, and paralegal-oriented document analysis. These capabilities are intended to help users locate relevant information, examine statements against source evidence, and navigate document-review tasks. Depending on the implemented backend functionality, these workspaces can support AI-assisted workflows associated with the broader concept of Agentic AI.

The main objective is to combine generative text summarization with supporting legal-document analysis in a single application. By making lengthy documents easier to review and providing tools to examine source information, the proposed system aims to improve document accessibility and review efficiency while encouraging verification against the original document.

**Keywords:** Generative AI, Agentic AI, Legal Document Summarization, Fine-Tuned Transformer, Longformer Encoder-Decoder, LED, Natural Language Processing, Deep Learning, Automated Text Generation, Evidence Search, Claim Review, Legal Document Analysis, Streamlit.

---

## 2. Introduction

Legal documents are essential in business transactions, employment agreements, financial arrangements, property matters, and many other areas. These documents frequently contain formal language, complex sentence structures, detailed conditions, and multiple clauses that must be reviewed carefully.

Reading a lengthy legal document manually can require significant time and effort. Users may need to identify the main purpose of an agreement, understand its important provisions, locate relevant sections, and verify particular statements against the original content.

Generative Artificial Intelligence provides techniques for producing human-readable text from an input. Transformer-based models have become important in text-generation tasks, including summarization. Longformer Encoder-Decoder (LED) is designed to handle longer input sequences than many conventional encoder-decoder transformer models, making it a candidate for long-document summarization.

Agentic AI focuses on AI systems that can use tools, work through multiple steps, and pursue tasks toward a defined objective. In legal-document applications, such workflows may involve finding relevant evidence, examining claims, and organizing document-review activities.

This project combines a fine-tuned transformer-based summarization component with a Streamlit application that provides document-processing and analysis workspaces. The intended result is an accessible research prototype for AI-assisted legal-document review.

---

## 3. Project Domain

### Generative AI and Agentic AI

**Generative AI:** The project uses a fine-tuned LED transformer model to generate summaries from legal-document text. This is the principal generative component of the application.

**Agentic AI:** The project includes separate workspaces for evidence search, claim review, and paralegal-oriented document analysis. These features provide a foundation for AI-assisted document workflows. The extent to which they qualify as autonomous or tool-using agents depends on the actual backend implementation.

The application brings these capabilities together through an interactive interface so that users can process documents, review generated summaries, and explore relevant source content.

---

## 4. Problem Statement

Legal documents frequently contain lengthy text, complex terminology, and detailed clauses that are difficult to review quickly. Manual analysis requires users to read the document carefully, identify important information, locate relevant sections, and verify statements against the original source.

Conventional summarization approaches can also face limitations when processing long documents because transformer models have finite input lengths. Important context may be lost if a document is truncated or divided without an appropriate strategy.

Therefore, this project aims to develop an interactive legal-document summarization application using a fine-tuned long-context transformer model, together with supporting document-analysis workspaces that assist users in reviewing relevant content and evidence.

---

## 5. Objectives

The main objectives of the project are:

1. To develop an AI-powered application for legal document summarization.
2. To use a fine-tuned LED transformer model for generating summaries of legal text.
3. To provide an interactive interface for uploading and processing supported legal documents.
4. To make lengthy legal documents easier to understand through concise summaries.
5. To provide evidence-search functionality for locating relevant document content.
6. To provide claim-review functionality for examining statements against available evidence.
7. To provide a paralegal-oriented workspace for additional document analysis.
8. To integrate generative AI capabilities into a practical application using Python and Streamlit.
9. To explore AI-assisted document workflows related to the Generative AI and Agentic AI domain.
10. To establish a foundation for future improvements in model quality, evaluation, automation, and deployment.

---

## 6. Existing System

In the traditional legal-document review process, users manually read lengthy contracts and agreements to understand their content and identify important clauses.

### Limitations of the existing approach

- Time-consuming manual document review.
- Difficulty locating specific clauses in long documents.
- Complex legal terminology and sentence structures.
- Repeated searches for particular information.
- Challenges in quickly understanding the overall content of an agreement.
- Potential omission of important details during manual review.

General-purpose summarization systems may also have difficulty processing long legal documents when their input-length limits are insufficient.

---

## 7. Proposed System

The proposed system uses a fine-tuned LED transformer model as the primary summarization component within a Streamlit-based application.

Users can upload supported legal documents and access the application's document-processing and analysis workspaces. The summarization component is intended to generate a concise representation of the document, while the supporting workspaces help users locate evidence, review claims, and inspect legal-document content.

### Main components

- **Document Studio:** The primary workspace for document processing and summarization.
- **Evidence Search:** A workspace for finding relevant content within available document evidence.
- **Claim Review:** A workspace for examining statements in relation to source evidence.
- **Paralegal Review:** A workspace for supporting legal-document inspection and analysis.

The system aims to reduce the effort required for an initial document review while keeping source verification central to the process.

The quality and reliability of the output depend on the model, document extraction, preprocessing, inference configuration, and correctness of the supporting analysis components.

---

## 8. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive web application interface |
| PyTorch | Deep-learning operations and model inference |
| Hugging Face Transformers | Transformer model and tokenizer integration |
| LED (Longformer Encoder-Decoder) | Legal-document summarization model |
| PyPDF | Extracting text from supported PDF documents |
| python-docx | Processing supported Word documents |
| Pandas | Tabular data processing where required |
| NumPy | Numerical operations where required |
| Git and GitHub | Source-code version control and repository hosting |
| Streamlit Community Cloud | Potential hosting platform for the application |

The package list and versions should be verified against the actual working environment before deployment.

---

## 9. System Architecture

The application follows a document-processing workflow with separate workspaces for supporting analysis.

```text
                      USER
                        |
                        v
               Streamlit Interface
                        |
                        v
              Upload Legal Document
                        |
                        v
               Extract Document Text
                        |
                        v
               Text Preprocessing
                        |
                        v
               LED Tokenizer
                        |
                        v
             Fine-Tuned LED Model
                        |
                        v
              Generate Summary
                        |
                        v
              Display Summary
                        |
                        v
              Supporting Workspaces
                 /      |       \
                /       |        \
               v        v         v
          Evidence   Claim    Paralegal
           Search    Review    Review
                \       |        /
                 \      |       /
                        v
                Source Verification
                        |
                        v
                  User Review
```

This is a high-level representation of the intended workflow. The exact internal sequence and integration of each workspace depend on the implementation.

---

## 10. Methodology

### 10.1 Document Upload

The user selects a legal document through the Streamlit interface. The application processes supported file formats using the relevant document-reading functions.

### 10.2 Text Extraction

The application extracts text from the uploaded document.

- PDF documents are processed using PyPDF.
- Supported Word documents are processed using python-docx.

The extracted text becomes the input for subsequent processing. Scanned PDFs may require OCR support if their text is not directly extractable.

### 10.3 Text Preprocessing

The extracted content is prepared for summarization. Depending on the implementation, preprocessing may include cleaning text, handling empty content, removing extraction artifacts, organizing text, and preparing the input for tokenization.

The quality of extracted text is important because missing or incorrectly extracted information can affect the generated summary.

### 10.4 Tokenization

The document text is converted into token IDs using the tokenizer associated with the fine-tuned LED model.

Tokenization converts text into a representation that can be processed by the transformer. Long documents require appropriate input-length handling because the model cannot process unlimited text in a single input.

### 10.5 Fine-Tuned LED Transformer

The application is designed to load a fine-tuned Longformer Encoder-Decoder model from the configured checkpoint directory.

LED uses an encoder-decoder architecture with sparse attention mechanisms intended to support longer input sequences than many conventional transformer architectures.

The fine-tuned model is intended to generate summaries suitable for legal-document content. Its actual performance depends on the pretrained model, training data, fine-tuning procedure, and saved checkpoint.

### 10.6 Summary Generation

After preprocessing and tokenization, the input is passed to the model's text-generation process.

The generated tokens are decoded into readable text and displayed in the application. Input truncation, chunking strategy, generation parameters, and maximum output length can affect the resulting summary.

### 10.7 Evidence Search

The Evidence Search workspace is intended to help users find relevant content in available document evidence. It supports the process of locating source information that can be examined alongside a generated summary or statement.

### 10.8 Claim Review

The Claim Review workspace is intended to help users examine statements in relation to source evidence. Users should verify the source text, context, and any conclusions before relying on the result.

### 10.9 Paralegal Review

The Paralegal Review workspace provides an additional interface for legal-document inspection and analysis. It is intended to support review activities rather than replace professional legal interpretation.

### 10.10 Output Review

The application displays the generated summary and the results provided by its supporting workspaces. Users should compare important details against the original document, especially names, dates, amounts, obligations, exceptions, and conditions.

---

## 11. Main Application Features

### 11.1 Legal Document Upload

Allows users to select supported legal documents for processing through the application interface.

### 11.2 Transformer-Based Summarization

Uses the configured fine-tuned LED model to generate a concise representation of legal-document text.

### 11.3 Document Studio

Provides the primary workspace for document processing and summary review.

### 11.4 Evidence Search

Supports locating relevant content within available source evidence.

### 11.5 Claim Review

Supports examining statements against the available source material.

### 11.6 Paralegal Review

Provides an additional workspace for inspecting legal-document content and supporting review activities.

### 11.7 Interactive User Interface

Uses Streamlit to provide an accessible interface without requiring users to interact directly with model code.

### 11.8 AI-Assisted Document Workflow

Brings summarization and supporting analysis workspaces together in a single application. More advanced agentic behavior can be developed by integrating tool use, task planning, evidence retrieval, and multi-step execution where appropriate.

### 11.9 Source Verification

Encourages users to check generated summaries and analysis results against the original document.

---

## 12. Generative AI and Agentic AI in the Project

### Generative AI component

The fine-tuned LED transformer is the project's primary generative AI component. It is intended to produce summaries from input legal-document text.

The generated output is based on the model's learned parameters and the input provided during inference. It may omit important information or produce inaccurate statements, so verification against the source document is necessary.

### Agentic AI component

Agentic AI systems generally perform tasks through coordinated steps, potentially including planning, using tools, retrieving information, and evaluating intermediate results.

This project includes separate workspaces for evidence search, claim review, and paralegal-oriented analysis. These features can support an agentic workflow, but their presence alone does not establish that the application has autonomous agents.

A more advanced agentic implementation could include:

1. Identifying the user's document-review objective.
2. Selecting an appropriate analysis tool.
3. Retrieving relevant document sections.
4. Comparing a claim against retrieved evidence.
5. Producing a response with supporting source references.
6. Flagging uncertain results for human review.

These are potential workflow enhancements unless they are already implemented in the application's backend.

---

## 13. Project Structure

The project is organized around the Streamlit application, model resources, supporting modules, and project data.

A representative structure is shown below. The actual folder names should match the repository.

```text
Legal_AI_Summarization/
│
├── src/
│   ├── app.py
│   └── pipeline/
│       ├── claim_evidence_pipeline.py
│       └── contract_analysis.py
│
├── models/
│   └── led_finetuned/
│       └── checkpoint-147/
│
├── data/
├── logs/
├── notebooks/
├── outputs/
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

**Important:** The names shown under `src/pipeline/` are examples based on the application's referenced modules. Preserve the actual filenames and folder structure in your local project.

The model directory must contain a complete, valid model checkpoint before summarization can work. A folder containing only training-state files is not sufficient for normal inference.

---

## 14. Dataset and Model Training

The application is intended to use a fine-tuned LED transformer model for legal-document summarization.

For reproducibility, the final project documentation should record the following information from the actual training process:

- Dataset name and source.
- Dataset licence and permitted use.
- Type of legal documents included.
- Input document and target summary fields.
- Text-cleaning and preprocessing methods.
- Training, validation, and test split.
- Pretrained LED model used.
- Fine-tuning hyperparameters.
- Training environment and available hardware.
- Checkpoint-selection procedure.
- Final model storage location.

The exact dataset, training configuration, and measured performance are not specified in this README and should be added only after they have been confirmed from the training notebook and evaluation results.

---

## 15. Evaluation and Testing

A legal-document summarization system should be evaluated for summary quality, factual consistency, and application reliability.

### 15.1 Summary Quality

Potential evaluation methods include:

- **ROUGE:** Measures overlap between generated summaries and reference summaries.
- **BERTScore:** Estimates semantic similarity between generated and reference text.
- **Human evaluation:** Examines readability, coverage of important clauses, factual consistency, and usefulness.

### 15.2 Functional Testing

The following application components should be tested:

- Document upload.
- PDF and Word text extraction.
- Text preprocessing.
- Model and tokenizer loading.
- Summary generation.
- Output display and downloads, where implemented.
- Evidence search.
- Claim review.
- Paralegal review.
- Handling of unsupported files and empty documents.
- Error handling when model files are missing.

### 15.3 Agentic Workflow Evaluation

If multi-step agentic behavior is implemented, additional tests can examine:

- Whether the correct tool is selected for a task.
- Whether retrieved evidence is relevant.
- Whether conclusions are supported by the source.
- Whether errors and uncertain results are handled appropriately.
- Whether the system avoids presenting unsupported claims as verified facts.

No numerical scores or final test results are reported here because the measured results have not been provided.

---

## 16. Installation and Local Setup

### Prerequisites

- Python installed on the computer.
- Git, if using the command line to manage the repository.
- Required project source files.
- A complete fine-tuned LED model checkpoint.
- Sufficient memory and storage for the model and its dependencies.

### Step 1: Clone the repository

After creating the GitHub repository, run:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Legal_AI_Summarization
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### Step 2: Create a virtual environment

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell prevents environment activation, follow the Python documentation for your environment or use Command Prompt activation instead.

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure that `requirements.txt` contains the actual dependencies required by your application and backend modules.

### Step 4: Verify the model files

Check that the configured model directory contains the model configuration and valid model weights.

For a typical Transformers checkpoint, the directory includes files such as:

```text
checkpoint-147/
├── config.json
├── model.safetensors
└── tokenizer files, if saved with the checkpoint
```

The exact filenames depend on how the model was saved. Some checkpoints use `pytorch_model.bin` or sharded model weights instead.

Do not assume the model is complete merely because the checkpoint directory exists.

### Step 5: Run the Streamlit application

From the project root:

```bash
python -m streamlit run src/app.py
```

Streamlit will display the local application address in the terminal.

---

## 17. GitHub Repository Setup

GitHub is used to store and version-control the project source code.

### Recommended repository contents

- Application source code.
- Supporting backend modules.
- `requirements.txt`.
- `README.md`.
- Tests and non-sensitive sample files.
- Configuration templates that contain no secrets.

### Files to exclude

Do not upload:

- API keys, passwords, access tokens, or private credentials.
- Private or confidential legal documents.
- Sensitive datasets without permission.
- Virtual environments and Python cache files.
- Temporary logs and generated outputs that are not intended for publication.
- Large model files unless their size and hosting method are appropriate.

A model checkpoint may contain large binary files. If the model exceeds GitHub's normal file limits, consider Git LFS or a suitable model-hosting service.

Create a `.gitignore` file in the project root, for example:

```gitignore
__pycache__/
*.py[cod]
.venv/
venv/
.env
.streamlit/secrets.toml
logs/
outputs/
```

Review the rules before committing to ensure you are not excluding files required by your application.

---

## 18. Streamlit Deployment

The application can potentially be deployed using Streamlit Community Cloud after the local app and model-loading process are working correctly.

### Deployment workflow

1. Push the project source code to GitHub.
2. Confirm that `requirements.txt` is complete.
3. Determine how the trained LED model will be hosted and accessed.
4. Open Streamlit Community Cloud.
5. Sign in using GitHub.
6. Create a new app and select the repository.
7. Select the branch containing the project.
8. Set the main file path to `src/app.py`.
9. Configure the Python version and other supported settings.
10. Deploy and inspect the application logs.

### Model hosting considerations

A fine-tuned LED model may require significant storage and memory. Before deploying, confirm:

- The total size of the model weights.
- Whether the hosting environment can accommodate model loading.
- Whether the model files are accessible to the application.
- Whether inference fits within available memory and execution limits.
- Whether model downloads and startup time are acceptable.

Large model weights may need to be hosted separately from the source-code repository. Do not place secrets or private documents in a public repository.

Deployment should be attempted only after the complete model loads successfully in the local environment.

---

## 19. Advantages

- Automates an initial summarization step for legal-document review.
- Uses a transformer architecture intended for long input sequences.
- Provides a dedicated interface for document processing.
- Combines summarization with supporting document-analysis workspaces.
- Supports source-based verification of generated information.
- Provides a foundation for integrating more advanced agentic workflows.
- Can be extended with improved retrieval, evaluation, and inference strategies.

---

## 20. Limitations

- Summary quality depends on the model, fine-tuning data, and inference settings.
- Long documents require careful input-length management.
- PDF extraction can be incomplete for scanned documents and complex layouts.
- Generated summaries may omit important conditions or exceptions.
- AI-generated statements can be incorrect or unsupported.
- Evidence-search and claim-review reliability depends on their actual backend implementation.
- Advanced autonomous-agent behavior should not be assumed unless implemented and tested.
- Model inference may require substantial memory and processing resources.
- The trained model weights must be present and accessible.
- The application does not replace professional legal advice or qualified legal review.

---

## 21. Future Enhancements

Potential improvements include:

1. Improving the quality and factual consistency of generated summaries.
2. Evaluating alternative long-context transformer architectures.
3. Adding systematic evaluation using suitable legal-document datasets and reference summaries.
4. Improving support for scanned PDFs through OCR.
5. Adding clause-level evidence retrieval with source references.
6. Developing tool-using agents for multi-step document-review tasks.
7. Adding structured claim verification and uncertainty handling.
8. Improving error handling and automated testing.
9. Optimizing inference memory usage and processing speed.
10. Hosting large model files through a suitable model-storage service.
11. Deploying the application on Streamlit Community Cloud or another appropriate platform.
12. Improving the interface for lengthy documents and detailed analysis results.

---

## 22. Conclusion

The project, **“A fine-tuned transformer summarization model-Based Approach to Legal document summarization,”** explores the application of Generative AI and AI-assisted workflows to the problem of reviewing lengthy legal documents.

The proposed system uses a fine-tuned LED transformer model for legal-text summarization and a Streamlit interface to provide document processing, evidence search, claim review, and paralegal-oriented analysis workspaces.

By combining generated summaries with supporting document-review capabilities, the project aims to make legal information easier to navigate and provide a foundation for more advanced AI-assisted workflows.

Further evaluation is necessary to establish summary quality, factual consistency, performance, and deployment suitability. Any conclusions drawn from the generated output should be verified against the original legal document.

---

## 23. Disclaimer

This project is intended for academic, research, and educational purposes. Generated summaries and analysis outputs may be incomplete or inaccurate and should not be treated as authoritative legal interpretations. Users should verify important information against the original document and consult qualified legal professionals when necessary.

The application is not a substitute for professional legal advice.

---

## 24. Project Information

- **Project Title:** A fine-tuned transformer summarization model-Based Approach to Legal document summarization
- **Domain:** Generative AI and Agentic AI
- **Application Area:** Legal Document Summarization and Analysis
- **Primary Model:** Fine-Tuned Longformer Encoder-Decoder (LED)
- **Programming Language:** Python
- **Application Framework:** Streamlit
- **Deep-Learning Framework:** PyTorch
