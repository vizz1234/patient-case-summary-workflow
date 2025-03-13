# Patient Case Summary Workflow 🏥

A sophisticated medical data processing system that generates comprehensive clinical summaries by analyzing patient data, medical conditions, and relevant guidelines using advanced LLM and vector search technologies.

## 🔷 High-Level Overview

Below is an architectural overview of the system:

```
FHIR Patient Data (JSON) 
       ↓
Data Extraction & Parsing
       ↓
Patient Information Structure
       ↓
Condition Bundle Creation (LLM)
       ↓
LlamaIndex Retrieval
       ├─→ Medical Guidelines Search
       └─→ Relevant Section Matching
       ↓
Guideline Analysis (LLM)
       ↓
Case Summary Generation
       |
       ├─→ Overall Assessment
       ├─→ Condition Summaries
       ├─→ Medication Analysis
       └─→ Encounter History
       ↓
Final Output Formats
```

## 🌟 Features

### Medical Case Summary System
- **Automated Case Summary Generation**: Transform complex medical records into clear, structured summaries
  - Process FHIR-formatted patient data
  - Extract relevant medical information
  - Generate comprehensive clinical summaries
- **Medical Guidelines Integration**: Automatically match patient conditions with relevant medical guidelines
  - LlamaIndex-powered guideline search
  - Contextual matching of conditions
  - Evidence-based recommendations
- **Smart Condition Bundling**: Group related medical events and treatments
  - Intelligent condition association
  - Treatment timeline organization
  - Medication correlation

## 🚀 Getting Started

### Prerequisites
```
python >= 3.8
streamlit
llama-index
pydantic
fpdf
```

### Installation

1. Clone the repository
```bash
git clone https://github.com/vizz1234/patient-case-summary-workflow.git
cd patient-case-summary-workflow
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up your environment variables in `.streamlit/secrets.toml`:
```toml
[api_keys]
OPENAI_API_KEY = "your-openai-key"
PHOENIX_API_KEY = "your-phoenix-key"

[llama_cloud]
INDEX_NAME = "your-index-name"
PROJECT_NAME = "your-project-name"
ORGANIZATION_ID = "your-org-id"
LLAMA_CLOUD_API_KEY = "your-llama-cloud-key"
```

4. Run the application
```bash
streamlit run patient_case_summary_workflow.py
```

## 💡 Usage

### Sample Workflow
1. Upload FHIR patient data
2. System processes and bundles conditions
3. Retrieves relevant medical guidelines
4. Generates comprehensive case summary

### Example Outputs
- Structured clinical summaries
- Condition-specific assessments
- Treatment recommendations
- PDF reports

## 🏗 Project Structure

```
patient-case-summary-workflow/
├── .streamlit/                      # Streamlit configuration
│   └── secrets.toml                 # API keys and configuration secrets
├── data/                           # Input data directory
│   └── almeta_buckridge.json       # Sample FHIR patient data
├── patient_case_summary_workflow.py  # Main workflow implementation
├── data_out/
│   └── workflow_output/             # Generated outputs
│       ├── patient_info.json
│       ├── condition_bundles.json
│       └── guideline_recommendations.jsonl
├── requirements.txt                 # Project dependencies
└── README.md                       # Project documentation
```

## 🔒 Security

- API keys and sensitive information are stored in `.streamlit/secrets.toml`
- The secrets file is excluded from version control
- Environment variables are used for deployment
- Different keys should be used for development and production

## 🛠 Technical Details

### Components
- **Streamlit**: Web interface and user interaction
- **LlamaIndex**: Medical guideline retrieval and matching
- **OpenAI**: Natural language processing and summary generation
- **Pydantic**: Data validation and serialization

### Tools
1. **FHIR Parser**
   - Processes standardized medical data
   - Extracts patient information
   - Organizes medical history

2. **Condition Bundler**
   - Groups related medical events
   - Associates treatments and medications
   - Creates coherent condition summaries

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---
**Note**: This tool is for informational purposes only and should not be used as a substitute for professional medical judgment.








