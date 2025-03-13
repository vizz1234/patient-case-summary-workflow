# Patient Case Summary Workflow 🏥

A sophisticated medical data processing system that generates comprehensive clinical summaries by analyzing patient data, medical conditions, and relevant guidelines using advanced LLM and vector search technologies.

## 🌟 Key Features

- **Automated Case Summary Generation**: Transform complex medical records into clear, structured summaries
- **Medical Guidelines Integration**: Automatically match patient conditions with relevant medical guidelines
- **Smart Condition Bundling**: Group related medical events and treatments by condition
- **Multiple Output Formats**: Generate both PDF and text-based reports
- **FHIR Compatible**: Process standard FHIR-formatted patient data

## 🏗️ Architecture

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

## 🛠️ Technical Components

### Data Models
- `PatientInfo`: Core patient demographic and medical data
- `ConditionBundle`: Logical grouping of conditions and related medical events
- `CaseSummary`: Structured format for final generated summaries

### Key Technologies
- **LlamaIndex**: Semantic search and retrieval of medical guidelines
- **OpenAI GPT**: Natural language processing and summary generation
- **Pydantic**: Data validation and serialization
- **FPDF**: PDF report generation

## 📋 Prerequisites

```bash
# Python 3.8 or higher required
python -m pip install -r requirements.txt
```

Required environment variables:

```bash
OPENAI_API_KEY=your_api_key
LLAMA_CLOUD_API_KEY=your_llama_cloud_key
```

## 🚀 Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Set Up Environment**
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Run the Application**
```bash
python patient_case_summary_workflow.py
```

## 💡 Usage Example

```python
from patient_case_summary_workflow import CaseSummaryWorkflow

# Initialize workflow
workflow = CaseSummaryWorkflow()

# Process patient data
summary = await workflow.process_patient_data("patient_data.json")

# Generate report
summary.to_pdf("patient_summary.pdf")
```

## 🔍 Condition Bundles

Condition Bundles are intelligent groupings of medical information:

```python
{
    "condition": "Asthma",
    "encounters": ["Asthma follow-up", "Emergency visit"],
    "medications": ["Albuterol inhaler", "Oral steroids"]
}
```

This structure helps in:
- Organizing related medical events
- Matching relevant guidelines
- Generating coherent summaries

## 📊 Output Examples

### Summary Structure
```
CLINICAL CASE SUMMARY
====================
PATIENT: John Doe (Age: 45)

OVERALL ASSESSMENT:
-----------------
[Comprehensive patient status]

CONDITION SUMMARIES:
------------------
1. Asthma
   - Current Status
   - Treatment Plan
   - Guideline Adherence
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- LlamaIndex team for vector search capabilities
- Medical guidelines databases
- FHIR standard documentation

## 📮 Contact

For questions and support, please open an issue in the GitHub repository.

---
**Note**: This tool is for informational purposes only and should not be used as a substitute for professional medical judgment.

## 📁 Project Structure

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

### Key Files

- **patient_case_summary_workflow.py**: Core implementation containing:
  - Data models (PatientInfo, ConditionBundle, etc.)
  - FHIR data parsing
  - LLM integration
  - Guideline retrieval
  - Case summary generation
  - Streamlit UI

- **.streamlit/secrets.toml**: Contains configuration for:
  - LlamaCloud API keys
  - Index configuration
  - Project settings
  - Organization details

- **data/**: Contains input FHIR-formatted patient data files
- **data_out/workflow_output/**: Generated analysis outputs and cached results

