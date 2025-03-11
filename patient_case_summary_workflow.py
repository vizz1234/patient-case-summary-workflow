from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
import json
from datetime import datetime
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
import asyncio
import streamlit as st
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Context,
    Workflow,
    step,
)
from llama_index.core.retrievers import BaseRetriever
from pathlib import Path
import logging
import os
from llama_index.utils.workflow import draw_all_possible_flows
from IPython.display import clear_output
import asyncio
from streamlit_lottie import st_lottie
import requests
import tempfile

class ConditionInfo(BaseModel):
    code: str
    display: str
    clinical_status: str

class EncounterInfo(BaseModel):
    date: str = Field(..., description="Date of the encounter.")
    reason_display: Optional[str] = Field(None, description="Reason for the encounter.")
    type_display: Optional[str] = Field(None, description="Type or class of the encounter.")

class MedicationInfo(BaseModel):
    name: str = Field(..., description="Name of the medication.")
    start_date: Optional[str] = Field(None, description="When the medication was prescribed.")
    instructions: Optional[str] = Field(None, description="Dosage instructions.")

class PatientInfo(BaseModel):
    given_name: str
    family_name: str
    birth_date: str
    gender: str
    conditions: List[ConditionInfo] = Field(default_factory=list)
    recent_encounters: List[EncounterInfo] = Field(default_factory=list, description="A few recent encounters.")
    current_medications: List[MedicationInfo] = Field(default_factory=list, description="Current active medications.")

    @property
    def demographic_str(self) -> str:
        """Get demographics string."""
        return f"""\
                Given name: {self.given_name}
                Family name: {self.family_name}
                Birth date: {self.birth_date}
                Gender: {self.gender}"""
def parse_synthea_patient(file_path: str, filter_active: bool = True) -> PatientInfo:
    # Load the Synthea-generated FHIR Bundle
    with open(file_path, "r") as f:
        bundle = json.load(f)

    patient_resource = None
    conditions = []
    encounters = []
    medication_requests = []

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")

        if resource_type == "Patient":
            patient_resource = resource
        elif resource_type == "Condition":
            conditions.append(resource)
        elif resource_type == "Encounter":
            encounters.append(resource)
        elif resource_type == "MedicationRequest":
            medication_requests.append(resource)

    if not patient_resource:
        raise ValueError("No Patient resource found in the provided file.")

    # Extract patient demographics
    name_entry = patient_resource.get("name", [{}])[0]
    given_name = name_entry.get("given", [""])[0]
    family_name = name_entry.get("family", "")
    birth_date = patient_resource.get("birthDate", "")
    gender = patient_resource.get("gender", "")

    # Define excluded conditions
    excluded_conditions = {"Medication review due (situation)", "Risk activity involvement (finding)"}
    condition_info_list = []
    for c in conditions:
        code_info = c.get("code", {}).get("coding", [{}])[0]
        condition_code = code_info.get("code", "Unknown")
        condition_display = code_info.get("display", "Unknown")
        clinical_status = (
            c.get("clinicalStatus", {})
             .get("coding", [{}])[0]
             .get("code", "unknown")
        )

        # Check exclusion and active filters
        if condition_display not in excluded_conditions:
            if filter_active:
                if clinical_status == "active":
                    condition_info_list.append(
                        ConditionInfo(
                            code=condition_code,
                            display=condition_display,
                            clinical_status=clinical_status
                        )
                    )
            else:
                # Include conditions regardless of their status if filter_active is False
                condition_info_list.append(
                    ConditionInfo(
                        code=condition_code,
                        display=condition_display,
                        clinical_status=clinical_status
                    )
                )

    # Parse encounters
    def get_encounter_date(enc):
        period = enc.get("period", {})
        start = period.get("start")
        return datetime.fromisoformat(start) if start else datetime.min

    encounters_sorted = sorted(encounters, key=get_encounter_date)
    recent_encounters = encounters_sorted[-3:] if len(encounters_sorted) > 3 else encounters_sorted

    encounter_info_list = []
    for e in recent_encounters:
        period = e.get("period", {})
        start_date = period.get("start", "")
        reason = e.get("reasonCode", [{}])[0].get("coding", [{}])[0].get("display", None)
        etype = e.get("type", [{}])[0].get("coding", [{}])[0].get("display", None)
        encounter_info_list.append(
            EncounterInfo(
                date=start_date,
                reason_display=reason,
                type_display=etype
            )
        )

    # Parse medications
    medication_info_list = []
    for m in medication_requests:
        status = m.get("status")
        if status == "active":
            med_code = m.get("medicationCodeableConcept", {}).get("coding", [{}])[0]
            med_name = med_code.get("display", "Unknown Medication")
            authored = m.get("authoredOn", None)
            dosage_instruction = m.get("dosageInstruction", [{}])[0].get("text", None)
            medication_info_list.append(
                MedicationInfo(
                    name=med_name,
                    start_date=authored,
                    instructions=dosage_instruction
                )
            )

    patient_info = PatientInfo(
        given_name=given_name,
        family_name=family_name,
        birth_date=birth_date,
        gender=gender,
        conditions=condition_info_list,
        recent_encounters=encounter_info_list,
        current_medications=medication_info_list
    )

    return patient_info
patient_info = parse_synthea_patient("data/almeta_buckridge.json")
patient_info.model_dump()
class ConditionBundle(BaseModel):
    condition: ConditionInfo
    encounters: List[EncounterInfo] = Field(default_factory=list)
    medications: List[MedicationInfo] = Field(default_factory=list)

class ConditionBundles(BaseModel):
    bundles: List[ConditionBundle]
CONDITION_BUNDLE_PROMPT = """\
You are an assistant that takes a patient's summarized clinical data and associates each active condition with any relevant recent encounters and current medications.

**Steps to follow:**
1. Review the patient's demographics, conditions, recent encounters, and current medications.
2. For each condition in 'conditions':
   - Determine which of the 'recent_encounters' are relevant. An encounter is relevant if:
     - The 'reason_display' or 'type_display' of the encounter mentions or is closely related to the condition.
     - Consider synonyms or partial matches. For example, for "Childhood asthma (disorder)", any encounter mentioning "asthma" or "asthma follow-up" is relevant.
   - Determine which of the 'current_medications' are relevant. A medication is relevant if:
     - The medication 'name' or 'instructions' are clearly related to managing that condition. For example, inhalers or corticosteroids for asthma, topical creams for dermatitis.
     - Consider partial matches. For "Atopic dermatitis (disorder)", a medication used for allergic conditions or skin inflammations could be relevant.
3. Ignore patient demographics for relevance determination; they are just context.
4. Return the final output strictly as a JSON object following the schema (provided as a tool call).
   Do not include extra commentary outside the JSON.

**Patient Data**:
{patient_info}
"""

async def create_condition_bundles(
    patient_data: PatientInfo, llm: Optional[LLM] = None
) -> ConditionBundles:
    llm = llm or OpenAI(model="gpt-4o-mini", api_key=st.secrets["api_keys"]["OPENAI_API_KEY"])

    # we will dump the entire patient info into an LLM and have it figure out the relevant encounters/medications
    # associated with each condition
    prompt = ChatPromptTemplate.from_messages([
        ("user", CONDITION_BUNDLE_PROMPT)
    ])
    condition_bundles = await llm.astructured_predict(
        ConditionBundles,
        prompt,
        patient_info=patient_data.json()
    )

    return condition_bundles
# condition_bundles = await create_condition_bundles(patient_info)
# condition_bundles.model_dump()
index = LlamaCloudIndex(
  name=st.secrets["llama_cloud"]["INDEX_NAME"],
  project_name=st.secrets["llama_cloud"]["PROJECT_NAME"],
  organization_id=st.secrets["llama_cloud"]["ORGANIZATION_ID"],
  api_key=st.secrets["llama_cloud"]["LLAMA_CLOUD_API_KEY"]
)

retriever = index.as_retriever(similarity_top_k=3)
class GuidelineQueries(BaseModel):
    """Represents a set of recommended queries to retrieve guideline sections relevant to the patient's conditions."""
    queries: List[str] = Field(
        default_factory=list,
        description="A list of query strings that can be used to search a vector index of medical guidelines."
    )
class GuidelineRecommendation(BaseModel):
    guideline_source: str = Field(..., description="The origin of the guideline (e.g., 'NHLBI Asthma Guidelines').")
    recommendation_summary: str = Field(..., description="A concise summary of the relevant recommendation.")
    reference_section: Optional[str] = Field(None, description="Specific section or reference in the guideline.")
class ConditionSummary(BaseModel):
    condition_display: str = Field(..., description="Human-readable name of the condition.")
    summary: str = Field(..., description="A concise narrative summarizing the condition's status, relevant encounters, medications, and guideline recommendations.")

class CaseSummary(BaseModel):
    patient_name: str = Field(..., description="The patient's name.")
    age: int = Field(..., description="The patient's age in years.")
    overall_assessment: str = Field(..., description="A high-level summary synthesizing all conditions, encounters, medications, and guideline recommendations.")
    condition_summaries: List[ConditionSummary] = Field(
        default_factory=list,
        description="A list of condition-specific summaries providing insight into each condition's current management and recommendations."
    )

    def render(self) -> str:
        lines = []
        lines.append(f"Patient Name: {self.patient_name}")
        lines.append(f"Age: {self.age} years")
        lines.append("")
        lines.append("Overall Assessment:")
        lines.append(self.overall_assessment)
        lines.append("")

        if self.condition_summaries:
            lines.append("Condition Summaries:")
            for csum in self.condition_summaries:
                lines.append(f"- {csum.condition_display}:")
                lines.append(f"  {csum.summary}")
        else:
            lines.append("No specific conditions were summarized.")

        return "\n".join(lines)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)



GUIDELINE_QUERIES_PROMPT = """\
You are an assistant tasked with determining what guidelines would be most helpful to consult for a given patient's condition data. You have:

- Patient information (demographics, conditions, encounters, medications)
- A single condition bundle that includes:
  - One specific condition and its related encounters and medications
- Your goal is to produce several high-quality search queries that can be used to retrieve relevant guideline sections from a vector index of medical guidelines.

**Instructions:**
1. Review the patient info and the condition bundle. Identify the key aspects of the condition that might require guideline consultation—such as disease severity, typical management steps, trigger avoidance, or medication optimization.
2. Consider what clinicians would look up:
   - Best practices for this condition's management (e.g., stepwise therapy for asthma, maintenance therapy for atopic dermatitis)
   - Medication recommendations (e.g., use of inhaled corticosteroids, timing and dose adjustments, rescue inhaler usage, antihistamines for atopic dermatitis)
   - Encounter follow-ups (e.g., what follow-up intervals are recommended, what tests or measurements to track)
   - Patient education and preventive measures (e.g., trigger avoidance, skincare routines, inhaler technique)
3. Formulate 3-5 concise, targeted queries that, if run against a medical guideline index, would return the most relevant sections. Each query should be a natural language string that could be used with a vector-based retrieval system.
4. Make the queries condition-specific, incorporating relevant medications or encounter findings.
5. Return the output as a JSON object following the schema defined as a tool call.

Patient Info: {patient_info}

Condition Bundle: {condition_info}

Do not include any commentary outside the JSON."""


GUIDELINE_RECOMMENDATION_PROMPT = """\
Given the following patient condition and the corresponding relevant medical guideline text (unformatted),
generate a guideline recommendation according to the schema defined as a tool call.

The condition details are given below. This includes the condition itself, along with associated encounters/medications
that the patient has taken already. Make sure the guideline recommendation is relevant.

**Patient Condition:**
{patient_condition_text}

**Matched Guideline Text(s):**
{guideline_text}
"""


CASE_SUMMARY_SYSTEM_PROMPT = """\
You are a medical assistant that produces a concise and understandable case summary for a clinician.

You have access to the patient's name, age, and a list of conditions.

For each condition, you also have related encounters, medications, and guideline recommendations.

Your goal is to produce a `CaseSummary` object in JSON format that adheres to the CaseSummary schema, defined as a tool call.

**Instructions:**
- Use the patient's name and age as given.
- Create an `overall_assessment` that integrates the data about their conditions, encounters, medications, and guideline recommendations.
- For each condition, write a short `summary` describing:
  - The current state of the condition.
  - Relevant encounters that indicate progress or issues.
  - Medications currently managing that condition and if they align with guidelines.
  - Any key recommendations from the guidelines that should be followed going forward.
- Keep the summaries patient-friendly but medically accurate. Be concise and clear.
- Return only the final JSON that matches the schema. No extra commentary.

"""

CASE_SUMMARY_USER_PROMPT = """\
**Patient Demographics**
{demographic_info}

**Condition Information**
{condition_guideline_info}


Given the above data, produce a `CaseSummary` as per the schema.
"""

def generate_condition_guideline_str(
    bundle: ConditionBundle,
    rec: GuidelineRecommendation
) -> str:
    return f"""\
**Condition Info**:
{bundle.json()}

**Recommendation**:
{rec.json()}
"""


class PatientInfoEvent(Event):
    patient_info: PatientInfo


class ConditionBundleEvent(Event):
    bundles: ConditionBundles


class MatchGuidelineEvent(Event):
    bundle: ConditionBundle


class MatchGuidelineResultEvent(Event):
    bundle: ConditionBundle
    rec: GuidelineRecommendation


class GenerateCaseSummaryEvent(Event):
    condition_guideline_info: List[Tuple[ConditionBundle, GuidelineRecommendation]]


class LogEvent(Event):
    msg: str
    delta: bool = False


class GuidelineRecommendationWorkflow(Workflow):
    """Guidline recommendation workflow."""

    def __init__(
        self,
        guideline_retriever: BaseRetriever,
        llm: LLM | None = None,
        similarity_top_k: int = 20,
        output_dir: str = "data_out",
        **kwargs,
    ) -> None:
        """Init params."""
        super().__init__(**kwargs)

        self.guideline_retriever = guideline_retriever

        self.llm = llm or OpenAI(model="gpt-4o-mini")
        self.similarity_top_k = similarity_top_k

        # if not exists, create
        out_path = Path(output_dir) / "workflow_output"
        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=True)
            os.chmod(str(out_path), 0o0777)
        self.output_dir = out_path

    @step
    async def parse_patient_info(
        self, ctx: Context, ev: StartEvent
    ) -> PatientInfoEvent:
        # load patient info from cache if exists, otherwise generate
        patient_info_path = Path(
            f"{self.output_dir}/patient_info.json"
        )
        if patient_info_path.exists():
            if self._verbose:
                ctx.write_event_to_stream(LogEvent(msg=">> Loading patient info from cache"))
            patient_info_dict = json.load(open(str(patient_info_path), "r"))
            patient_info = PatientInfo.model_validate(patient_info_dict)
        else:
            if self._verbose:
                ctx.write_event_to_stream(LogEvent(msg=">> Reading patient info"))
            patient_info = parse_synthea_patient(ev.patient_json_path)

            if not isinstance(patient_info, PatientInfo):
                raise ValueError(f"Invalid patient info: {patient_info}")
            # save patient info to file
            with open(patient_info_path, "w") as fp:
                fp.write(patient_info.model_dump_json())
        if self._verbose:
            ctx.write_event_to_stream(LogEvent(msg=f">> Patient Info: {patient_info.dict()}"))

        await ctx.set("patient_info", patient_info)

        return PatientInfoEvent(patient_info=patient_info)

    @step
    async def create_condition_bundles(
        self, ctx: Context, ev: PatientInfoEvent
    ) -> ConditionBundleEvent:
        """Create condition bundles."""
        # load patient condition info from cache if exists, otherwise generate
        condition_info_path = Path(
            f"{self.output_dir}/condition_bundles.json"
        )
        if condition_info_path.exists():
            condition_bundles = ConditionBundles.model_validate(
                json.load(open(str(condition_info_path), "r"))
            )
        else:
            condition_bundles = await create_condition_bundles(ev.patient_info)
            with open(condition_info_path, "w") as fp:
                fp.write(condition_bundles.model_dump_json())

        return ConditionBundleEvent(bundles=condition_bundles)

    @step
    async def dispatch_guideline_match(
        self, ctx: Context, ev: ConditionBundleEvent
    ) -> MatchGuidelineEvent:
        """For each condition + associated information, find relevant guidelines.

        Use a map-reduce pattern.

        """
        await ctx.set("num_conditions", len(ev.bundles.bundles))

        for bundle in ev.bundles.bundles:
            ctx.send_event(MatchGuidelineEvent(bundle=bundle))

    @step
    async def handle_guideline_match(
        self, ctx: Context, ev: MatchGuidelineEvent
    ) -> MatchGuidelineResultEvent:
        """Generate guideline recommendation for each condition."""
        patient_info = await ctx.get("patient_info")

        # We will first generate the right set of questions to ask given the patient info.
        prompt = ChatPromptTemplate.from_messages([
            ("user", GUIDELINE_QUERIES_PROMPT)
        ])
        guideline_queries = await llm.astructured_predict(
            GuidelineQueries,
            prompt,
            patient_info=patient_info.demographic_str,
            condition_info=ev.bundle.json()
        )

        guideline_docs_dict = {}
        # fetch all relevant guidelines as text
        for query in guideline_queries.queries:
            if self._verbose:
                ctx.write_event_to_stream(LogEvent(msg=f">> Generating query: {query}"))
            cur_guideline_docs = self.guideline_retriever.retrieve(query)
            guideline_docs_dict.update({
                d.id_: d for d in cur_guideline_docs
            })
        guideline_docs = guideline_docs_dict.values()
        guideline_text="\n\n".join([g.get_content() for g in guideline_docs])
        if self._verbose:
            ctx.write_event_to_stream(
                LogEvent(msg=f">> Found guidelines: {guideline_text[:200]}...")
            )

        # generate guideline recommendation
        prompt = ChatPromptTemplate.from_messages([
            ("user", GUIDELINE_RECOMMENDATION_PROMPT)
        ])
        guideline_rec = await llm.astructured_predict(
            GuidelineRecommendation,
            prompt,
            patient_info=patient_info.demographic_str,
            condition_info=ev.bundle.json(),
            guideline_text=guideline_text
        )
        if self._verbose:
            ctx.write_event_to_stream(
                LogEvent(msg=f">> Guideline recommendation: {guideline_rec.json()}")
            )

        if not isinstance(guideline_rec, GuidelineRecommendation):
            raise ValueError(f"Invalid guideline recommendation: {guideline_rec}")

        return MatchGuidelineResultEvent(bundle=ev.bundle, rec=guideline_rec)

    @step
    async def gather_guideline_match(
        self, ctx: Context, ev: MatchGuidelineResultEvent
    ) -> GenerateCaseSummaryEvent:
        """Handle matching clause against guideline."""
        num_conditions = await ctx.get("num_conditions")
        events = ctx.collect_events(ev, [MatchGuidelineResultEvent] * num_conditions)
        if events is None:
            return

        match_results = [(e.bundle, e.rec) for e in events]
        # save match results
        recs_path = Path(f"{self.output_dir}/guideline_recommendations.jsonl")
        with open(recs_path, "w") as fp:
            for _, rec in match_results:
                fp.write(rec.model_dump_json() + "\n")


        return GenerateCaseSummaryEvent(condition_guideline_info=match_results)

    @step
    async def generate_output(
        self, ctx: Context, ev: GenerateCaseSummaryEvent
    ) -> StopEvent:
        if self._verbose:
            ctx.write_event_to_stream(LogEvent(msg=">> Generating Case Summary"))

        patient_info = await ctx.get("patient_info")
        demographic_info = patient_info.demographic_str

        condition_guideline_strs = []
        for condition_bundle, guideline_rec in ev.condition_guideline_info:
            condition_guideline_strs.append(
                generate_condition_guideline_str(condition_bundle, guideline_rec)
            )
        condition_guideline_str = "\n\n".join(condition_guideline_strs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", CASE_SUMMARY_SYSTEM_PROMPT),
            ("user", CASE_SUMMARY_USER_PROMPT)
        ])
        case_summary = await llm.astructured_predict(
            CaseSummary,
            prompt,
            demographic_info=demographic_info,
            condition_guideline_info=condition_guideline_str
        )

        return StopEvent(result={"case_summary": case_summary})
llm = OpenAI(model="gpt-4o-mini", api_key=st.secrets["api_keys"]["OPENAI_API_KEY"])
workflow = GuidelineRecommendationWorkflow(
    guideline_retriever=retriever,
    llm=llm,
    verbose=True,
    timeout=None,  # don't worry about timeout to make sure it completes
)
draw_all_possible_flows(GuidelineRecommendationWorkflow, filename="data/guideline_rec_workflow.html")
async def run_workflow_async(patient_json_path):
    handler = workflow.run(patient_json_path="data/almeta_buckridge.json")

    async for event in handler.stream_events():
      if isinstance(event, LogEvent):
          if event.delta:
              print(event.msg, end="")
          else:
              print(event.msg)

    # Wait for the handler and get the final response
    response_dict = await handler

    return response_dict

# Add these imports for enhanced UI
from streamlit_lottie import st_lottie
import requests
from datetime import datetime

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Configure the page with a wider layout and custom theme
st.set_page_config(
    page_title="Patient Case Summary Generator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
    <style>
    /* Modern gradient background */
    .stApp {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 50%, #90caf9 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 0 10px 10px 0;
        padding: 2rem;
        margin: 0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1a365d;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Feature cards */
    .feature-card {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
    }
    
    /* Guide section */
    .guide-section {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Custom button styling */
    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
    }
    
    /* Upload area */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📤 Upload Patient Data")
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Choose a JSON file",
        type=['json'],
        help="Upload FHIR format patient data",
        key="patient_file_uploader"
    )
    
    if uploaded_file:
        st.success("✅ File uploaded successfully!")
        file_contents = uploaded_file.getvalue()
        st.session_state['file_contents'] = file_contents
        if st.button("�� Generate Summary", 
                    use_container_width=True,
                    key="generate_summary_button"):
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.json') as tmp_file:
                tmp_file.write(file_contents)
                st.session_state['temp_path'] = tmp_file.name

# Main content
st.title("🏥 Patient Case Summary Generator")

# Welcome screen
if not uploaded_file:
    # Introduction
    st.markdown("""
        <div class="feature-card">
            <h2 style="margin-top:0">Transform Patient Data into Clinical Insights</h2>
            <p style="font-size: 1.1em; color: #4a5568; margin-bottom: 1rem;">
                Our AI-powered system analyzes patient data to generate comprehensive clinical summaries,
                helping healthcare providers make informed decisions quickly and efficiently.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Features in a grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h3 style="color: #2563eb; margin-top:0">🤖 Smart Analysis</h3>
                <ul style="color: #4a5568; margin: 0; padding-left: 1.5rem;">
                    <li>Automated condition analysis</li>
                    <li>Real-time data processing</li>
                    <li>AI-powered insights</li>
                </ul>
            </div>
            
            <div class="feature-card">
                <h3 style="color: #2563eb; margin-top:0">📊 Comprehensive Reports</h3>
                <ul style="color: #4a5568; margin: 0; padding-left: 1.5rem;">
                    <li>Detailed clinical summaries</li>
                    <li>Clear data visualization</li>
                    <li>Easy-to-read formats</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h3 style="color: #2563eb; margin-top:0">📚 Evidence-Based</h3>
                <ul style="color: #4a5568; margin: 0; padding-left: 1.5rem;">
                    <li>Guideline-based recommendations</li>
                    <li>Up-to-date medical knowledge</li>
                    <li>Clinical best practices</li>
                </ul>
            </div>
            
            <div class="feature-card">
                <h3 style="color: #2563eb; margin-top:0">🔒 Secure & Reliable</h3>
                <ul style="color: #4a5568; margin: 0; padding-left: 1.5rem;">
                    <li>Secure data processing</li>
                    <li>Data privacy focused</li>
                    <li>Reliable analysis</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Quick Guide
    st.markdown("""
        <div class="guide-section">
            <h3 style="color: #2563eb; margin-top:0">📋 Quick Start Guide</h3>
            <ol style="color: #4a5568; margin: 0; padding-left: 1.5rem;">
                <li style="margin-bottom: 0.5rem">Upload your patient JSON file using the sidebar</li>
                <li style="margin-bottom: 0.5rem">Click "Generate Summary" to start the analysis</li>
                <li style="margin-bottom: 0.5rem">Review the generated clinical summary and patient details</li>
                <li style="margin-bottom: 0.5rem">Download the report for your records</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

# Processing and results display
else:
    async def process_patient():
        try:
            # Create columns for progress tracking
            col1, col2 = st.columns([3, 1])
            
            with col1:
                progress_bar = st.progress(0)
                status = st.empty()
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.json') as tmp_file:
                    # Write the contents from session state
                    tmp_file.write(st.session_state['file_contents'])
                    temp_path = tmp_file.name
                
                # Parse patient info
                status.info("🔄 Analyzing patient data...")
                patient_info = parse_synthea_patient(temp_path)
                progress_bar.progress(25)
                
                # Create condition bundles
                status.info("🔍 Processing medical conditions...")
                condition_bundles = await create_condition_bundles(patient_info)
                progress_bar.progress(50)
                
                # Set up LlamaCloud
                status.info("🌐 Retrieving guidelines...")
                index = LlamaCloudIndex(
                    name=st.secrets["llama_cloud"]["INDEX_NAME"],
                    project_name=st.secrets["llama_cloud"]["PROJECT_NAME"],
                    organization_id=st.secrets["llama_cloud"]["ORGANIZATION_ID"],
                    api_key=st.secrets["llama_cloud"]["LLAMA_CLOUD_API_KEY"]
                )
                retriever = index.as_retriever(similarity_top_k=3)
                progress_bar.progress(75)
                
                # Run workflow
                status.info("📝 Generating comprehensive summary...")
                llm = OpenAI(model="gpt-4o-mini", api_key=st.secrets["api_keys"]["OPENAI_API_KEY"])
                workflow = GuidelineRecommendationWorkflow(
                    guideline_retriever=retriever,
                    llm=llm,
                    verbose=True,
                    timeout=None,
                )
                response_dict = await run_workflow_async(temp_path)
                progress_bar.progress(100)
                status.success("✅ Analysis Complete!")

            # Results display
            st.markdown("### 📊 Analysis Results")
            tab1, tab2 = st.tabs(["📑 Clinical Summary", "👤 Patient Details"])
            
            with tab1:
                st.markdown("""
                <div class="card">
                    <h3>Clinical Summary Report</h3>
                    <div style="margin-top: 1rem;">
                """, unsafe_allow_html=True)
                st.markdown(response_dict["case_summary"].render())
                st.markdown("</div></div>", unsafe_allow_html=True)
                
                # Download button
                st.download_button(
                    label="📥 Download Full Report",
                    data=response_dict["case_summary"].render(),
                    file_name="patient_case_summary.txt",
                    mime="text/plain",
                    key="download_report_button"
                )
            
            with tab2:
                st.markdown("""
                <div class="card">
                    <h3>Patient Profile</h3>
                """, unsafe_allow_html=True)
                
                # Demographics section
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div class="info-section">
                        <h4>📋 Demographics</h4>
                        <table class="patient-table">
                            <tr><td><strong>Name</strong></td><td>{} {}</td></tr>
                            <tr><td><strong>Birth Date</strong></td><td>{}</td></tr>
                            <tr><td><strong>Gender</strong></td><td>{}</td></tr>
                        </table>
                    </div>
                    """.format(
                        patient_info.given_name,
                        patient_info.family_name,
                        patient_info.birth_date,
                        patient_info.gender.capitalize()
                    ), unsafe_allow_html=True)

                # Add CSS for the patient details
                st.markdown("""
                <style>
                    .info-section {
                        background-color: rgba(255, 255, 255, 0.8);
                        border-radius: 8px;
                        padding: 1rem;
                        margin-bottom: 1rem;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    }
                    
                    .info-section h4 {
                        color: #2c3e50;
                        margin-bottom: 1rem;
                        border-bottom: 2px solid #e3f2fd;
                        padding-bottom: 0.5rem;
                    }
                    
                    .patient-table {
                        width: 100%;
                        border-collapse: separate;
                        border-spacing: 0 0.5rem;
                    }
                    
                    .patient-table td {
                        padding: 0.5rem;
                    }
                    
                    .patient-table tr:hover {
                        background-color: rgba(227, 242, 253, 0.3);
                    }
                    
                    .condition-card {
                        background-color: rgba(255, 255, 255, 0.8);
                        border-radius: 8px;
                        padding: 1rem;
                        margin-bottom: 0.5rem;
                        border-left: 4px solid #90caf9;
                    }
                    
                    .medication-card {
                        background-color: rgba(255, 255, 255, 0.8);
                        border-radius: 8px;
                        padding: 1rem;
                        margin-bottom: 0.5rem;
                        border-left: 4px solid #81c784;
                    }
                    
                    .encounter-card {
                        background-color: rgba(255, 255, 255, 0.8);
                        border-radius: 8px;
                        padding: 1rem;
                        margin-bottom: 0.5rem;
                        border-left: 4px solid #ffb74d;
                    }
                    
                    .section-title {
                        color: #2c3e50;
                        margin: 1.5rem 0 1rem 0;
                        font-size: 1.1rem;
                        font-weight: 600;
                    }
                </style>
                """, unsafe_allow_html=True)

                # Conditions section
                st.markdown("<div class='section-title'>🏥 Active Conditions</div>", unsafe_allow_html=True)
                for condition in patient_info.conditions:
                    st.markdown(f"""
                    <div class="condition-card">
                        <strong>{condition.display}</strong><br>
                        <small>Status: {condition.clinical_status.capitalize()}</small>
                    </div>
                    """, unsafe_allow_html=True)

                # Medications section
                st.markdown("<div class='section-title'>💊 Current Medications</div>", unsafe_allow_html=True)
                for med in patient_info.current_medications:
                    st.markdown(f"""
                    <div class="medication-card">
                        <strong>{med.name}</strong><br>
                        {f"<small>Started: {med.start_date}</small><br>" if med.start_date else ""}
                        {f"<small>{med.instructions}</small>" if med.instructions else ""}
                    </div>
                    """, unsafe_allow_html=True)

                # Recent Encounters section
                st.markdown("<div class='section-title'>👨‍⚕️ Recent Encounters</div>", unsafe_allow_html=True)
                for encounter in patient_info.recent_encounters:
                    st.markdown(f"""
                    <div class="encounter-card">
                        <strong>Date: {encounter.date}</strong><br>
                        {f"<small>Type: {encounter.type_display}</small><br>" if encounter.type_display else ""}
                        {f"<small>Reason: {encounter.reason_display}</small>" if encounter.reason_display else ""}
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_path)
            except:
                pass

    if st.sidebar.button("🔄 Generate Summary", use_container_width=True):
        asyncio.run(process_patient())

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Powered by LlamaIndex and OpenAI | For clinical use only</p>
    </div>
    """, unsafe_allow_html=True)