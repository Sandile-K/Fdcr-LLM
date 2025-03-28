import streamlit as st
import logging
from datetime import datetime
import json
import requests
from neo4j import GraphDatabase

# ------------- CONFIG & LOGGING -------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEO4J_URI = "bolt://localhost:8868"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
OLLAMA_URL = "http://localhost:11434"
RESULTS_LIMIT = 100
TOKEN_LIMIT = 1000

# A quick approximation of token counting by splitting on whitespace
def count_tokens(text: str) -> int:
    return len(text.split())

# ------------- DEFAULT PROMPTS & TEMPLATES -------------
default_system_prompt = (
    "You are a senior financial and economic analyst specializing in R&D project analysis. "
    "Analyze the project context focusing on financial performance, budget allocation, impact assessment, "
    "and strategic objectives. Provide clear insights and actionable recommendations."
)

system_prompt_options = {
    "Default": default_system_prompt,
    "Brief": "Provide a concise answer highlighting key metrics and insights.",
    "Detailed": "Provide a comprehensive answer with background, detailed analysis, and actionable recommendations.",
    "Custom": ""
}

response_structure_templates = {
    "Executive Summary + Detailed Analysis + Recommendations": (
        "Start with an executive summary, followed by a detailed analysis, and conclude with actionable recommendations."
    ),
    "Detailed Analysis Only": "Provide only the detailed analysis in logical sections.",
    "Bullet Point Summary": "Present the response as bullet points focusing on key insights.",
    "Custom": ""
}

faq_options = [
    "Give me a breakdown of which projects are producing the most and least outputs for each KPI category",
    "Give me a breakdown of underperforming and overperforming projects?",
    "Tell me what factors are being considered for evaluating performance",
    "What national challenges are the different domains tackling?",
    "What are the biggest challenges experienced by projects in each domain?",
    "Provide a view of progress of each programme in each domain",
    "Provide a view of underperforming and overperforming programmes and the factors used to evaluate progress",
    "Which projects exhibit synergies in the national challenges being addressed",
    "Which projects have shared objectives that could benefit from collaboration",
    "What are the average cost of outputs (KPIs) for the various programmes and domains? Provide a breakdown of funding per programme and domain"
]

# ------------- STYLING -------------
# Set wide layout
st.set_page_config(layout="wide")

# Dark background and white text
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
    }
    .stTextInput > label, .stTextArea > label, .stSelectbox > label, .stMultiselect > label {
        color: #ffffff !important;
    }
    .stMarkdown, .css-10trblm, .css-18e3th9 {
        color: #ffffff !important;
    }
    .st-bk {
        background-color: #1e1e1e !important;
    }
    .stButton button {
        background-color: #2b2b2b !important;
        color: #ffffff !important;
        border: 1px solid #3c3c3c !important;
    }
    .stButton button:hover {
        background-color: #3c3c3c !important;
        color: #ffffff !important;
    }
    .model-response {
        border: 1px solid #ffffff;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------- CONNECTION TESTING -------------
def test_neo4j_connection(uri, user, password):
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
        return driver, count
    except Exception as e:
        return None, str(e)

def test_ollama_connection():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            return model_names, None
        else:
            return [], f"Ollama Connection Failed: {response.status_code}"
    except Exception as e:
        return [], str(e)

neo4j_driver, neo4j_msg = test_neo4j_connection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
available_models, ollama_msg = test_ollama_connection()

# ------------- HEADER / CONNECTION STATUS -------------
if neo4j_driver and available_models:
    st.success("Successfully connected to the database and LLM backend.")
else:
    error_msg = "Connection issues:\n"
    if not neo4j_driver:
        error_msg += f"- Neo4j Connection Failed: {neo4j_msg}\n"
    if not available_models:
        error_msg += f"- Ollama Connection Failed: {ollama_msg}\n"
    st.error(error_msg)

# ------------- MAIN LAYOUT -------------
# We'll use container or just place items in the main area in order

# 1. Model selection
st.subheader("Select Models")
selected_models = st.multiselect(
    "Choose exactly two models",
    available_models,
    max_selections=2,
    help="Select two different models to generate side-by-side responses."
)

# 8. LLM Parameters
if selected_models and len(selected_models) == 2:
    st.subheader("LLM Parameters")
    params = {}
    for model in selected_models:
        st.write(f"Parameters for {model}:")
        params[model] = {
            "temperature": st.slider(f"Temperature ({model})", 0.0, 1.0, 0.7, help="Controls the randomness of the output."),
            "max_tokens": st.number_input(f"Max Tokens ({model})", 1, 2000, 1000, help="Maximum number of tokens in the response."),
            "top_k": st.number_input(f"Top-k ({model})", 1, 100, 40, help="Controls the diversity of the output.")
        }

# 2. System Prompt Template
st.subheader("System Prompt Template")
chosen_prompt_option = st.selectbox(
    "Choose an option",
    list(system_prompt_options.keys()),
    help="Select a system prompt template."
)

if chosen_prompt_option == "Custom":
    system_prompt_text = st.text_area(
        "Custom System Prompt",
        value=default_system_prompt,
        help="Enter your custom system prompt here. (Max 1000 tokens)"
    )
else:
    system_prompt_text = system_prompt_options[chosen_prompt_option]

system_prompt_tokens = count_tokens(system_prompt_text)
st.text(f"System Prompt Token Count: {system_prompt_tokens} / {TOKEN_LIMIT}")
if system_prompt_tokens > TOKEN_LIMIT:
    st.error("System prompt exceeds the 1000-token limit. Please shorten it.")

# 3. Response Structure Template
st.subheader("Response Structure Template")
selected_structure = st.selectbox(
    "Select a structure",
    list(response_structure_templates.keys()),
    help="Choose a suggested structure or pick 'Custom' to provide your own."
)

custom_structure = ""
if selected_structure == "Custom":
    custom_structure = st.text_area(
        "Custom Response Structure",
        "",
        help="Describe how you'd like the response to be structured."
    )

# 4. Specific Metrics / Information Request
st.subheader("Specific Metrics / Information Request (Optional)")
metrics_request = st.text_area(
    "Enter specific metrics or information to include in the response",
    help="For example: 'ROI, cost per KPI, milestone achievements, etc.'"
)

# 5. User Question
st.subheader("Ask A.I")
question = st.text_input(
    "Type your query",
    help="Enter your question here. (Max 1000 tokens)"
)
question_tokens = count_tokens(question)
st.text(f"Token Count: {question_tokens} / {TOKEN_LIMIT}")
if question_tokens > TOKEN_LIMIT:
    st.error("Your question exceeds the 1000-token limit. Please shorten it.")

# 6. FAQ Dropdown 
faq_choice = st.selectbox(
    "FAQ",
    ["(None)"] + faq_options,
    help="Select a FAQ to auto-fill the question box if you like."
)

if faq_choice != "(None)":
    # Overwrite the question input with the selected FAQ
    question = faq_choice

# 7. Submit Button
submit_clicked = st.button("Submit")

# ------------- QUERY & RESPONSE LOGIC -------------
def extract_keywords(question_text):
    keywords_words = {"domain", "project", "programme", "budget"}
    words = ''.join(c if c.isalnum() else ' ' for c in question_text.lower()).split()
    keywords = [word for word in words if word not in keywords_words and len(word) > 2]
    return keywords

def format_result_with_id(record, database_label, query_type):
    result = {
        "text": record["text"],
        "labels": record["labels"],
        "source": "primary_db",
        "database": database_label,
        "query_type": query_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    for field in ["id", "title", "url", "document_source"]:
        if field in record and record[field] is not None:
            result[field] = record[field]
    if "id" not in result:
        import hashlib
        text_hash = hashlib.md5(result["text"][:100].encode()).hexdigest()[:10]
        result["id"] = f"{database_label.lower().replace(' ', '_')}_{text_hash}"
    return result

def query_database(driver, user_question):
    keywords = extract_keywords(user_question)
    results = []

    if not driver:
        return {"results": [], "error": "No Neo4j driver available."}

    try:
        with driver.session() as session:
            # 1) Exact text match
            query1 = """
                MATCH (n)
                WHERE n.text IS NOT NULL AND n.text CONTAINS $question
                RETURN n.text AS text, n.id AS id, n.title AS title, labels(n) AS labels,
                       n.url AS url, n.source AS document_source
                LIMIT $limit
            """
            query1_params = {"question": user_question, "limit": RESULTS_LIMIT}
            result1 = session.run(query1, query1_params)
            query1_results = [format_result_with_id(record, "Primary Database", "exact_match") for record in result1]
            results.extend(query1_results)

            # 2) Keyword search if few results
            if len(query1_results) < 5 and keywords:
                keyword_clauses = " OR ".join([f"n.text CONTAINS '{kw}'" for kw in keywords[:15]])
                query2 = f"""
                    MATCH (n)
                    WHERE n.text IS NOT NULL AND ({keyword_clauses})
                    RETURN n.text AS text, n.id AS id, n.title AS title, labels(n) AS labels,
                           n.url AS url, n.source AS document_source
                    LIMIT $limit
                """
                query2_params = {"limit": RESULTS_LIMIT}
                result2 = session.run(query2, query2_params)
                query2_results = [
                    format_result_with_id(record, "Primary Database", "keyword_match")
                    for record in result2
                    if not any(r.get("id") == record["id"] for r in query1_results)
                ]
                results.extend(query2_results)

            # 3) Generic property search if still few results
            if len(results) < 5:
                query3 = """
                    MATCH (n)
                    WHERE (n.content IS NOT NULL OR n.body IS NOT NULL OR n.description IS NOT NULL)
                    RETURN COALESCE(n.content, n.body, n.description) AS text,
                           n.id AS id, n.title AS title, labels(n) AS labels,
                           n.url AS url, n.source AS document_source
                    LIMIT $limit
                """
                query3_params = {"limit": RESULTS_LIMIT}
                result3 = session.run(query3, query3_params)
                query3_results = [
                    format_result_with_id(record, "Primary Database", "property_match")
                    for record in result3
                    if not any(r.get("id") == record["id"] for r in results)
                ]
                results.extend(query3_results)

        return {"results": results}

    except Exception as e:
        return {"results": [], "error": str(e)}

def generate_llm_response(context, user_question, model,
                          system_prompt, response_structure, custom_struct, metrics, params):
    # Determine final structure text
    if response_structure == "Custom":
        structure_text = custom_struct
    else:
        structure_text = response_structure_templates.get(response_structure, "")

    prompt = (
        f"{system_prompt}\n\n"
        f"Context:\n{context}\n\n"
        f"Response Structure:\n{structure_text}\n"
    )
    if metrics:
        prompt += f"(Include these specific metrics/information: {metrics})\n"
    prompt += f"\nQuestion:\n{user_question}\n\nAnalysis:"

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "temperature": params["temperature"],
                "max_tokens": params["max_tokens"],
                "top_k": params["top_k"],
                "stream": False
            },
            stream=False
        )
        if response.status_code == 200:
            result = response.json()
            llm_response = result.get("response", "No response generated")
            return llm_response
        else:
            return f"Error: Unable to generate response (status code {response.status_code})"
    except Exception as e:
        return f"Error: {str(e)}"

if submit_clicked:
    if not selected_models or len(selected_models) != 2:
        st.error("Please select exactly two models.")
    elif system_prompt_tokens > TOKEN_LIMIT or question_tokens > TOKEN_LIMIT:
        st.error("Token limit exceeded. Please shorten your prompt or question.")
    else:
        # Query the DB for context
        with st.spinner("Querying database and generating responses..."):
            context_results = query_database(neo4j_driver, question)
            if not context_results["results"]:
                st.warning("No relevant context found in the database.")
                responses = []
            else:
                # Format context
                formatted_context = "\n".join([
                    f"Document from {item['database']} (ID: {item['id']})"
                    + (f" - Title: '{item['title']}'" if 'title' in item and item['title'] else "")
                    + f" - Labels: {', '.join(item['labels'])}: {item['text']}"
                    for item in context_results["results"]
                ])

                responses = []
                for model in selected_models:
                    st.info(f"Generating response with {model} ...")
                    resp = generate_llm_response(
                        formatted_context,
                        question,
                        model,
                        system_prompt_text,
                        selected_structure,
                        custom_structure,
                        metrics_request,
                        params[model]
                    )
                    responses.append((model, resp))

        # Display the two responses side by side
        if responses:
            col1, col2 = st.columns(2)
            for col, (model, resp) in zip([col1, col2], responses):
                with col:
                    st.subheader(f"Response from {model}")
                    if "deepseek" in model.lower():
                        # Split the response into parts and hide <think> part
                        parts = resp.split("<think>")
                        main_response = parts[0].strip()
                        think_part = parts[1].strip() if len(parts) > 1 else ""

                        st.markdown(f'<div class="model-response">{main_response}</div>', unsafe_allow_html=True)
                        if think_part:
                            with st.expander("View <think> part"):
                                st.write(think_part)
                    else:
                        st.markdown(f'<div class="model-response">{resp}</div>', unsafe_allow_html=True)
