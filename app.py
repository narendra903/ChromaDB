import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import os
import time
import asyncio
from aiohttp import ClientSession  # For async HTTP requests
from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.vectordb.lancedb import LanceDb
from agno.embedder.google import GeminiEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.knowledge.combined import CombinedKnowledgeBase
from agno.document.chunking.document import DocumentChunking
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.models.google import Gemini
from textwrap import dedent
from agno.vectordb.search import SearchType
from agno.document.chunking.agentic import AgenticChunking
import plotly.express as px
import re

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("⚠️ API Key not found. Please set GEMINI_API_KEY in .env.")
    st.stop()


embedder = GeminiEmbedder(id="models/text-embedding-004", dimensions=768,api_key=api_key)
chunking_strategy = AgenticChunking(embedder=embedder)

# Custom CSS for gradient background
page_bg_css = """
<style>
    html, body, [class*="css"]  {
        min-height: 100vh;
        background: linear-gradient(to right, #f7f9fc, #ddeeff);
        color: white;
    }
    .stApp {
        background: transparent;
        min-height: 100vh;
    }
    .stTextInput>div>div>input, 
    .stTextArea>div>div>textarea, 
    .stButton>button {
        background-color: rgba(255, 255, 255, 0.2) !important;
        color: #333 !important;
        border: 1px solid #aaccee !important;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #aaccee !important;
        color: #333 !important;
        border-radius: 12px;
        font-size: 16px;
        padding: 10px 20px;
        transition: all 0.3s ease-in-out;
        border: none;
    }
    h1, h2, h3 {
        color: #444 !important;
    }
    .st-emotion-cache-6qob1r {
        background: rgba(255, 255, 255, 0.5) !important;
    }
</style>
"""

# Streamlit App Setup
st.set_page_config(page_title="💰 Budget AI Assistant", layout="centered")
st.markdown(page_bg_css, unsafe_allow_html=True)

st.title("💡 Indian Budget Analysis AI")
st.markdown("🚀 Ask me anything about the **Indian Union Budget 2025-26**")

# Async function to fetch URLs
async def fetch_url(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return url, await response.read()
            else:
                st.warning(f"⚠️ Failed to fetch {url}: Status {response.status}")
                return url, None
    except Exception as e:
        st.warning(f"⚠️ Error fetching {url}: {str(e)}")
        return url, None

# Function to initialize knowledge bases asynchronously
@st.cache_resource(ttl=86400)  # Cache for 24 hours
async def initialize_knowledge_bases():
    progress_bar = st.progress(0, text="Initializing Vector DB...")
    status_text = st.empty()

    status_text.text("🔄 Initializing Vector DB...")
    await asyncio.sleep(1)  # Simulate DB setup (replace with actual async init if available)
    vector_db = LanceDb(
        table_name="budget",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=embedder,
    )
    progress_bar.progress(30, text="📄 Loading Budget Local PDF Documents...")
    await asyncio.sleep(1)
    status_text.text("📄 Loading Budget Local PDF Documents...")
    # Create CSV knowledge base
    await asyncio.sleep(1)
    pdf_folder = Path(".")
    pdf_files = [  # Corrected to PDF but should be CSV
        pdf_folder / "Union_Budget_FY25-26.pdf"           # Corrected to PDF but should be CSV
    ]
    combined_pdf_kb = []  #create a empty list to add the knowledge bases
    for pdf_file in pdf_files:
         pdf_kb = PDFKnowledgeBase(
            path = pdf_file, # changed the path to be a variable for each csv file
            vector_db= LanceDb(
                table_name=f"pdf_{pdf_file.stem}",
                uri="tmp/lancedb",
                search_type=SearchType.vector,
                embedder=embedder,
            ),
            name=f"Indian Budget PDF - {pdf_file.name}",
            instructions=[
                "Prioritize checking the CSV for answers.",
                "Chunk the CSV in a way that preserves context.",
                "Ensure important sections like summaries and conclusions remain intact.",
                "Maintain the integrity of the logical sections if needed.",
                "Each chunk should provide enough information to answer questions independently.",
                "Create self-contained information units that can provide a full answer to a query.",
            ]
        )
         combined_pdf_kb.append(pdf_kb) 


    status_text.text("📄 Loading Budget PDF Documents...")
    pdf_urls = [
        "https://www.indiabudget.gov.in/doc/rec/allrec.pdf",
        "https://prsindia.org/files/budget/budget_parliament/2025/Union_Budget_Analysis_2025-26.pdf",
        "https://www.ey.com/content/dam/ey-unified-site/ey-com/en-in/technical/alerts-hub/documents/2025/ey-union-budget-2025-alert-infra-sector.pdf",
        "https://www.indiabudget.gov.in/doc/bh1.pdf",
        "https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/feb/doc202524496501.pdf",
        "https://www.indiabudget.gov.in/doc/AFS/allafs.pdf",
        "https://www.indiabudget.gov.in/doc/eb/alldg.pdf",
        "https://www.indiabudget.gov.in/doc/eb/allsbe.pdf",
        "https://www.indiabudget.gov.in/doc/Finance_Bill.pdf",
        "https://www.indiabudget.gov.in/doc/Budget_Speech.pdf",
        "https://www.indiabudget.gov.in/doc/OutcomeBudgetE2025_2026.pdf",
        "https://www.indiabudget.gov.in/doc/memo.pdf",
        "https://www.indiabudget.gov.in/doc/eb/vol1.pdf",
        "https://www.indiabudget.gov.in/doc/frbm1.pdf",
        "https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/feb/doc202521492801.pdf",
        "https://www.indiabudget.gov.in/budget2024-25/doc/Key_to_Budget_Document_2024.pdf",
    ]

    async with ClientSession() as session:
        tasks = [fetch_url(session, url) for url in pdf_urls]
        results = await asyncio.gather(*tasks)
        valid_urls = [url for url, content in results if content is not None]

    pdf_knowledge_base = PDFUrlKnowledgeBase(
        urls=valid_urls,
        vector_db=vector_db,
        name="Indian Budget Records",
        instructions=[
            "For user questions first check the pdf_knowledge_base.",
            "Divide the document into chunks that maintain context around key concepts.",
            "Ensure important sections like summaries and conclusions remain intact.",
            "Each chunk should provide enough information to answer questions independently."
        ]
    )
    progress_bar.progress(60,text="📄 Loading Budget PDF Documents...")

    status_text.text("🌍 Fetching Budget Website Data...")
    website_urls = [
        "https://www.india.gov.in/spotlight/union-budget-2025-2026",
        "https://www.india.gov.in/spotlight/union-budget-2024-25",
        "https://idronline.org/article/advocacy-government/budget-2025-understanding-social-sector-spending/?gad_source=1&gclid=CjwKCAiAlPu9BhAjEiwA5NDSA8hXbzwy3kj1HhhuaRlFZx4kgbgJsgDrPNIbigkD0WJQaocfzFZSwRoCnkYQAvD_BwE",
        "https://frontline.thehindu.com/news/india-budget-2025-key-announcements-tax-relief-agriculture-healthcare-reforms/article69167699.ece",
        "https://www.moneycontrol.com/budget/budget-2025-speech-highlights-key-announcements-of-nirmala-sitharaman-in-union-budget-of-india-article-12926372.html",
        "https://www.bajajfinserv.in/investments/income-tax-slabs"
    ]

    async with ClientSession() as session:
        tasks = [fetch_url(session, url) for url in website_urls]
        results = await asyncio.gather(*tasks)
        valid_website_urls = [url for url, content in results if content is not None]

    website_knowledge_base = WebsiteKnowledgeBase(
        urls=valid_website_urls,
        vector_db=LanceDb(
            table_name="website_documents",
            uri="tmp/lancedb",
            search_type=SearchType.vector,
            embedder=embedder,
        ),
        max_links=10,
        name="Indian Budget Website",
        instructions=[
            "Focus on extracting information that directly answers user questions about the Indian Union Budget.",
            "Prioritize sections like headlines, key findings, summaries, announcements, and data tables.",
            "Identify and extract specific budget-related data, such as allocations, policy changes, tax reforms, or economic forecasts.",
            "Ignore content that is not directly related to the Indian Union Budget, such as advertisements, site navigation, or unrelated news.",
            "Ensure each extracted piece of information retains its original context and meaning, allowing it to be understood independently.",
            "When extracting data, include any associated explanatory text or context that explains what the data means and where it comes from.",
            "Maintain the logical flow and coherence of extracted content, avoiding fragmented or disconnected sentences.",
            "If a section contains multiple related data points or insights, keep them together as a single coherent unit.",
            "Extract exact information related to the user query."
        ]
    )
    progress_bar.progress(80,text="🌍 Fetching Budget Website Data...")

    status_text.text("🔍 Combining Knowledge Bases...")
    combined_knowledge_base = CombinedKnowledgeBase(
        sources=[pdf_knowledge_base, website_knowledge_base] + combined_pdf_kb,
        vector_db=LanceDb(
            table_name="combined_documents",
            uri="tmp/lancedb",
            search_type=SearchType.vector,
            embedder=embedder,
        ),
    )

    await asyncio.sleep(1)  # Simulate async loading (replace with actual async load if available)
    combined_knowledge_base.load(recreate=False)

    progress_bar.progress(100)
    status_text.text("✅ Knowledge Base Loaded Successfully!")
    return combined_knowledge_base

# Load knowledge base in session state
if 'combined_knowledge_base' not in st.session_state:
    # Run async function in Streamlit's event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    st.session_state.combined_knowledge_base = loop.run_until_complete(initialize_knowledge_bases())

# Initialize Agents
knowledge_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp", api_key=api_key),
    knowledge=st.session_state.combined_knowledge_base,
    search_knowledge=True,
    description="📖 Expert on Indian Budget Documents & Websites",
    instructions=[
        "When answering user questions, first delegating the query to  knowledge base for accurate information.",
        "If the answer is not found in the knowledge base, automatically use DuckDuckGoTools for further web research.",
        "Present your response in a formal manner with headings like 'Overview', 'Details', 'Conclusion', Visualization,etc.",
        "For complex queries, break them down into simpler parts if necessary.",
        "Ensure responses are accurate and reference the document or website explicitly where possible.",
        "Use markdown for formatting responses, including bullet points and tables where appropriate.",
        "If the query seems ambiguous, ask for clarification from the user."
    ]
)

searcher = Agent(
    name="Searcher",
    model=Gemini(id="gemini-2.0-flash-exp", api_key=api_key),
    role="🔎 Web Searcher for Budget Analysis",
    description="Specialist in retrieving and analyzing Indian Budget information.",
    instructions=[
        "Activate only when explicitly delegated a query by the 'budget_agent' after the 'knowledge_agent' fails to provide a sufficient answer from the knowledge base.",
        "First, check if the user answer can be found in the existing knowledge_agent or knowledge base.",
        "Reputable financial news outlets (e.g., Moneycontrol, Economic Times)",
        "Economic think tanks and analyses (e.g., NITI Aayog, IMF).",
        "If the information is not available in the knowledge_agent or knowledge base, automatically initiate a web search using DuckDuckGoTools.",
        "Prioritize Indian financial news, government websites, and international news discussing India's budget.",
        "Search specifically for documents or articles related to the Indian Union Budget, focusing on official sources from the government, reputable financial news, and analysis platforms.",
        "When searching, use keywords like 'Indian Budget Analysis', 'Union Budget India', 'Budget 2025-2026 India', along with any specific terms from the query to refine the search.",
        "Compile the results, summarizing key points in your response.",
        "Focus on time-sensitive information related to budget announcements and their immediate global impact.",
        "Maintain objectivity with a focus on data from economic think tanks and financial analysts.",
        "Ensure that the response includes an 'Overview', 'Details' section for in-depth information, and a 'Conclusion' or 'Summary'.",
        "Use markdown for formatting responses, incorporating bullet points for clarity and tables where data comparison is needed.",
        "If the query is ambiguous or requires further clarification, ask for more details from the user.",
        "Keep responses formal and precise, always citing or referencing the source of information when possible.",
        "Use bullet points for clarity."
        "**📈 Data Visualization** (if applicable)"
        "**Table**: For numerical comparisons, e.g."
    ],
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    search_knowledge=True,
)

budget_agent = Agent(
    name="Budget Analysis Agent",
    model=Gemini(id="gemini-2.0-flash-exp", api_key=api_key),
    team=[knowledge_agent, searcher],
    description=dedent("""\
        🤖 An expert analyst team dedicated to analyzing and providing insights on the Indian Budget. 
        This agent leverages pre-existing knowledge from official documents from the knowledge base or current web-based information to deliver comprehensive budget analysis. 
        It works in tandem with a knowledge agent for document-based queries and a searcher agent for the latest updates and analyses from the web.
        If no data found in the Knowledge agent Automatically run the searcher agent.
    """),
    instructions=dedent("""\
        - Start by delegating the user’s query to the 'knowledge_agent' to search the existing knowledge base, which contains official budget documents and trusted website data.
        - Wait for the 'knowledge_agent' to provide a response before taking any further action.
        - Begin by delegating the query to the 'knowledge_agent' to check for any relevant information in the existing knowledge bases.
        - Do not run the 'searcher' agent unless the 'knowledge_agent' explicitly fails to provide a sufficient answer, as defined above.
        - If the 'knowledge_agent' does not respond or no answer, automatically delegate the task to the 'searcher' agent to perform a web search.
        - For both agents, ensure searches are tailored with keywords like 'Indian Budget', 'Union Budget India', 'Budget Analysis', and any query-specific terms.
        - Key viewpoints from budget speeches, financial reviews, and economic think tanks.
        - If the response from knowledge_agent is empty or inadequate, **run the searcher**.
        - Stakeholder reactions: Include responses from industry bodies/opposition.
        - Format the response with clear headings:
          - **📌 Overview**: A brief summary of the budget point in question.
          - **📊 Details**: In-depth analysis, including any numerical data, policy implications, or sector-specific impacts.
          - **✅ Conclusion**: Summarize key takeaways, expected outcomes, or areas for further research.
          -- **Numerical Data**: Tables or figures for budgetary allocations and expenditures.
          -- **Sources**: Cite documents or URLs where applicable.
        - Comparisons with previous budgets for trend analysis.
        - Use markdown for formatting outputs, including bullet points, tables, or code blocks for clarity.
        - If the query lacks clarity, prompt the user for additional details or clarification.
        - Maintain a formal and professional tone in responses, always citing sources where applicable.
        "- Use bullet points for clarity."
        
    """),
    expected_output=dedent("""\
        # {Compelling Headline}                   
        ### Overview
        - Here's a brief summary of the budget analysis for the query.
        ### Details
        - Detailed breakdown including numbers, policies, and sector analysis.
        ## Expert Insights
        - Quotes from economists and market analysts
        ## Numerical Breakdown
        - Create comparative tables for allocations:
            | Sector | 2024-25 | 2025-26 | Change (%) |
            |--------|---------|---------|------------|
            | Health | 89,000cr| 1,05,000cr | +18%     |
        ## Geographic Distribution
        - State-wise fund allocation patterns
        - Special focus regions/aspirational districts
        Always include:
        - Reference to specific document/page numbers
        - Source URLs for web-sourced information
        - Last updated timestamps for time-sensitive data                           
        ## Sources & Methodology
        - Description of research process and sources
        ## For technical queries:
        - Create flowchart for complex processes
        - Use code blocks for formula explanations
        - Add footnotes for legal citations        
                                                                                                    
        ### Conclusion
        - Key insights and implications from the budget analysis.
        - Suggested compliance strategies.
                           
        ### Visualization
        - "**📈 Data Visualization** (if applicable):"
        - "**Table**: For numerical comparisons, "
        - Include tables/pie charts when data is sufficient (3+ points) and relevant.
        ---
        Research conducted by Financial Agent
        Credit Rating Style Report
        Published: {current_date}
        Last Updated: {current_time}                                                    
    """),
    add_datetime_to_instructions=True,
    markdown=True,
    show_tool_calls=True,
)

# Streamlit UI for User Input
query = st.text_input("🔍 Enter your budget-related query:", placeholder="E.g., What are the major tax changes in Budget 2025?")

# Custom Button CSS
button_css = """
    <style>
        .stButton > button {
            background-color: #008CBA;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background-color: #005F73;
            color: #FFD700;
        }
    </style>
"""
st.markdown(button_css, unsafe_allow_html=True)

# Button to Generate Response
if st.button("🚀 Generate Response"):
    if query:
        with st.spinner("📊 Analyzing budget data... Please wait."):
            st.session_state.response_container = st.empty() # added a session state
            with st.session_state.response_container:
                try:
                    run_response = budget_agent.run(query, markdown=True)
                    st.markdown(run_response.content, unsafe_allow_html=True)
                    # Check for pie chart data in response
                    pie_chart_match = re.search(r"Pie Chart:.*?(?=###|\n\n|$)", run_response.content, re.DOTALL)
                    if pie_chart_match:
                        pie_text = pie_chart_match.group()
                        labels = []
                        values = []
                        for line in pie_text.split("\n")[1:]:
                            if line.strip().startswith("-"):
                                label, value = line.split(":")
                                labels.append(label.strip("- ").strip())
                                values.append(float(value.strip().replace("%", "")))
                        fig = px.pie(values=values, names=labels, title="Budget Allocation")
                        st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"⚠️ An error occurred: {str(e)}. Please try again or contact support.")
    else:
            st.warning("⚠️ Please enter a query before generating a response!")

# Sidebar Info
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
🔹 This AI assistant provides **real-time** insights on the Indian **Union Budget 2025-26**  
🔹 Sources: **Government Budget Documents** 📄 + **Trusted Financial Websites** 🌐  
🔹 Uses **AI-powered Agents** 🤖 for **document-based** & **web-based research**
""")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ How It Works")
st.sidebar.markdown("""
1️⃣ **Knowledge Base Search** 📚  
2️⃣ **Web Search (if needed)** 🌍  
3️⃣ **Formatted AI Response** 📊  
""")

st.sidebar.subheader("💡 Example Queries")
st.sidebar.markdown("""
- What are the major tax changes in Budget 2025?
- How much is allocated to healthcare in 2025-26?
- What are the key highlights of the Budget Speech?
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📞 Contact")
st.sidebar.markdown("💡 Created by: **AI & Finance Enthusiasts**")
st.sidebar.markdown("📩 Email: narendra.insights@gmail.com")

# Footer
st.markdown("---")
st.markdown(
    "🛠️ **Built with AI & Love ❤️** | 📅 *Updated: 2025* | "
    "[<img src='https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg' width='20' height='20'>](https://www.linkedin.com/in/nk-analytics/)"
    " Connect on LinkedIn",
    unsafe_allow_html=True,
)