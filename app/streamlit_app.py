import sys
import os
from io import StringIO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from src.agentic import run_agentic_analysis
from src.pipeline import run_pipeline


st.set_page_config(page_title="Meeting Intelligence System", layout="wide")

st.title("Meeting Intelligence System")
st.write("Upload a meeting transcript and analyze action items, decisions, topics, semantic search, and agentic insights.")


uploaded_file = st.file_uploader("Upload transcript (.txt)", type=["txt"])
query = st.text_input("Ask a question about the meeting", value="What was decided about the demo?")
analysis_mode = st.radio("Analysis mode", ["Agentic", "Standard"], horizontal=True)
top_k = st.slider("Search results", min_value=1, max_value=10, value=5 if analysis_mode == "Agentic" else 3)

if uploaded_file is not None:
    transcript = StringIO(uploaded_file.getvalue().decode("utf-8")).read()

    st.subheader("Transcript Preview")
    st.text_area("Transcript Content", transcript, height=200)

    if st.button("Analyze Meeting"):
        with st.spinner("Analyzing meeting..."):
            if analysis_mode == "Agentic":
                output = run_agentic_analysis(transcript, query, top_k=top_k)
            else:
                output = run_pipeline(transcript, query, top_k=top_k)

        st.success("Analysis complete")

        if output.get("agentic"):
            agentic = output["agentic"]

            st.subheader("Agentic Answer")
            st.write(agentic["answer"].get("answer", "No answer generated."))

            plan_tab, decision_tab, risk_tab, question_tab, quality_tab, trace_tab = st.tabs(
                ["Plan", "Decisions", "Risks", "Questions", "Quality", "Trace"]
            )

            with plan_tab:
                st.json(agentic["execution_plan"])

            with decision_tab:
                st.json(agentic["decision_register"])

            with risk_tab:
                st.json(agentic["risks"])

            with question_tab:
                st.json(agentic["follow_up_questions"])

            with quality_tab:
                st.json(agentic["quality"])

            with trace_tab:
                st.json(agentic["agent_trace"])

        st.subheader("Summary")
        st.json(output["summary"])

        st.subheader("Action Items")
        if output["action_items"]:
            st.json(output["action_items"])
        else:
            st.write("No action items found.")

        st.subheader("Decisions")
        if output["decisions"]:
            st.json(output["decisions"])
        else:
            st.write("No decisions found.")

        st.subheader("Topics")
        if output["topics"]:
            st.json(output["topics"])
        else:
            st.write("No topics found.")

        st.subheader("Search Results")
        if output["search_results"]:
            st.json(output["search_results"])
        else:
            st.write("No search results found.")

        st.subheader("Full Output JSON")
        st.json(output)
