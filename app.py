import streamlit as st
from agents import run_crew

# Streamlit UI setup
st.set_page_config(page_title="SEO Article Generator", page_icon="📝", layout="wide")

st.title("📝 SEO Article Generator with CrewAI and Groq")
st.markdown("""
Welcome to the **SEO Article Generator**! Enter a topic below, and watch as our AI agents collaborate to produce a high-quality, SEO-optimized article.
""")

# Input for the topic
st.sidebar.header("User Input")
topic = st.sidebar.text_input("Enter the topic you want to generate an article about:")

# Button to trigger the Crew execution
if st.sidebar.button("Generate Article"):
    if topic:
        st.markdown("## Process Overview")
        with st.spinner("Generating SEO-optimized content..."):
            try:
                # Run the crew and get the result
                result = run_crew(topic)

                # Extract the process logs
                process_logs = result.get("process_logs", "")
                final_output = result.get("final_output", "No final output available")

                # Parse the process logs to separate outputs from each agent
                agent_sections = process_logs.split("=== Agent Execution ===")
                for section in agent_sections:
                    if section.strip():
                        st.write("---")  # Separator between agents
                        lines = section.strip().splitlines()
                        role_line = next((line for line in lines if "Role:" in line), None)
                        if role_line:
                            role = role_line.split("Role:")[1].strip()
                            st.markdown(f"### **{role}**")
                            with st.expander(f"See what the {role} did"):
                                st.text('\n'.join(lines))
                        else:
                            st.text(section)

                # Display the final generated article
                st.write("---")
                st.markdown("## 📄 Final Generated Article")
                st.success("Your SEO-optimized article is ready!")
                st.write(final_output)

                st.balloons()  # Celebration animation

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a topic to generate an article.")
