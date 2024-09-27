import os
import io
import logging
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize the Groq LLM
llm = ChatGroq(
    temperature=0,
    model_name="llama3-70b-8192",
    api_key=api_key
)

# Define agents
planner = Agent(
    llm=llm,
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You're working on planning a blog article about the topic: {topic}.",
    allow_delegation=False,
    verbose=True
)

writer = Agent(
    llm=llm,
    role="Content Writer",
    goal="Write an insightful opinion piece on the topic: {topic}",
    backstory="You write based on the Content Planner's outline and provide objective insights.",
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    llm=llm,
    role="Editor",
    goal="Edit the blog post to align with the organization's style.",
    backstory="You review the content for journalistic standards and voice alignment.",
    allow_delegation=False,
    verbose=True
)

# Define tasks
plan_task = Task(
    description="Create an outline and key SEO points for {topic}.",
    expected_output="A detailed content plan with outline, keywords, and sources.",
    agent=planner
)

write_task = Task(
    description="Write the article based on the outline and content structure.",
    expected_output="A draft of the article with clear, engaging content.",
    agent=writer
)

edit_task = Task(
    description="Edit the draft for grammar, flow, and alignment.",
    expected_output="A polished, publication-ready article.",
    agent=editor
)

# Create the Crew
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan_task, write_task, edit_task],
    verbose=2
)

def run_crew(topic):
    # Prepare inputs
    inputs = {"topic": topic}
    
    # Configure logging to capture logs
    logger = logging.getLogger('crewai')
    logger.setLevel(logging.DEBUG)
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    # Execute the crew
    final_output = crew.kickoff(inputs=inputs)

    # Retrieve logs
    process_logs = log_stream.getvalue()
    logger.removeHandler(handler)
    handler.close()

    # Structure the output
    result = {
        "final_output": final_output if isinstance(final_output, str) else "No output generated",
        "process_logs": process_logs
    }

    return result
