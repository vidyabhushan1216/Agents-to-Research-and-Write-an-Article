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
    backstory=(
        "You are tasked with planning a blog article on {topic}. "
        "Your goal is to research trends, gather insights, and "
        "create a structure that will be handed to the writer."
    ),
    allow_delegation=False,
    verbose=True
)

writer = Agent(
    llm=llm,
    role="Content Writer",
    goal="Write an insightful opinion piece on {topic}",
    backstory=(
        "You are a writer working with the plan created by the Planner. "
        "Your goal is to write a clear, engaging article, "
        "providing factual information and well-reasoned opinions."
    ),
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    llm=llm,
    role="Editor",
    goal="Edit the blog post to align with the organization's style",
    backstory=(
        "You receive the article from the Writer. Your role is to polish it, "
        "ensuring it adheres to journalistic standards, the brand voice, "
        "and is free of errors."
    ),
    allow_delegation=False,
    verbose=True
)

# Define tasks
plan_task = Task(
    description=(
        "Create an outline and key SEO points for {topic}. "
        "This includes audience analysis, introduction, main points, and conclusion."
    ),
    expected_output="A detailed content plan with outline, keywords, and sources.",
    agent=planner
)

write_task = Task(
    description=(
        "Using the content plan, craft a detailed article with a structured flow, "
        "incorporating SEO and well-written sections."
    ),
    expected_output="A draft of the article with clear, engaging content.",
    agent=writer
)

edit_task = Task(
    description=(
        "Edit the draft for grammar, flow, alignment with the brand's voice, "
        "and readiness for publication."
    ),
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

# Example execution
if __name__ == "__main__":
    result = run_crew("The impact of AI on healthcare")
    print(result['process_logs'])
    print("Final Output:")
    print(result['final_output'])
