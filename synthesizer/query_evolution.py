import random

import pandas as pd
from jinja2 import Template
from loguru import logger
from langchain_core.prompts import PromptTemplate
from langchain_cohere import ChatCohere
from synthesizer.config import co
from openinference.instrumentation.langchain import LangChainInstrumentor

LangChainInstrumentor().instrument()

chat = ChatCohere(client=co, model="command-r-plus",             max_tokens=300,
        temperature=0.7,
        k=0,
        p=0,)



# Function to perform random evolution steps
def evolve_query(original_input: str, text: str, num_evolution_steps: int, evolution_templates: list) -> dict:
    for _ in range(num_evolution_steps):
        # Choose a random (or using custom logic) template from the list
        chosen_template = random.choice(evolution_templates)
        # template = Template(chosen_template)
        logger.info(f"ORIGINAL INPUT: {original_input}")

        prompt = PromptTemplate.from_template(template=chosen_template, template_format="jinja2")

        chain = prompt | chat
        # Update the current input with the "Rewritten Input" section
        response = chain.invoke({"original_input": original_input, "context": text})
        logger.info(f"NEW INPUT: {response.content}")
        original_input = {"original_input": response.content}

    return original_input

def query_evolution():
    logger.info("START QUERY EVOLUTION")
    with open("prompt/query_evolution_multi_step_reasoning.j2") as file:
        evolution_template_1 = file.read()

    with open("prompt/query_evolution_multi_context_template.j2") as file:
        evolution_template_2 = file.read()

    with open("prompt/query_evolution_hypothetical_scenario.j2") as file:
        evolution_template_3 = file.read()

    with open("prompt/expected_output_generation.j2") as file:
        expectation_output = file.read()


    expectation_output_prompt = PromptTemplate.from_template(template=expectation_output, template_format="jinja2")

    expectation_output_chain = expectation_output_prompt | chat

    contexts = pd.ExcelFile(r"results/second_stage_results.xlsx")
    contexts = contexts.parse(0)
    nrow = len(contexts)

    evolution_templates = [evolution_template_1, evolution_template_2, evolution_template_3]
    num_evolution_steps = 3

    processing_context = contexts
    # processing_context = contexts[contexts.iloc[:,-2] == i]
    context = processing_context

    evolved_query = []

    for j, _row in contexts.iterrows():
        original_input = context.iloc[j, 6]
        prompt = original_input
        logger.info(prompt)
        text = context.iloc[j, 4]

        # Evolve the input by randomly selecting the evolution type
        temp = evolve_query(original_input, text, num_evolution_steps, evolution_templates=evolution_templates)
        logger.info(temp)
        evolved_query.append(temp)

    df_3 = pd.concat([contexts, pd.DataFrame(evolved_query)], ignore_index=False, axis=1)
    df_3.to_excel("results/third_stage_results.xlsx")

    expected_query = []
    j = 0
    contexts = pd.ExcelFile(r"results/third_stage_results.xlsx")
    contexts = contexts.parse(0)

    for j, _row in contexts.iterrows():
        evolved_query = contexts.iloc[j, 7]
        text = contexts.iloc[j, 5]
        response = expectation_output_chain.invoke({"evolved_query": evolved_query, "context": text})
        logger.info(f"EVOLVED QUERY: {evolved_query}")
        logger.info(f"EXPECTED QUERY: {response.content}")
        expected_query.append(response.content)

    df_4 = pd.concat([contexts, pd.DataFrame(expected_query)], ignore_index=False, axis=1)
    df_4.to_excel("results/fourth_stage_results.xlsx")

    logger.info("END QUERY EVOLUTION")