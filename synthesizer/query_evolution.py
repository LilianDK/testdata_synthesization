import random

import pandas as pd
from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from loguru import logger
from openinference.instrumentation.langchain import LangChainInstrumentor

from synthesizer.config import co, generation_model, llm_model

LangChainInstrumentor().instrument()


# Function to perform random evolution steps
def evolve_query(
    original_input: str, text: str, num_evolution_steps: int, evolution_templates: list
) -> dict:
    for _ in range(num_evolution_steps):
        # Choose a random (or using custom logic) template from the list
        chosen_template = random.choice(evolution_templates)

        prompt = PromptTemplate.from_template(
            template=chosen_template, template_format="jinja2"
        )

        if llm_model == "cohere":
            chat = ChatCohere(
                client=co,
                model="command-r-plus",
                max_tokens=300,
                temperature=0.7,
                k=0,
                p=0,
            )
            chain = prompt | chat
        if llm_model == "ollama":
            llm = ChatOllama(
                model=generation_model,
                temperature=0.7,
                # other params...
            )
            chain = prompt | llm
        # Update the current input with the "Rewritten Input" section
        response = chain.invoke({"original_input": original_input, "context": text})
        original_input = {"original_input": [response.content.replace("\n", " ")]}

    return original_input


def query_evolution() -> None:
    logger.info("START QUERY EVOLUTION")
    with open("prompt/query_evolution_multi_step_reasoning.j2") as file:
        evolution_template_1 = file.read()

    with open("prompt/query_evolution_multi_context_template.j2") as file:
        evolution_template_2 = file.read()

    with open("prompt/query_evolution_hypothetical_scenario.j2") as file:
        evolution_template_3 = file.read()

    with open("prompt/expected_output_generation.j2") as file:
        expectation_output = file.read()

    with open("prompt/query_formatting.j2") as file:
        formatting_output = file.read()

    expectation_output_prompt = PromptTemplate.from_template(
        template=expectation_output, template_format="jinja2"
    )

    formatting_output_prompt = PromptTemplate.from_template(
        template=formatting_output, template_format="jinja2"
    )

    if llm_model == "cohere":
        chat = ChatCohere(
            client=co,
            model="command-r-plus",
            max_tokens=300,
            temperature=0.7,
            k=0,
            p=0,
        )
        expectation_output_chain = expectation_output_prompt | chat
        formatting_output_prompt_chain = formatting_output_prompt | chat
    if llm_model == "ollama":
        llm = ChatOllama(
            model=generation_model,
            temperature=0.7,
            # other params...
        )
        expectation_output_chain = expectation_output_prompt | llm
        formatting_output_prompt_chain = formatting_output_prompt | llm

    contexts = pd.read_csv(r"results/second_stage_results.csv", encoding="utf-8")

    evolution_templates = [
        evolution_template_1,
        evolution_template_2,
        evolution_template_3,
    ]
    num_evolution_steps = 3

    processing_context = contexts
    context = processing_context

    evolved_query = []

    for j, _row in contexts.iterrows():
        original_input = context.iloc[j, 5]
        text = context.iloc[j, 4]

        # Evolve the input by randomly selecting the evolution type
        temp = evolve_query(
            original_input,
            text,
            num_evolution_steps,
            evolution_templates=evolution_templates,
        )
        query_raw = temp["original_input"]
        formatted_query = formatting_output_prompt_chain.invoke(
            {"context": query_raw}
        )
        evolved_query.append(formatted_query.content)

    df_3 = pd.concat(
        [contexts, pd.DataFrame(evolved_query)], ignore_index=False, axis=1
    )
    df_3.to_csv("results/third_stage_results.csv")

    expected_query = []
    j = 0
    contexts = pd.read_csv(r"results/third_stage_results.csv", encoding="utf-8")

    for j, _row in contexts.iterrows():
        evolved_query = contexts.iloc[j, 7]
        text = contexts.iloc[j, 5]
        response = expectation_output_chain.invoke(
            {"evolved_query": evolved_query, "context": text}
        )
        expected_query.append([response.content.replace("\n", " ")])

    df_4 = pd.concat(
        [contexts, pd.DataFrame(expected_query)], ignore_index=False, axis=1
    )
    df_4.to_csv("results/fourth_stage_results.csv")

    logger.info("END QUERY EVOLUTION")
