import pandas as pd
from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from loguru import logger
from openinference.instrumentation.langchain import LangChainInstrumentor

from synthesizer.config import co, generation_model, llm_model

LangChainInstrumentor().instrument()

def query_production() -> None:
    logger.info("START QUERY PRODUCTION")
    # Hier eine andere Lösung finden statt aus dem zuvor produzierten Excel wieder laden
    contexts = pd.read_csv(r"results/first_stage_results.csv", encoding="utf-8")

    with open("prompt/query_generation.j2") as file:
        template_str = file.read()

    prompt = PromptTemplate.from_template(
        template=template_str, template_format="jinja2"
    )
    if llm_model == "cohere":
        chat = ChatCohere(
            client=co,
            model="command-r-plus",
            max_tokens=300,
            temperature=0.7,
            k=0,
            p=0
        )
        chain = prompt | chat
    if llm_model == "ollama":
        llm = ChatOllama(
            model=generation_model,
            temperature=0.7,
            # other params...
        )
        chain = prompt | llm

    original_input = []
    for i, _row in contexts.iterrows():
        result = chain.invoke({"contexts": contexts.iloc[i, 3]})
        original_input.append([result.content.replace("\n", " ")])

    original_input = pd.DataFrame(original_input)
    original_input["initial_query"] = original_input.replace("\\n", " ")
    original_input["initial_query"] = original_input.iloc[:,1].apply(lambda x: [x]) # potentiell dämlichj
    df_2 = pd.concat(
        [contexts, original_input.iloc[:,1]], ignore_index=False, axis=1
    )

    df_2.to_csv("results/second_stage_results.csv", encoding="utf-8")


    logger.info("END QUERY PRODUCTION")
