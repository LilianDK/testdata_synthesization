import pandas as pd
from jinja2 import Template
from loguru import logger
from langchain_core.prompts import PromptTemplate
from langchain_cohere import ChatCohere
from synthesizer.config import co
from openinference.instrumentation.langchain import LangChainInstrumentor

LangChainInstrumentor().instrument()


def query_production():
    logger.info("START QUERY PRODUCTION")
    # Hier eine andere Lösung finden statt aus dem zuvor produzierten Excel wieder laden
    contexts = pd.ExcelFile(r"results/first_stage_results.xlsx")
    contexts = contexts.parse(0)
    nrow = len(contexts)

    chat = ChatCohere(client=co, model="command-r-plus",             max_tokens=300,
            temperature=0.7,
            k=0,
            p=0,)

    with open("prompt/query_generation.j2") as file:
        template_str = file.read()

    prompt = PromptTemplate.from_template(template=template_str, template_format="jinja2")

    chain = prompt | chat

    original_input = []
    for i, _row in contexts.iterrows():

        result = chain.invoke({"contexts": contexts.iloc[i, 3]})

        logger.info(result.content)
        original_input.append(result.content)

    df_2 = pd.concat([contexts, pd.DataFrame(original_input)], ignore_index=False, axis=1)
    df_2.to_excel("results/second_stage_results.xlsx")

    logger.info("END QUERY PRODUCTION")