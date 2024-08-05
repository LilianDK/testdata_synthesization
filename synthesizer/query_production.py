import pandas as pd
from jinja2 import Template
from loguru import logger

from synthesizer.config import co

# Hier eine andere Lösung finden statt aus dem zuvor produzierten Excel wieder laden
contexts = pd.ExcelFile(r"results/first_stage_results.xlsx")
contexts = contexts.parse(0)
nrow = len(contexts)

original_input = []
for i, _row in contexts.iterrows():
    with open("prompt/query_generation.j2") as file:
        template_str = file.read()

    template = Template(template_str)
    text_chunk = {"contexts": contexts.iloc[i, 3]}
    prompt = template.render(text_chunk)

    response = co.chat(
        model="command-r-plus",
        message=prompt,
        max_tokens=300,
        temperature=0.7,
        k=0,
        p=0,
    )
    logger.info(response.text)
    original_input.append(response.text)

df_2 = pd.concat([contexts, pd.DataFrame(original_input)], ignore_index=False, axis=1)
df_2.to_excel("results/second_stage_results.xlsx")
