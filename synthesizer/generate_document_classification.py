import random

import pandas as pd
from jinja2 import Template
from loguru import logger

from synthesizer.config import co

n = 100
text = []
label = []
for i in range(n):
 print(i)
 response = co.chat(
        model="command-r-plus",
        message="Generate a german template report document. Document: ",
        max_tokens=100,
        temperature=0.7,
        k=0,
        p=0,
    )
 print(response.text)
 text.append(response.text)
 label.append("Protokoll")

df = pd.DataFrame({"text": text, "label": label})
df.to_excel("results_protokoll.xlsx")