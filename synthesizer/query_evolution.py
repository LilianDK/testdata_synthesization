import random
import pandas as pd
from jinja2 import Template
from synthesizer.config import co

co = co

with open("prompt/query_evolution_multi_step_reasoning.j2", "r") as file:
    template_str = file.read()
evolution_template_1 = Template(template_str)

with open("prompt/query_evolution_multi_context_template.j2", "r") as file:
    template_str = file.read()
evolution_template_2 = Template(template_str)

with open("prompt/query_evolution_hypothetical_scenario.j2", "r") as file:
    template_str = file.read()
evolution_template_3 = Template(template_str)

with open("prompt/expected_output_generation.j2", "r") as file:
    template_str = file.read()
expectation_output = Template(template_str)

contexts = pd.ExcelFile(r"results/second_stage_results.xlsx")
contexts = contexts.parse(0)
nrow = len(contexts)

evolution_templates = [evolution_template_1, evolution_template_2, evolution_template_3]
num_evolution_steps = 3

processing_context = contexts
# processing_context = contexts[contexts.iloc[:,-2] == i]
context = processing_context

evolved_query = []

for j, row in contexts.iterrows():
    original_input = {"original_input": context.iloc[j, 6]}
    prompt = original_input
    print(prompt)
    text = {"context": context.iloc[j, 4]}

    # Function to perform random evolution steps
    def evolve_query(original_input: str, text: str, num_evolution_steps: int):
        for _ in range(num_evolution_steps):
            # Choose a random (or using custom logic) template from the list
            chosen_template = random.choice(evolution_templates)
            # template = Template(chosen_template)
            print(f"ORIGINAL INPUT: {original_input}")
            prompt = chosen_template.render(text=text, original_input=original_input)
            # Update the current input with the "Rewritten Input" section
            response = co.chat(
                model="command-r-plus",
                message=prompt,
                max_tokens=300,
                temperature=0.7,
                k=0,
                p=0,
            )
            print(f"NEW INPUT: {response.text}")
            original_input = {"original_input": response.text}
        return original_input

    # Evolve the input by randomly selecting the evolution type
    temp = evolve_query(original_input, text, num_evolution_steps)
    print(temp)
    evolved_query.append(temp)

df_3 = pd.concat([contexts, pd.DataFrame(evolved_query)], ignore_index=False, axis=1)
df_3.to_excel(f"results/third_stage_results.xlsx")

expected_query = []
j = 0
contexts = pd.ExcelFile(r"results/third_stage_results.xlsx")
contexts = contexts.parse(0)

for j, row in contexts.iterrows():
    evolved_query = {"original_input": contexts.iloc[j, 7]}
    text = {"context": contexts.iloc[j, 5]}
    print(f"EVOLVED QUERY: {evolved_query}")
    prompt = expectation_output.render(text=text, evolved_query=evolved_query)
    response = co.chat(
        model="command-r-plus",
        message=prompt,
        max_tokens=300,
        temperature=0.7,
        k=0,
        p=0,
    )
    print(f"EXPECTED QUERY: {response.text}")
    expected_query.append(response.text)

df_4 = pd.concat([contexts, pd.DataFrame(expected_query)], ignore_index=False, axis=1)
df_4.to_excel(f"results/fourth_stage_results.xlsx")
