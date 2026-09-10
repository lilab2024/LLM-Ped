# Domain-Informed Prompting and Cascaded Retrieval-Augmented

LLM-Ped is a research codebase for predicting whether a driver will yield to pedestrians at unsignalized crosswalks. The repository combines structured interaction records, intersection images, transportation-domain knowledge, large language models, retrieval-augmented generation, and parameter-efficient fine-tuning. 

## Project Overview

The code includes:

- **Prompt for GPT-4o/GPT-4o mini**: including relevant domain knowledge, structured thinking guidance, few-shot prompting, and multimodal prompting with intersection images.

- **Prompt for DeepSeek-V3/DeepSeek-R1**: including relevant domain knowledge, structured thinking guidance, and few-shot prompting.

- **CRAFT**: a two-stage fine-tuning framework that adapts compact open-source language models, specifically **Qwen3-4B** and **Qwen3.5-4B**, to driver-yielding prediction.



# Datasets

This study draws on the open source Minnesota driver pedestrian dataset documented comprising 3,314 interactions recorded at 18 unsignalised intersections. These sites were specifically chosen in collaboration with the Minnesota Department of Transportation to ensure a diverse representation of traffic volumes, land use contexts, and intersection geometries.
