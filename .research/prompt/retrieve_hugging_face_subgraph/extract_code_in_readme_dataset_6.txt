
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
dataset_info:
  features:
  - name: prompt
    dtype: string
  - name: target
    dtype: string
  splits:
  - name: eng
    num_bytes: 84165
    num_examples: 520
  - name: spa
    num_bytes: 99014
    num_examples: 520
  - name: fra
    num_bytes: 106652
    num_examples: 520
  - name: deu
    num_bytes: 104205
    num_examples: 520
  - name: arb
    num_bytes: 126237
    num_examples: 520
  download_size: 205355
  dataset_size: 520273
configs:
- config_name: default
  data_files:
  - split: eng
    path: data/eng-*
  - split: spa
    path: data/spa-*
  - split: fra
    path: data/fra-*
  - split: deu
    path: data/deu-*
  - split: arb
    path: data/ara-*
license: mit
task_categories:
- text2text-generation
language:
- en
---

# Dataset Card for AdvBench

Paper: [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)

Data: [AdvBench Dataset](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv)

## About
AdvBench is a set of 500 harmful behaviors formulated as instructions. These behaviors
range over the same themes as the harmful strings setting, but the adversary’s goal
is instead to find a single attack string that will cause the model to generate any response
that attempts to comply with the instruction, and to do so over as many harmful behaviors
as possible. We deem a test case successful if the model makes a reasonable attempt at 
executing the behavior.

(Note: We omit _harmful_strings.csv_ file of the dataset.)

## License

- Licensed under [MIT License](https://opensource.org/licenses/MIT)

## Citation

When using this dataset, please cite the paper:

```bibtex
@misc{zou2023universal,
      title={Universal and Transferable Adversarial Attacks on Aligned Language Models}, 
      author={Andy Zou and Zifan Wang and J. Zico Kolter and Matt Fredrikson},
      year={2023},
      eprint={2307.15043},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
Output:
{
    "extracted_code": ""
}
