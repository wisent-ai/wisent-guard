# Wisent-Guard Benchmarks

This directory contains research papers and documentation related to the benchmarks and datasets used in the Wisent-Guard project.

## Contents

### Documents
- **neurips_2024.pdf** - NeurIPS 2024 submission containing the main research findings
- **neurips_2024.tex** - LaTeX source file for the NeurIPS paper
- **bibliography.bib** - Bibliography and references used in the research

## neurips_2024.pdf

In appendix A (work in progress), there is a big table which contains benchmarks supported by wisent, their evaluation methods, contrastive pairs extraction methods and few words summarization of the topic of benchmark. Benchmarks are divided into smaller categories such as reading comprehension, multi choice reasoning, exams and knowledge tests, mathematics, coding and other. Those are indicated by colors. Evaluation and extraction methods are described.

## benchmarks in practise

python -m wisent_guard tasks --list-tasks

📋 VALID TASKS (79 available out of 131 total)

The 79 available tasks:
aime,aime2024,aime2025,apps,arc_challenge,arc_easy,arithmetic,asdiv,boolq,codexglue_code_to_text_go,codexglue_code_to_text_java,codexglue_code_to_text_javascript,codexglue_code_to_text_php,codexglue_code_to_text_python,codexglue_code_to_text_ruby,conala,concode,copa,coqa,drop,ds1000,gpqa,gpqa_diamond,gpqa_diamond_cot_zeroshot,gpqa_diamond_zeroshot,gpqa_extended,gpqa_extended_cot_zeroshot,gpqa_extended_zeroshot,gpqa_main_cot_zeroshot,gpqa_main_zeroshot,gsm8k,hellaswag,hendrycks_math,hle,hle_exact_match,hle_multiple_choice,hmmt,hmmt_feb_2025,humaneval,humaneval_plus,instructhumaneval,livemathbench,livemathbench_cnmo_en,livemathbench_cnmo_zh,math,math500,mbpp,mbpp_plus,mercury,mmmlu,multiple_cpp,multiple_go,multiple_java,multiple_js,multiple_py,multiple_rs,naturalqs,openbookqa,piqa,polymath,polymath_en_high,polymath_en_medium,polymath_zh_high,polymath_zh_medium,race,recode,record,squad2,supergpqa,supergpqa_biology,supergpqa_chemistry,supergpqa_physics,swag,triviaqa,truthfulqa_mc1,truthfulqa_mc2,webqs,wikitext,winogrande

Notice that there is smaller number of available tasks than benchmarks in the table, this is because some of them are not fully implemented yet.

Notice that even if task works it does not mean that it works as intended or reasonable.

## updates

I follow the table from neurips_2024.pdf and systematically and methodically check benchmarks and fix them if broken or add new one.

23 september: there are 95 tasks available now, so 16 were added and about the same number were fixed.
24 september: 108 available



