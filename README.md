# Nerfstudio workspace

## Projects list

- [ ] Reproduce nerfacto results from siggraph paper
- [ ] Implement mipnerf360

## Projects
### Reproduce benchmarks from siggraph paper
Benchmarks to to reproduce:
* nerfacto on Mip-NeRF 360 Dataset
* nerfacto on nerfstudio dataset

Original benchmark script:
https://github.com/nerfstudio-project/nerfstudio/blob/SIGGRAPH-2023-Code/projects/nerfstudio_paper/benchmark_nerfstudio_paper.py

Remarks: This branch adds `eval_optimize_cameras` and `use-appearance-embedding`, which never made it to the main branch. However, for the mipsnerf360 data these values are set to `False` which should turn off any new features and match the main branch performance.

Mismatch between benchmark script and paper:  script: ` --max-num-iterations 30001` paper: "train for 70k iterations"

But also results after 5k iterations are off and even the final results of 30k iterations does not match the results for 5k as reported in the paper.