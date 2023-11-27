## Draphnet: drug and phenotype network
The Jupyter notebooks here enable readers to reproduce parts or all of the analysis.

The notebooks are intended to be read in order, but users interested in only parts of the analysis should be able to pick up from later notebooks:

- `1. curate_data.ipynb`: this notebook organizes the data from the three data sources so that they match and can be put into the same model. 
- `2. run_models.ipynb`: this notebook uses the data from the previous one to run the models, creating the estimates of $W$
- `3. Results part 1--evaluation.ipynb`: this notebook runs the assessments of the signal in the data and the initial model evaluation (figure 1A, 2A)
- `4. Fig 2B,C drug target phenome.ipynb`: this notebook assesses whether the signal aligns with drug target data
- `5. Fig 3 drug genome.ipynb`: this notebook creates figure 3 and the assessment of whether drugs predicted to impact the same disease gene also share known targets
- `6. categorize_drugs.ipynb`: this analysis groups drugs by shared predicted disease genes and creates figure 4.