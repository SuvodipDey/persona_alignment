# Improving Persona-Conditioned Behavioral Fidelity in Language Models

## Install dependencies
Create a Python 3.13 (or later) environment and install the dependencies.
```console
❱❱❱ pip install -r requirements.txt
```

## OPENROUTER API keys
Set your OpenRouter API key in the .env file.

## DATA

1. Create three directories - "opinion_qa", "website_likability", and "morale_machine".

2. Download the [OpinionsQA](https://worksheets.codalab.org/worksheets/0x6fb693719477478aac73fc07db333f69) dataset and put it in the "opinion_qa/data" directory. After this step, the "opinion_qa/data" directory will contain three folders - human_resp, model_input, runs.

3. Download or git clone the [Website Aesthetics](https://github.com/calista-ai/website-aesthetics-datasets) dataset in the "website_likability" directory.

4. Download the [Moral Machine](https://osf.io/3hvt2/files/osfstorage?view_only=4bb49492edee4a8eb1758552a362a2cf) dataset in the "morale_machine" directory. After this step, the "morale_machine" directory should contain a directory named "Datasets".

## Create dataset for the experiments

Create train and test sets from OpinionQA, Website Aesthetics, and Moral Machine by running the following scripts.

```console
python create_data_opinion_qa.py
```

```console
python create_data_website_aes.py
```

```console
python create_data_morale_machine.py
```

After running the scripts, files named train_dataset.csv and test_dataset.csv will be created in the "opinion_qa", "website_likability", and "morale_machine" directories.

## Run Baselines

Run the baselines for the three datasets by running the following scripts.

```console
python run_baseline_opinion_qa.py
```

```console
python run_baseline_website_aes.py
```

```console
python run_baseline_morale_machine.py
```

The baselines will generate the output for the corresponding test_dataset.csv files. After running the scripts, the generated output will be saved in "output_baseline.csv" in the respective directories.

## Run Model with In-Context Examples

Run the RAG-augmented model for the three datasets by running the following scripts.

```console
python model_opinion_qa.py
```

```console
python model_website_aes.py
```

```console
python model_morale_machine.py
```

The scripts will generate the output for the corresponding test_dataset.csv files. After running the scripts, the generated output will be saved in "output_model.csv" in the respective directories.

## Evaluation

```console
python evaluate_opinion_qa.py
```

```console
python evaluate_website_aes.py
```

```console
python evaluate_morale_machine.py
```

The result file will be generated in the respective directories.

The repository includes all datasets, baseline and model output files, and evaluation results referenced in the report.
