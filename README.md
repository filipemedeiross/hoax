# HOAX

## Application of Machine Learning Techniques to the Siema Database

   Siema is the IBAMA system responsible for registering reports of environmental accidents from citizens across Brazil. According to database, 28% of the reports are false, resulting in financial and human costs to the institution during the verification of these incidents. The proposal in this project is to use machine learning techniques to build a component called Hoax for Siema, with the aim of classifying future accident reports as valid or not.

   The article **Classificador de notificações de acidentes ambientais no Sistema Nacional de Emergências Ambientais do IBAMA** contained in `docs\paper\` was submitted and approved at the XI Workshop de Computação Aplicada em Governo Eletrônico ([WCGE](https://csbc.sbc.org.br/2023/wcge/)), a satellite event of the 43º Congresso da Sociedade Brasileira de Computação (CSBC). It serves as a forum for presenting and discussing ICT techniques, methodologies, models, and tools to support Electronic Government initiatives.

## Reproducing the Experiments

Using any version of the Windows operating system with [Python 3](https://www.python.org/) and [Git](https://git-scm.com/) installed.

Clone the repository:

```bash
  git clone https://github.com/filipemedeiross/hoax.git
```

Navigate to the project directory:

```bash
  cd hoax
```

Create a virtual environment (in this example, it's named `.venv`):

```bash
  python -m venv .venv
```

Activate the virtual environment:

```bash
  .venv\Scripts\activate
```

Install all necessary libraries as specified in requirements.txt:

```bash
  pip install -r requirements.txt
```

Load the database:

```bash
  dvc pull
```

At this point, you can verify the pipeline's structure using (it resembles the following directed acyclic graph):

```bash
  dvc dag
```

<p align="center"> 
    <img src="https://github.com/filipemedeiross/hoax/blob/main/docs/media/dag.png" width="650" height="400">
</p>

> Pipeline description:
> - Starts with data featurization
> - Training three different machine learning models
> - Concludes with model evaluation

You can enqueue multiple experiments for later execution with:

```bash
  dvc exp run --queue -S [...] -S [...] ...
```

And you will receive a result similar to the following:

<p align="center"> 
    <img src="https://github.com/filipemedeiross/hoax/blob/main/docs/media/queuing_experiments.png?raw=true" width="650" height="400">
</p>

To begin experimentation, use `dvc exp run --run-all` you will notice that only the changed stages will be executed, and those without modifications will utilize the cached result, as follows:

<p align="center"> 
    <img src="https://github.com/filipemedeiross/hoax/blob/main/docs/media/run_exp1.png?raw=true" width="400" height="250">
    <img src="https://github.com/filipemedeiross/hoax/blob/main/docs/media/run_exp2.png?raw=true" width="400" height="250">
</p>

To list the obtained results, you can use `dvc exp show --only-changed` and will receive a result similar to the following:

<p align="center"> 
    <img src="https://github.com/filipemedeiross/hoax/blob/main/docs/media/exp_show.png?raw=true" width="650" height="400">
</p>

If you only want to execute the pipeline found in `dvc.yaml` with the parameters saved in `params.yaml`:

```bash
  dvc repro
```

And to check the metrics and evaluation graphs of the models:

```bash
  dvc metrics show

  dvc plots show
```

## References

Yaser S. Abu-Mostafa, Malik Magdon-Ismail, and Hsuan-Tien Lin. **Learning from Data**. AMLBook, 2012.
