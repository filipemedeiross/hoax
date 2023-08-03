# HOAX

## Aplicação de Técnicas de Aprendizado de Máquina a Base de Dados do Siema

   O Siema é o sistema do IBAMA responsável pelo cadastro dos relatos de acidentes ambientais de cidadãos de todo o Brasil. Segundo a base de dados do próprio IBAMA, 28% dos relatos são falsos, gerando um custo financeiro e humano para a instituição durante a verificação destas supostas ocorrências. 
   
   Propõe-se a utilização de técnicas de aprendizado de máquina para a construção de um componente, chamado Hoax, para o Siema, com o objetivo de classificar futuros relatos de acidentes como válidos ou não.

   O artigo contido em `paper\` foi submetido e aprovado no XI Workshop de Computação Aplicada em Governo Eletrônico ([WCGE](https://csbc.sbc.org.br/2023/wcge/)), evento satélite do 43º Congresso da Sociedade Brasileira de Computação (CSBC) e que constitui-se de um fórum para apresentação e discussão de técnicas, metodologias, modelos e ferramentas de TICs para apoio a iniciativas de Governo Eletrônico.

## Reproduzindo os Experimentos

Utilizando alguma versão do sistema operacional Windows com o [Python 3](https://www.python.org/) e o [Git](https://git-scm.com/) instalado.

Clone o repositório:

```bash
  git clone https://github.com/filipemedeiross/hoax.git
```

Acesse o diretório do projeto:

```bash
  cd hoax
```

Criando o ambiente virtual (para o exemplo foi nomeado como `.venv`):

```bash
  python -m venv .venv
```

Ativando o ambiente virtual:

```bash
  .venv\Scripts\activate
```

Instalando todas as bibliotecas necessárias e especificadas em requirements.txt:

```bash
  pip install -r requirements.txt
```

Carregue a base de dados:

```bash
  dvc pull
```

Execute o pipeline presente em `dvc.yaml` com os parâmetros salvos em `params.yaml`:

```bash
  dvc repro
```

Verifique as métricas e gráficos de avaliação dos modelos:

```bash
  dvc metrics show
```

```bash
  dvc plots show
```
