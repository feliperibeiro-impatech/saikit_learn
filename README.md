# Saikit-learn

![PyPI](https://img.shields.io/pypi/v/saikit-learn?color=FFD43B)
![Python](https://img.shields.io/pypi/pyversions/saikit-learn?color=336B9C)
![License](https://img.shields.io/github/license/feliperibeiro-impatech/saikit_learn)
![Repository Size](https://img.shields.io/github/repo-size/feliperibeiro-impatech/saikit_learn)

## 🚩 Introdução

A **Saikit-learn** é uma biblioteca desenvolvida em Python como projeto final da disciplina de Aprendizado de Máquina I, do curso de graduação do [IMPA Tech](https://impatech.edu.br/), durante o período 2026.2.

O objetivo do projeto é desenvolver uma biblioteca instalável que reúna ferramentas e implementações relacionadas ao aprendizado de máquina, permitindo sua utilização e experimentação em diferentes aplicações.

A biblioteca foi desenvolvida colaborativamente pelos integrantes do grupo, colocando em prática os conhecimentos adquiridos ao longo da disciplina.

<br>

**Instalação e utilização:**

```bash
pip install saikit-learn
```

<br>

**Sumário**

- [🚩 Introdução](#-introdução)
- [🛠️ Colaboradores](#️-colaboradores)
- [⌨️ Instalação](#️-instalação)
- [📦 Dependências](#-dependências)
- [🧪 Testes](#-testes)
- [🔗 Referências](#-referências)
- [📝 Licença](#-licença)

## 🛠️ Colaboradores

Este projeto foi desenvolvido graças ao trabalho conjunto dos integrantes do grupo. Agradecemos a todos pela dedicação, colaboração e contribuição durante o desenvolvimento da biblioteca.

| Integrante | 
|:---:|
| André Oliveira |
| Davi Pestana |
| Felipe Ribeiro |
| Gabriel Krauss |
| Huann Vicente |
| Lucas Corazza |
| Nicole Freire |

## ⌨️ Instalação

### Básico

Para utilizar a biblioteca, é necessário possuir o **Python 3.8 ou superior**.

A instalação pode ser realizada diretamente pelo gerenciador de pacotes `pip`:

```bash
pip install saikit-learn
```

Após a instalação, a biblioteca estará disponível no ambiente Python.

### Desenvolvedores

Caso queira obter o código-fonte e trabalhar diretamente no projeto, clone o repositório:

```bash
git clone https://github.com/feliperibeiro-impatech/saikit_learn.git
```

Entre na pasta do projeto:

```bash
cd saikit_learn
```

Para instalar a biblioteca em modo de desenvolvimento, juntamente com as dependências de desenvolvimento, execute:

```bash
pip install -e ".[dev]"
```

## 📦 Dependências

A biblioteca utiliza as seguintes dependências:

- [NumPy](https://github.com/numpy/numpy): biblioteca utilizada para operações numéricas e manipulação de arrays.
- [Pytest](https://github.com/pytest-dev/pytest): ferramenta utilizada para a execução dos testes automatizados (dependência de desenvolvimento).

## 🧪 Testes

A biblioteca possui testes automatizados para auxiliar na verificação do funcionamento de suas implementações.

Para executar os testes, primeiro certifique-se de que as dependências de desenvolvimento estejam instaladas.

Em seguida, execute o comando na raiz do projeto:

```bash
pytest
```

O Pytest identificará e executará automaticamente os testes disponíveis no diretório `tests/`.

## 🔗 Referências

- [Repositório no GitHub](https://github.com/feliperibeiro-impatech/saikit_learn)
- [Pacote no PyPI](https://pypi.org/project/saikit-learn/)
- [Documentação do NumPy](https://numpy.org/doc/)
- [Documentação do Pytest](https://docs.pytest.org/)

## 📝 Licença

Este projeto é distribuído sob os termos da [Licença MIT](./LICENSE), sendo um software de código aberto (*open source*).

