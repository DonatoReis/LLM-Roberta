# Treinamento de Modelo de Linguagem com Transformers e Optuna

## Descrição
Este script é usado para treinar um modelo de linguagem Roberta usando a biblioteca Transformers da Hugging Face e otimização de hiperparâmetros com Optuna. O script começa importando todas as bibliotecas necessárias, definindo algumas variáveis de ambiente e configurando o logging.

O script usa o tokenizador ByteLevelBPETokenizer para tokenizar o texto e salva o tokenizador treinado. Em seguida, ele inicializa um modelo Roberta com uma configuração específica e move o modelo para a GPU, se disponível.

O script carrega o dataset de texto, tokeniza o dataset e prepara um DataCollator para modelagem de linguagem. Ele define alguns argumentos de treinamento e inicializa um Trainer com o modelo, argumentos de treinamento, DataCollator e dataset de treinamento.

Finalmente, o script treina o modelo e salva o modelo treinado. Ele também oferece a opção de usar Optuna para otimização de hiperparâmetros.

## Objetivo

O objetivo principal do script é treinar o modelo para prever a próxima palavra em uma sequência de texto.

## Requisitos
- Python 3.12.3
- PyTorch
- Transformers
- Datasets
- Tokenizers
- scikit-learn
- Numpy

## Instalação
Para instalar as dependências, você pode usar o seguinte comando:
```bash
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Uso
Para usar este script, você pode executá-lo diretamente a partir da linha de comando com:

```bash
python modal.py
```

Depois de treinado, o modelo pode ser usado para preencher lacunas em sentenças. Basta digitar uma frase com uma lacuna (representada por <mask>) e o modelo tentará preencher a lacuna.

## Otimização de Hiperparâmetros

Este script também suporta a otimização de hiperparâmetros usando Optuna. Para usar essa funcionalidade, escolha ‘optuna’ quando solicitado a selecionar o modo de treinamento. O script irá então realizar uma série de experimentos para encontrar os melhores hiperparâmetros para o seu modelo. Os resultados serão salvos e podem ser visualizados usando a biblioteca de visualização Optuna.

## Contribuição
Contribuições são bem-vindas! Por favor, faça um fork deste repositório e abra um Pull Request.

## Licença
Este projeto está licenciado sob a licença MIT.
