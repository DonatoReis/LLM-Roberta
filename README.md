# Treinamento de Modelo de Linguagem com Transformers

## Descrição
Este script é usado para treinar um modelo de linguagem Roberta usando a biblioteca Transformers da Hugging Face. O script começa importando todas as bibliotecas necessárias e definindo algumas variáveis de ambiente. Em seguida, ele lê um documento de texto para treinar o modelo.

O script usa o tokenizador ByteLevelBPETokenizer para tokenizar o texto e salva o tokenizador treinado. Em seguida, ele inicializa um modelo Roberta com uma configuração específica e move o modelo para a GPU, se disponível.

O script então carrega o dataset de texto, tokeniza o dataset e prepara um DataCollator para modelagem de linguagem. Ele define alguns argumentos de treinamento e inicializa um Trainer com o modelo, argumentos de treinamento, DataCollator e dataset de treinamento.

Finalmente, o script treina o modelo e salva o modelo treinado. Ele também cria um pipeline de preenchimento de máscara que pode ser usado para preencher lacunas em sentenças.

## Requisitos
- Python 3.12.3
- PyTorch
- Transformers
- Datasets
- Tokenizers

## Instalação
Para instalar as dependências, você pode usar o seguinte comando:
```bash
pip install transformers datasets tokenizers
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Uso
Para usar este script, você pode executá-lo diretamente a partir da linha de comando com:

```bash
python modal.py
```

Depois de treinado, o modelo pode ser usado para preencher lacunas em sentenças. Basta digitar uma frase com uma lacuna (representada por <mask>) e o modelo tentará preencher a lacuna.

## Contribuição
Contribuições são bem-vindas! Por favor, faça um fork deste repositório e abra um Pull Request.

## Licença
Este projeto está licenciado sob a licença MIT.
