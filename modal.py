import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from datasets import load_dataset

def read_document(document_path):
    with open(document_path, 'r', encoding='utf-8') as file:
        return file.read()

def main():
    # Cria o diretório se ele não existir
    if not os.path.exists('raw_model'):
        os.makedirs('raw_model')

    dados_treino = 'crepusculoDosIdolos.txt'

    os.path.join(os.path.abspath('.'), 'raw_model')

    # Initialize a ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[dados_treino], vocab_size=52_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    tokenizer.save_model('raw_model')

    tokenizer = RobertaTokenizerFast.from_pretrained('raw_model', max_len=512)

    tokenizer._tokenizer.post_process = BertProcessing(
        ("</s>", tokenizer.convert_tokens_to_ids("</s>")),
        ("<s>", tokenizer.convert_tokens_to_ids("<s>")),
    )

    # Criando nosso Transformer
    config = RobertaConfig(
        vocab_size=52_000,
        max_position_embeddings=512,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )

    model = RobertaForMaskedLM(config=config)
    model.num_parameters()

    # Verifique se uma GPU está disponível e, se não, use a CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Mova o modelo para o dispositivo
    model = model.to(device)

    # Carregando o dataset
    dataset = load_dataset('text', data_files=dados_treino)

    # Tokenizando o dataset
    def tokenize_function(examples):
        encodings = tokenizer(examples["text"], truncation=True, padding='max_length', max_length=512)
        #print("input_ids:", encodings['input_ids'])
        return {"input_ids": encodings['input_ids']}

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.1
    )

    output_dir = os.path.abspath('raw_model')

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets['train'],
    )

    # Verifica se o diretório de saída existe antes de iniciar o treinamento
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Habilita a precisão mista
    trainer.train()

    trainer.save_model('raw_model')

    # Crie um pipeline de preenchimento de máscara com o modelo e o tokenizador
    fill_mask = pipeline(
        "fill-mask",
        model='raw_model',
        tokenizer='raw_model'
    )

    while True:  # Loop infinito
        question = input("Você: ")  # Solicita uma pergunta ao usuário
        if question.lower() == "sair":  # Se a pergunta for "sair"
            break  # Sai do loop
        response = fill_mask(question)  # Gera uma resposta para a pergunta
        print("IA: ", response)  # Imprime a resposta

if __name__ == '__main__':
    main()
