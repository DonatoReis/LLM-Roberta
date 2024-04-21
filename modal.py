import optuna
import os
import sys
import threading
import time
import torch
import warnings
import sklearn
import numpy as np
import logging

from datasets import load_dataset
from pathlib import Path
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.tensorboard import SummaryWriter
from transformers import DataCollatorForLanguageModeling, RobertaConfig, RobertaForCausalLM, RobertaTokenizerFast, Trainer, TrainingArguments
from optuna.visualization import plot_optimization_history

# Configurações de ambiente
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = '1'
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuração de logging
logging.basicConfig(level=logging.INFO)

# Diretórios
MODEL_DIR = Path.cwd() / 'model_dir'
TRAINING_DATA_DIR = Path.cwd() / 'training_data'

# Cria os diretórios se eles não existirem
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Constantes para treinamento
LEARNING_RATE = 0.00013153683519278748
WEIGHT_DECAY = 0.054383403283778595
ADAM_EPSILON = 8.46211496713214e-07
NUM_TRAIN_EPOCHS = 2
PER_DEVICE_TRAIN_BATCH_SIZE = 16

# Definindo as cores ASCII como variáveis
COLORS = {
    "blue": "\033[94m",
    "red": "\033[91m",
    "purple": "\033[95m",
    "cyan": "\033[96m",
    "end": "\033[0m"
}

# Função para verificar se uma GPU está disponível e, se não, usar a CPU (Usamos a GPU se estiver disponível para acelerar o treinamento, caso contrário, usamos a CPU.)
def set_device():
    """
    Função para verificar se uma GPU está disponível e, se não, usar a CPU
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Função para criar o diretório se ele não existir
def create_directory(path):
    """
    Função para criar o diretório se ele não existir
    """
    Path(path).mkdir(parents=True, exist_ok=True)

# Função para otimização de hiperparâmetros com Optuna
def objective(trial, model, tokenized_datasets, data_collator, tokenizer):
    """
    Função para otimização de hiperparâmetros com Optuna
    """
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 4)
    weight_decay = trial.suggest_float("weight_decay", 1e-12, 1e-1, log=True)
    adam_epsilon = trial.suggest_float("adam_epsilon", 1e-10, 1e-6, log=True)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])

    print(f"\nTestando Hiperparametros: learning_rate={learning_rate}, num_train_epochs={num_train_epochs}, weight_decay={weight_decay}, adam_epsilton={adam_epsilon}, per_device_train_batch_size={per_device_train_batch_size}\n")

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        adam_epsilon=adam_epsilon,
        save_steps=1_000,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=10,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    #logging.getLogger("transformers.trainer").setLevel(logging.ERROR)

    # Treine o modelo
    trainer.train()

    eval_loss = None
    if trainer.eval_dataset is not None:
        eval_result = trainer.evaluate()
        eval_loss = eval_result["eval_loss"]
        print(f"Eval loss: {eval_loss}")

        # Aqui adicionamos a acurácia do modelo, semelhante ao segundo script.
        preds = trainer.predict(trainer.eval_dataset)
        pred_labels = np.argmax(preds.predictions, axis=-1)
        accuracy = sklearn.metrics.accuracy_score(trainer.eval_dataset['labels'], pred_labels)
        print(f"Acurácia: {accuracy}")

    return eval_loss if eval_loss is not None else float('inf')

# Função principal
def main():
    # Função para exibir uma animação de carregamento durante o treinamento
    def spin():
        print()
        frames = [f"{COLORS['blue']}{frame}{COLORS['end']}" for frame in ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]]
        messages = ["Preparando os pacotes", "Aperfeiçoando melhorias", "Codificando tokens"]
        start_time = time.time()
        frame_index = 0
        msg_index = 0
        while True:
            print('\r' + ' ' * 50, end='')  # Limpa a linha
            print('\r' + frames[frame_index] + ' ' + messages[msg_index], end='')
            sys.stdout.flush()
            time.sleep(0.08)
            frame_index = (frame_index + 1) % len(frames)  # Atualiza o índice do frame
            if time.time() - start_time > 1:  # Verifica se já se passaram 2 segundos
                start_time = time.time()  # Reinicia o contador de tempo
                msg_index = (msg_index + 1) % len(messages)  # Atualiza o índice da mensagem
                if msg_index == 0:  # Se todas as mensagens foram exibidas, encerra o ciclo
                    print('\r' + ' ' * 50, end='')  # Limpa a linha
                    print(f'\r{COLORS["blue"]}✔{COLORS["end"]} Refinamento concluído.')  # Imprime o símbolo de verificação e a mensagem final
                    break

    # Cria uma thread para o spinner
    spinner = threading.Thread(target=spin)

    # Inicia o spinner
    spinner.start()

    # Interrompe o spinner
    spinner.do_run = False
    spinner.join()

    # Pergunte ao usuário se eles querem treinar o modelo ou entrar no chat
    mode = input(f"\n Digite '{COLORS['blue']}treinar{COLORS['end']}' para iniciar o treinamento do modelo ou '{COLORS['red']}chat{COLORS['end']}' para conversar com a IA: ")
    print()

    device = set_device()

    create_directory(MODEL_DIR)

    def count_words_in_files(files):
        total_words = 0
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
                words = text.split()
                total_words += len(words)
        return total_words

    if mode.lower() == 'treinar':
        files = [str(path) for path in Path(TRAINING_DATA_DIR).glob('*.txt')]
        total_words = count_words_in_files(files)

        # Inicialize um ByteLevelBPETokenizer
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=files, vocab_size=50265, min_frequency=2, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])

        tokenizer.save_model(str(MODEL_DIR))

        tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR, model_max_lenght=512)

        tokenizer._tokenizer.post_process = BertProcessing(
        ("</s>", tokenizer.convert_tokens_to_ids("</s>")),
        ("<s>", tokenizer.convert_tokens_to_ids("<s>")),)

        # Criando nosso Transformer
        config = RobertaConfig(
            vocab_size=52_000,
            max_position_embeddings=512,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
            is_decoder=True,
        )

        # Criando nosso Transformer
        model = RobertaForCausalLM.from_pretrained('roberta-base', is_decoder=True) # Pode passar seus proprios parametros utilizando config=config
        print("\n Total de parametros: ",model.num_parameters(), " ")
        print(" Número de parâmetros treináveis: ",model.num_parameters(only_trainable=True), " ")
        print(f" Total de palavras nos arquivos de treinamento: {total_words}  \n")

        # Mova o modelo para o dispositivo
        model = model.to(device)

        # Carregando o dataset
        dataset = load_dataset('text', data_files=files)

        # Tokenizando o dataset
        def tokenize_function(examples):
            """Tokeniza o dataset."""
            encodings = tokenizer(examples["text"], truncation=True, padding='max_length', max_length=32)

            # Abrindo o arquivo em modo de escrita
            with open('indices.txt', 'w') as f:
                # Escrevendo os índices no arquivo
                f.write("input_ids: " + str(encodings['input_ids']) + "\n")
            
            return {"input_ids": encodings['input_ids']}
        
        tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"])
        eval_dataset = tokenized_datasets['validation'] if 'validation' in tokenized_datasets else None

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.1)

        output_dir = os.path.abspath(MODEL_DIR)

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            adam_epsilon=ADAM_EPSILON,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            save_steps=1_000,
            save_total_limit=2,
            prediction_loss_only=True,
            logging_steps=10,
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=eval_dataset,
        )

        # Pergunte ao usuário se eles gostariam de usar o Optuna para otimização de hiperparâmetros ou utilizar os parametros personalizados.
        optuna_mode = input(f"\n➤ Você gostaria de usar as configurações padrão ou Optuna para treinar? ({COLORS['blue']}padrao{COLORS['end']}/{COLORS['red']}optuna{COLORS['end']}): ")
        print()
        # Defina um nome para o seu estudo e o caminho do banco de dados
        study_name = "tess"
        db_path = "example.db"
        if optuna_mode.lower() == 'optuna':
            study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{db_path}", direction="minimize", load_if_exists=True)
            study.optimize(lambda trial: objective(trial, model, tokenized_datasets, data_collator, tokenizer), n_trials=5)

            # Obtenha os trials que têm um valor
            valid_trials = [trial for trial in study.trials if trial.value is not None]

            # Obtenha os 10 melhores trials
            best_trials = sorted(valid_trials, key=lambda trial: trial.value)[:10]

            # Abra o arquivo em modo de escrita
            with open('best_trials.txt', 'w') as f:
                for i, trial in enumerate(best_trials):
                    f.write(f"Trial {i+1}\n")
                    f.write(f"Value: {trial.value}\n")
                    f.write(f"Params: {trial.params}\n")
                    f.write("\n")

            if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) > 0:
                best_params = study.best_params
                print("\nMelhores hiperparâmetros: ", best_params)
                # Visualize os resultados
                fig = plot_optimization_history(study)
                fig.write_html("optuna_visualization.html")
                fig.show()
            else:
                print("Nenhuma tentativa foi concluída.")
        else:
            # Obtém o estado do treinador e a saída do treinamento
            trainer_state = trainer.state
            train_output = trainer.train() # Treine o modelo

            # Crie um SummaryWriter
            writer = SummaryWriter()

            # Registre a perda de treinamento com o writer
            writer.add_scalar('Loss/train', train_output.training_loss, trainer_state.global_step)

            trainer.save_model(MODEL_DIR)

            # Feche o writer no final do treinamento
            writer.close()

        # Chame a função chat para iniciar a conversação
        chat(model, tokenizer, device)

    elif mode.lower() == 'chat':
        # Carregue o modelo e o tokenizador previamente treinados
        model = RobertaForCausalLM.from_pretrained(MODEL_DIR).to(device)
        tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)

        # Chame a função chat
        chat(model, tokenizer, device)
    else:
        print("Modo desconhecido. Por favor, digite 'treinar' ou 'chat'.")

# Função Convesar
def chat(model, tokenizer, device):
    # Solicita os parâmetros ao usuário
    do_sample = input("\nEscolha uma amostra (True/False): ") == "True" # Se definido como True, o modelo irá amostrar as palavras de saída em vez de usar a palavra com a maior probabilidade. Isso pode levar a resultados mais diversos, mas também pode produzir sequências menos coerentes.
    temperature = 1.0  # valor padrão para quando do_sample é False
    if do_sample:
        temp_input = input("\nTemperature (e.g. 0.5, press Enter for default 0.7): ") # temperature: Este parâmetro controla a “aleatoriedade” das previsões do modelo. Um valor de temperatura mais alto (por exemplo, 1.0) torna as previsões do modelo mais aleatórias, enquanto um valor de temperatura mais baixo (por exemplo, 0.7, como você está usando) torna as previsões do modelo mais determinísticas.
        if temp_input != "":
            temperature = float(temp_input)
        else:
            temperature = 0.7  # valor padrão para quando do_sample é True

    # Loop infinito
    while True:
        question = input("\nVocê: ") # Solicita uma pergunta ao usuário
        if question.lower() == 'sair': # Se a pergunta for "sair", sai do loop
            break
        input_ids = tokenizer.encode(question, return_tensors='pt') # Codifica a pergunta e adiciona os tokens necessários
        input_ids = input_ids.to(device) # Move os input_ids para a mesma GPU onde o modelo está
        output = model.generate(input_ids, max_length=50, num_return_sequences=1, do_sample=do_sample ,temperature=temperature) # Gera uma resposta para a pergunta
        response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True) # Decodifica a resposta
        print("\nIA: ", response) # Imprime a resposta

if __name__ == '__main__':
    main()
