import os
from torch.utils.tensorboard import SummaryWriter
# Configurações de ambiente
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline
from transformers import RobertaForCausalLM
from transformers import TextGenerationPipeline
from datasets import load_dataset

def main():
    # Pergunte ao usuário se eles querem treinar o modelo ou entrar no chat
    mode = input("\nDigite 'treinar' para treinar o modelo ou 'chat' para entrar no chat: ")

    # Verifique se uma GPU está disponível e, se não, use a CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if mode.lower() == 'treinar':
        # Cria o diretório se ele não existir
        if not os.path.exists('raw_model'):
            os.makedirs('raw_model')

        dados_treino = 'C:\\Users\\caiqu\\OneDrive\\Documentos\\IA\\biblia.txt'

        os.path.join(os.path.abspath('.'), 'raw_model')

        # Inicialize um ByteLevelBPETokenizer
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=[dados_treino], vocab_size=52_000, min_frequency=2, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])

        tokenizer.save_model('raw_model')

        tokenizer = RobertaTokenizerFast.from_pretrained('raw_model', model_max_lenght=512)

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
        model = RobertaForCausalLM(config=config)
        print("Total de parametros: ", model.num_parameters())
        print("Número de parâmetros treináveis: ", model.num_parameters(only_trainable=True))

        # Mova o modelo para o dispositivo
        model = model.to(device)

        # Crie um pipeline de geração de texto com o modelo e o tokenizador
        text_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)

        # Carregando o dataset
        dataset = load_dataset('text', data_files=dados_treino)

        # Tokenizando o dataset
        def tokenize_function(examples):
            encodings = tokenizer(examples["text"], truncation=True, padding='max_length', max_length=128)

            # Abrindo o arquivo em modo de escrita
            with open('indices.txt', 'w') as f:
                # Escrevendo os índices no arquivo
                f.write("input_ids: " + str(encodings['input_ids']) + "\n")
            
            return {"input_ids": encodings['input_ids']}
        
        tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"])

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.1)

        output_dir = os.path.abspath('C:\\Users\\caiqu\\OneDrive\\Documentos\\IA\\raw_model')

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=100,
            per_device_train_batch_size=16,
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
        )

        # Verifica se o diretório de saída existe antes de iniciar o treinamento
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Crie um SummaryWriter
        writer = SummaryWriter()

        # Obtém o estado do treinador e a saída do treinamento
        trainer_state = trainer.state
        train_output = trainer.train()

        # Registre a perda de treinamento com o writer
        writer.add_scalar('Loss/train', train_output.training_loss, trainer_state.global_step)

        # Formata a saída como uma string
        output_str = f"""
        Trainer State:
        Epoch: {trainer_state.epoch}
        Global Step: {trainer_state.global_step}
        Max Steps: {trainer_state.max_steps}
        Train Batch Size: {trainer_state.train_batch_size}
        Num Train Epochs: {trainer_state.num_train_epochs}
        Total FLOPs: {trainer_state.total_flos}

        Train Output:
        Global Step: {train_output.global_step}
        Training Loss: {train_output.training_loss}
        Train Runtime: {train_output.metrics['train_runtime']}
        Train Samples Per Second: {train_output.metrics['train_samples_per_second']}
        Train Steps Per Second: {train_output.metrics['train_steps_per_second']}
        Train Loss: {train_output.metrics['train_loss']}
        Epoch: {train_output.metrics['epoch']}
        """

        # Imprime a saída formatada
        print(output_str)

        trainer.save_model('raw_model')

        # Feche o writer no final do treinamento
        writer.close()

    # Chame a função chat para iniciar a conversação
        chat(model, tokenizer, device)
    elif mode.lower() == 'chat':
        # Carregue o modelo e o tokenizador previamente treinados
        model = RobertaForCausalLM.from_pretrained('raw_model').to(device)
        tokenizer = RobertaTokenizerFast.from_pretrained('raw_model')

    # Chame a função chat
        chat(model, tokenizer, device)
    else:
        print("Modo desconhecido. Por favor, digite 'treinar' ou 'chat'.")

def chat(model, tokenizer, device):
    # Solicita os parâmetros ao usuário
    do_sample = input("\nDo sample (True/False): ") == "True"
    temperature = 0.7  # valor padrão
    if do_sample:
        temperature = float(input("Temperature (e.g. 0.5): "))

    # Loop infinito
    while True:
        question = input("\nVocê: ") # Solicita uma pergunta ao usuário
        if question.lower() == 'sair': # Se a pergunta for "sair", sai do loop
            break
        input_ids = tokenizer.encode(question, return_tensors='pt') # Codifica a pergunta e adiciona os tokens necessários
        input_ids = input_ids.to(device) # Move os input_ids para a mesma GPU onde o modelo está
        if do_sample:
            output = model.generate(input_ids, max_length=150, num_return_sequences=1, do_sample=do_sample ,temperature=temperature) # Gera uma resposta para a pergunta
        else:
            output = model.generate(input_ids, max_length=150, num_return_sequences=1, do_sample=do_sample) # Gera uma resposta para a pergunta
        response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True) # Decodifica a resposta
        print("\nIA: ", response) # Imprime a resposta

if __name__ == '__main__':
    main()
