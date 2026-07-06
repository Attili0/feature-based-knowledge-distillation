import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import multiprocessing as mp
from pathlib import Path
import matplotlib.pyplot as plt

# Importamos as configurações e funções de dados do arquivo da Fase 1
from train import (
    INPUT_FOLDER, OUTPUT_FOLDER, MODELS, PARALEL, check_gpu_thermal_safety,
    get_cifar100_dataloaders, get_intel_scenes_dataloaders,
    log_info, log_warning
)

num_epochs = 100
INTERVAL = 5 # Intervalo de épocas para checar temperatura da GPU
STUDENT = "shufflenet_v2" # Pode ser "mobilenet_v2", "shufflenet_v2" ou "efficientnet_b0"


class PostGAPStudent(nn.Module):
    def __init__(self, teacher_feature_dim, student_name=STUDENT):
        super().__init__()
        
        if student_name == "mobilenet_v2":
            self.encoder = torchvision.models.mobilenet_v2(width_mult=0.5)
            encoder_out_dim = self.encoder.last_channel
            self.encoder.classifier = nn.Identity()
            
        elif student_name == "shufflenet_v2":
            self.encoder = torchvision.models.shufflenet_v2_x0_5()
            # ShuffleNet usa 'fc' na última camada
            encoder_out_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
            
        elif student_name == "efficientnet_b0":
            self.encoder = torchvision.models.efficientnet_b0()
            # EfficientNet usa um Sequential 'classifier' onde o índice 1 é a Linear
            encoder_out_dim = self.encoder.classifier[1].in_features
            self.encoder.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Estudante {student_name} não suportado.")
        
        # PREDICTOR: Mapeia as features do estudante para a dimensão do professor
        # Isso atende à exigência da Fase 2 de usar blocos densos para prever o vetor post-GAP
        self.predictor = nn.Sequential(
            nn.Linear(encoder_out_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, teacher_feature_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        predicted_teacher_features = self.predictor(features)
        return predicted_teacher_features


def split_teacher_encoder_classifier(teacher_model):
    """
    Separa o modelo professor em (Encoder_Congelado, Classificador_Treinado)
    e retorna a dimensão (C) do vetor Post-GAP.
    """
    if hasattr(teacher_model, 'fc'): # ResNets
        classifier = teacher_model.fc
        feature_dim = classifier.in_features
        teacher_model.fc = nn.Identity()
    elif hasattr(teacher_model, 'classifier'): # VGGs e ConvNeXts
        classifier = teacher_model.classifier
        # ConvNeXt tem um nn.Sequential, VGG também. Vamos pegar a dimensão de entrada da última linear
        if isinstance(classifier, nn.Sequential):
            feature_dim = classifier[-1].in_features
            linears = [layer for layer in classifier if isinstance(layer, nn.Linear)]
            if linears:
                feature_dim = linears[0].in_features  # Para VGG pegará 25088; para ConvNeXt pegará 1024
            else:
                feature_dim = classifier[-1].in_features
        else:
            feature_dim = classifier.in_features
        teacher_model.classifier = nn.Identity()
    else:
        raise ValueError("Arquitetura de professor não reconhecida para extração de features.")
        
    return teacher_model, classifier, feature_dim


def train_student_distillation(teacher_model, student_model, classifier, train_loader, val_loader, 
                               model_name, save_dir, gpu_id, alpha=1.0, beta=1.0):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # Prepara Professor e Classificador (Totalmente congelados)
    teacher_model = teacher_model.to(device).eval()
    classifier = classifier.to(device).eval()
    for param in teacher_model.parameters(): param.requires_grad = False
    for param in classifier.parameters(): param.requires_grad = False
        
    # Prepara Estudante (Sendo treinado)
    student_model = student_model.to(device)
    trainable_params = filter(lambda p: p.requires_grad, student_model.parameters())
    optimizer = optim.Adam(trainable_params, lr=1e-3)
    # optimizer = optim.Adam(student_model.parameters(), lr=1e-3)
    
    # Funções de Perda do Projeto
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    
    epochs = num_epochs
    best_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}

    # Para saber se a arquitetura atual exige tensores 4D no classificador
    requires_4d_classifier = "convnext" in model_name.lower()
    
    # scaler = torch.amp.GradScaler('cuda') # para modelos muito grandes, mas aqui não é necessário
    epoch = 0
    for epoch in range(epochs):
        epoch += 1
        student_model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            # with torch.autocast(device_type='cuda', dtype=torch.float16):
                
            #     with torch.no_grad():
            #         real_teacher_features = teacher_model(inputs)

            #     is_4d = real_teacher_features.dim() == 4
            #     if is_4d:
            #         real_teacher_features_flat = torch.flatten(real_teacher_features, 1)
            #     else:
            #         real_teacher_features_flat = real_teacher_features
                
            #     predicted_features = student_model(inputs)

            #     if is_4d:
            #         classifier_input = predicted_features.unsqueeze(2).unsqueeze(3)
            #     else:
            #         classifier_input = predicted_features
                
            #     student_logits = classifier(classifier_input)
                
            #     loss_mse = mse_loss_fn(predicted_features, real_teacher_features_flat)
            #     loss_ce = ce_loss_fn(student_logits, labels)
            #     loss = (alpha * loss_mse) + (beta * loss_ce)
            
            # # Substituímos o loss.backward() e optimizer.step() por:
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            
            # 1. Extrai features reais do professor (Sem gradiente)
            with torch.no_grad():
                real_teacher_features = teacher_model(inputs)

            # Verifica se o professor devolveu 4D (ex: ConvNeXt -> B, C, 1, 1)
            is_4d = real_teacher_features.dim() == 4

            if is_4d:
                # Nivela as features reais para 2D (B, C) para podermos calcular o MSE corretamente
                real_teacher_features_flat = torch.flatten(real_teacher_features, 1)
            else:
                real_teacher_features_flat = real_teacher_features
            
            # 2. Estudante tenta prever as features
            predicted_features = student_model(inputs)

            # 3. Prepara a entrada para o classificador original do professor
            if is_4d:
                # O classificador do ConvNeXt exige 4D de volta. 
                # Adicionamos as dimensões vazias: (B, C) -> (B, C, 1, 1)
                classifier_input = predicted_features.unsqueeze(2).unsqueeze(3)
            else:
                classifier_input = predicted_features
            
            # 3. Passa as features do estudante pelo classificador do professor
            student_logits = classifier(classifier_input)
            
            # 4. Calcula a Perda Combinada (MSE do Predictor + CE do Classificador)
            loss_mse = mse_loss_fn(predicted_features, real_teacher_features_flat)
            loss_ce = ce_loss_fn(student_logits, labels)
            loss = (alpha * loss_mse) + (beta * loss_ce)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            try:
                del inputs, labels, real_teacher_features, predicted_features 
                del classifier_input, student_logits, loss, loss_mse, loss_ce
            except Exception as e:
                log_warning(f"Erro ao liberar memória: {e}", gpu_id)
        
        if (epoch + 1) % INTERVAL == 0:
            check_gpu_thermal_safety(gpu_id)
            
        # Validação
        student_model.eval()
        correct = 0
        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                predicted_features = student_model(inputs)

                if requires_4d_classifier:
                    classifier_input = predicted_features.unsqueeze(2).unsqueeze(3)
                else:
                    classifier_input = predicted_features

                logits = classifier(classifier_input)
                _, preds = torch.max(logits, 1)
                correct += (preds == labels.data).sum().item()

        acc = correct / len(val_loader.dataset) # acc é obtido como float
        # acc = correct.double() / len(val_loader.dataset)
        history['train_loss'].append(running_loss / len(train_loader))
        history['val_acc'].append(acc)
        
        log_info(f"Época {epoch+1}/{epochs} | Modelo: {model_name} | Val Acc: {acc:.4f}", gpu_id)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(student_model.state_dict(), save_dir / f"student_of_{model_name}_best.pth")
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), history['val_acc'], label='Val Accuracy (Student)', color='green', marker='o')
    plt.title(f'Curva de Aprendizado - Estudante de {model_name}')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.grid(True)
    plt.legend()
    # Salva na mesma pasta onde o modelo foi salvo
    plot_path = save_dir / f"student_of_{model_name}_accuracy_plot.png"
    plt.savefig(plot_path)
    plt.close() # Libera a memória do servidor
            
    return history


def run_phase2_pipeline(gpu_id, task_queue):
    while not task_queue.empty():
        try:
            ds_name, model_name, model_fn, weights = task_queue.get_nowait()
        except mp.queues.Empty:
            log_info("Fila de tarefas vazia. Encerrando worker.", gpu_id)
            break
        except Exception as e:
            log_warning(f"Erro inesperado na fila: {e}", gpu_id)
            break
        
        # 1. Carregar o professor treinado da Fase 1
        fase1_path = OUTPUT_FOLDER / "checkpoints_fase1" / ds_name / f"{model_name}_best.pth"
        if not fase1_path.exists():
            log_warning(f"Pesos da Fase 1 para {model_name} não encontrados. Pulando...", gpu_id)
            task_queue.task_done()
            continue

        # Define os caminhos exatos onde os arquivos deveriam estar salvos
        save_dir = OUTPUT_FOLDER / "checkpoints_fase2" / ds_name
        save_dir.mkdir(parents=True, exist_ok=True)
        expected_model_path = save_dir / f"student_of_{model_name}_best.pth"
        expected_plot_path = save_dir / f"student_of_{model_name}_accuracy_plot.png"
        
        if expected_model_path.exists() and expected_plot_path.exists():
            log_info(f"[SKIP] O estudante para o professor {model_name} já foi destilado com sucesso. Pulando...", gpu_id)
            task_queue.task_done()
            continue # Vai para o próximo modelo da fila sem gastar tempo ou memória

        try:
            # Carrega dados
            if ds_name == "cifar100":
                train_loader, val_loader, _, classes = get_cifar100_dataloaders()
            else:
                train_loader, val_loader, _, classes = get_intel_scenes_dataloaders("intel_scenes")
                
            num_classes = len(classes)

            # Recria o modelo e aplica o mesmo ajuste da última camada feito na Fase 1
            teacher_model = model_fn(weights=weights)
            if hasattr(teacher_model, 'fc'):
                teacher_model.fc = nn.Linear(teacher_model.fc.in_features, num_classes)
            elif hasattr(teacher_model, 'classifier'):
                if isinstance(teacher_model.classifier, nn.Sequential):
                    teacher_model.classifier[-1] = nn.Linear(teacher_model.classifier[-1].in_features, num_classes)
                else:
                    teacher_model.classifier = nn.Linear(teacher_model.classifier.in_features, num_classes)
                    
            # Carrega os pesos finetunados da Fase 1
            teacher_model.load_state_dict(torch.load(fase1_path, map_location="cpu"))
            
            # 2. Separa o Encoder do Classificador
            teacher_encoder, classifier, feature_dim = split_teacher_encoder_classifier(teacher_model)
            
            # 3. Instancia o Estudante Post-GAP
            student_model = PostGAPStudent(teacher_feature_dim=feature_dim)
            
            # 4. Inicia Destilação
            save_dir = OUTPUT_FOLDER / "checkpoints_fase2" / ds_name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            log_info(f"Iniciando Destilação (Fase 2) para o professor {model_name} em {ds_name}", gpu_id)
            train_student_distillation(teacher_encoder, student_model, classifier, 
                                    train_loader, val_loader, model_name, save_dir, gpu_id)
                                    
            task_queue.task_done()

        except Exception as e:
            log_info(f"Erro ao processar {model_name}: {e}", gpu_id)
            # Dependendo da estratégia, pode dar task_done() ou reinserir na fila
            task_queue.task_done()

        finally:
            # Limpeza pesada de memória (Garbage Collector) mesmo se houver erro
            if 'train_loader' in locals(): del train_loader
            if 'val_loader' in locals(): del val_loader
            if 'teacher_encoder' in locals(): del teacher_encoder
            if 'student_model' in locals(): del student_model
            if 'classifier' in locals(): del classifier
            gc.collect()
            torch.cuda.empty_cache()




def phase2_worker(gpu_id, task_queue):
    """Função isolada chamada por cada processo de GPU para a Destilação."""
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_info(f"Worker da Fase 2 iniciado na GPU {gpu_id}", gpu_id)
    
    # Chama o pipeline de destilação consumindo a fila
    run_phase2_pipeline(gpu_id=str(gpu_id), task_queue=task_queue)


def paralel_phase2_pipeline():
    # 'spawn' é obrigatório para PyTorch com CUDA
    mp.set_start_method('spawn', force=True)
    
    # Cria a fila e popula com as tarefas
    task_queue = mp.JoinableQueue()
    DATASETS = ["cifar100", "intel_scenes"]
    
    for ds_name in DATASETS:
        for model_name, (model_fn, weights) in MODELS.items():
            task_queue.put((ds_name, model_name, model_fn, weights))
    
    # Descobre quantas GPUs temos (ou força 2 se estiver usando NVML fixo)
    device_count = torch.cuda.device_count()
    if device_count == 0: device_count = 2 # Garante 2 workers se rodar mascarado
    
    processes = []
    log_info(f"Iniciando orquestração paralela da FASE 2 em {device_count} GPUs...", "PAI")
    
    for gpu_id in range(device_count):
        p = mp.Process(target=phase2_worker, args=(gpu_id, task_queue))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        
    log_info("Todos os estudantes foram destilados com sucesso!", "PAI")


if __name__ == "__main__":
    log_info("Iniciando Destilation", "PAI")
    
    # Cria a pasta de destino dos estudantes caso não exista
    (OUTPUT_FOLDER / "checkpoints_fase2").mkdir(parents=True, exist_ok=True)
    
    if PARALEL:
        paralel_phase2_pipeline()
    else:
        # Para rodar em modo sequencial: chamar o worker passando "0"
        log_warning("Modo sequencial não implementado na Fase 2. Ative PARALEL=True.", "PAI")


