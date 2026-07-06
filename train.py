import gc
import os
import sys
import shutil
import multiprocessing as mp
from pathlib import Path
import logging
from PIL import Image
import time
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pynvml
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    convnext_small, ConvNeXt_Small_Weights,
    convnext_base, ConvNeXt_Base_Weights,
    convnext_large, ConvNeXt_Large_Weights,
    vgg11_bn, VGG11_BN_Weights,
    vgg16_bn, VGG16_BN_Weights,
    vgg19_bn, VGG19_BN_Weights
)

# Configurações de Diretórios e Alvo
INPUT_FOLDER = Path("./data/input")
OUTPUT_FOLDER = Path("./data/output")
BATCH_SIZE = 512
EPOCHS = 100
NUM_WORKERS = 6
PERSISTENT_WORKERS = True
PARALEL = True
INTERVAL = 5 # Intervalo de épocas para checar temperatura da GPU
STD_GPU = 1 # GPU padrão caso NVML falhe
PREFETCH_FACTOR = 2
TARGET_SIZE = (224, 224)  

# Transformações usadas apenas no momento de CRIAR o cache em disco (se necessário)
data_transforms = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
])

# Transformações em Tempo de Execução
# Como as imagens salvas no cache já estão no tamanho correto (224x224), 
# aqui APENAS convertemos para Tensor e Normalizamos com a média/desvio do ImageNet.
runtime_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Configura o sistema de logs
log_dir = OUTPUT_FOLDER
log_dir.mkdir(parents=True, exist_ok=True)
log_filepath = log_dir / "treino.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (GPU %(message_ctx)s): %(message)s",
    handlers=[
        logging.FileHandler(log_filepath, encoding="utf-8"),
        logging.StreamHandler(sys.stdout) # Exibe no terminal do VS Code
    ]
)

def log_info(msg, gpu_id="PAI"):
    """Helper para injetar o ID da GPU no formato do log."""
    print(f"[GPU {gpu_id}] {msg}")
    logging.info(msg, extra={'message_ctx': str(gpu_id)})

def log_warning(msg, gpu_id="PAI"):
    print(f"[GPU {gpu_id}] {msg}")
    logging.warning(msg, extra={'message_ctx': str(gpu_id)})

DATASETS = ["cifar100", "intel_scenes"]

MODELS = {
    "ResNet-18": (resnet18, ResNet18_Weights.DEFAULT),
    "ResNet-50": (resnet50, ResNet50_Weights.DEFAULT),
    "ResNet-101": (resnet101, ResNet101_Weights.DEFAULT),
    "ConvNeXt-Tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),
    "ConvNeXt-Small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT),
    "ConvNeXt-Base": (convnext_base, ConvNeXt_Base_Weights.DEFAULT),
    "ConvNeXt-Large": (convnext_large, ConvNeXt_Large_Weights.DEFAULT),
    "VGG-11-BN": (vgg11_bn, VGG11_BN_Weights.DEFAULT),
    "VGG-16-BN": (vgg16_bn, VGG16_BN_Weights.DEFAULT),
    "VGG-19-BN": (vgg19_bn, VGG19_BN_Weights.DEFAULT),
}


# MODELS = {
#         # --- FAMÍLIA RESNET ---
#         "ResNet-18": (resnet18, ResNet18_Weights.DEFAULT),
#         "ResNet-34": (resnet34, ResNet34_Weights.DEFAULT),
#         "ResNet-50": (resnet50, ResNet50_Weights.DEFAULT),
#         "ResNet-101": (resnet101, ResNet101_Weights.DEFAULT),
#         "ResNet-152": (resnet152, ResNet152_Weights.DEFAULT),
        
#         # --- FAMÍLIA VGG ---
#         "VGG-11-BN": (vgg11_bn, VGG11_BN_Weights.DEFAULT),
#         "VGG-13-BN": (vgg13_bn, VGG13_BN_Weights.DEFAULT),
#         "VGG-16-BN": (vgg16_bn, VGG16_BN_Weights.DEFAULT),
#         "VGG-19-BN": (vgg19_bn, VGG19_BN_Weights.DEFAULT),
        
#         # --- FAMÍLIA CONVNEXT ---
#         "ConvNeXt-Atto": (convnext_atto, ConvNeXt_Atto_Weights.DEFAULT),
#         "ConvNeXt-Femto": (convnext_femto, ConvNeXt_Femto_Weights.DEFAULT),
#         "ConvNeXt-Nano": (convnext_nano, ConvNeXt_Nano_Weights.DEFAULT),
#         "ConvNeXt-Tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),
#         "ConvNeXt-Small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT),
#         "ConvNeXt-Base": (convnext_base, ConvNeXt_Base_Weights.DEFAULT),
#         "ConvNeXt-Large": (convnext_large, ConvNeXt_Large_Weights.DEFAULT),
        
#         # --- FAMÍLIA EFFICIENTNET V2 ---
#         "EfficientNet-V2-S": (efficientnet_v2_s, EfficientNet_V2_S_Weights.DEFAULT),
#         "EfficientNet-V2-M": (efficientnet_v2_m, EfficientNet_V2_M_Weights.DEFAULT),
#         "EfficientNet-V2-L": (efficientnet_v2_l, EfficientNet_V2_L_Weights.DEFAULT),
        
#         # --- FAMÍLIA VIT (VISION TRANSFORMER) ---
#         "ViT-Base-P16": (vit_b_16, ViT_B_16_Weights.DEFAULT),
#         "ViT-Base-P32": (vit_b_32, ViT_B_32_Weights.DEFAULT),
#         "ViT-Large-P16": (vit_l_16, ViT_L_16_Weights.DEFAULT),
#         "ViT-Large-P32": (vit_l_32, ViT_L_32_Weights.DEFAULT),
#         "ViT-Huge-P14": (vit_h_14, ViT_H_14_Weights.DEFAULT),
        
#         # --- FAMÍLIA SWIN TRANSFORMER ---
#         "Swin-Tiny": (swin_t, Swin_T_Weights.DEFAULT),
#         "Swin-Small": (swin_s, Swin_S_Weights.DEFAULT),
#         "Swin-Base": (swin_b, Swin_B_Weights.DEFAULT),
        
#         # --- FAMÍLIA MOBILENET ---
#         "MobileNet-V3-Small": (mobilenet_v3_small, MobileNet_V3_Small_Weights.DEFAULT),
#         "MobileNet-V3-Large": (mobilenet_v3_large, MobileNet_V3_Large_Weights.DEFAULT),
#     }


##############################################################################################
##############################################################################################
############################## Carregamento de modelos e dados ###############################
##############################################################################################
##############################################################################################


def download_pretrained_models(models=MODELS):
    """
    Baixa e faz o cache dos pesos pré-treinados dos classificadores.
    """
    log_info("=== Verificando/Baixando Modelos Professores ===")

    for name, (model_fn, weights) in models.items():
        log_info(f"Carregando pesos pré-treinados para {name}...")
        _ = model_fn(weights=weights)
    log_info("Todos os modelos foram verificados no cache!\n")
    


def check_dataset_cache(dataset_name, target_size=TARGET_SIZE):
    """
    Verifica se existem imagens salvas no output, se as pastas de split existem
    e se uma amostragem delas possui o tamanho alvo de 224x224.
    """
    dataset_out_dir = OUTPUT_FOLDER / dataset_name
    splits = ['train', 'val', 'test']
    
    if not dataset_out_dir.exists():
        return False
        
    for split in splits:
        split_dir = dataset_out_dir / split
        if not split_dir.exists():
            return False
            
        # Coleta arquivos de imagem jpg/png na pasta do split
        images = list(split_dir.glob("**/*.jpg")) + list(split_dir.glob("**/*.png"))
        if len(images) == 0:
            return False
            
        # Valida uma amostragem de 5 imagens para garantir que a resolução está correta
        for img_path in images[:5]:
            try:
                with Image.open(img_path) as img:
                    if img.size != target_size:
                        log_warning(f"[Cache Inválido] Imagem {img_path.name} possui tamanho {img.size} em vez de {target_size}.")
                        return False
            except Exception:
                return False
                
    return True


def get_cifar100_dataloaders():
    dataset_name = "cifar100"
    dataset_out_dir = OUTPUT_FOLDER / dataset_name
    
    # SE EXISTE O OUTPUT E ELE SEGUE O PADRÃO DESEJADO
    if check_dataset_cache(dataset_name, TARGET_SIZE):
        log_info(f"=== [CACHE HIT] Carregando CIFAR-100 pré-processado do disco ===")
        train_dataset = ImageFolder(root=dataset_out_dir / "train", transform=runtime_transforms)
        val_dataset = ImageFolder(root=dataset_out_dir / "val", transform=runtime_transforms)
        test_dataset = ImageFolder(root=dataset_out_dir / "test", transform=runtime_transforms)
        classes = train_dataset.classes
    
    # SE NÃO EXISTE OU ESTÁ INCORRETO: Processa, Salva no disco e depois carrega
    else:
        log_info(f"=== [CACHE MISS] Processando e Salvando CIFAR-100 no disco ===")
        if dataset_out_dir.exists():
            shutil.rmtree(dataset_out_dir)
            
        cifar_path = INPUT_FOLDER / "cifar100_raw"
        raw_train = torchvision.datasets.CIFAR100(root=cifar_path, train=True, download=True)
        raw_test = torchvision.datasets.CIFAR100(root=cifar_path, train=False, download=True)
        
        # Split estratificado de Validação (80/20)
        targets = raw_train.targets
        indices = list(range(len(raw_train)))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=targets, random_state=42)
        
        splits = {
            'train': Subset(raw_train, train_idx),
            'val': Subset(raw_train, val_idx),
            'test': raw_test
        }
        
        # Iterando e salvando fisicamente no formato estruturado para ImageFolder
        for split_name, dataset in splits.items():
            log_info(f"Salvando partição '{split_name}' do CIFAR-100...")
            for idx in tqdm(range(len(dataset))):
                img, target_id = dataset[idx]
                class_name = raw_train.classes[target_id].replace("/", "_")
                
                class_dir = dataset_out_dir / split_name / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Redimensiona na CPU uma única vez usando interpolação Bilinear (upsampling)
                resized_img = img.resize(TARGET_SIZE, Image.BILINEAR)
                resized_img.save(class_dir / f"{idx}.jpg")
                
        # Agora que está salvo, instanciamos via ImageFolder estruturado
        train_dataset = ImageFolder(root=dataset_out_dir / "train", transform=runtime_transforms)
        val_dataset = ImageFolder(root=dataset_out_dir / "val", transform=runtime_transforms)
        test_dataset = ImageFolder(root=dataset_out_dir / "test", transform=runtime_transforms)
        classes = train_dataset.classes

    # Geração definitiva dos loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS, prefetch_factor=PREFETCH_FACTOR)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS, prefetch_factor=PREFETCH_FACTOR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS, prefetch_factor=PREFETCH_FACTOR)

    log_info(f"CIFAR-100 pronto | Treino: {len(train_dataset)} | Val: {len(val_dataset)} | Teste: {len(test_dataset)}\n")
    return train_loader, val_loader, test_loader, classes


def get_intel_scenes_dataloaders(dataset_dirname="intel_scenes"):
    dataset_name = "intel_scenes"
    dataset_out_dir = OUTPUT_FOLDER / dataset_name
    
    # SE EXISTE O OUTPUT E ELE SEGUE O PADRÃO DESEJADO
    if check_dataset_cache(dataset_name, TARGET_SIZE):
        log_info(f"=== [CACHE HIT] Carregando Intel Scenes pré-processado do disco ===")
        train_dataset = ImageFolder(root=dataset_out_dir / "train", transform=runtime_transforms)
        val_dataset = ImageFolder(root=dataset_out_dir / "val", transform=runtime_transforms)
        test_dataset = ImageFolder(root=dataset_out_dir / "test", transform=runtime_transforms)
        classes = train_dataset.classes
        
    # SE NÃO EXISTE OU ESTÁ INCORRETO: Processa, Salva no disco e depois carrega
    else:
        log_info(f"=== [CACHE MISS] Processando e Salvando Intel Scenes no disco ===")
        if dataset_out_dir.exists():
            shutil.rmtree(dataset_out_dir)
            
        intel_path = INPUT_FOLDER / dataset_dirname
        train_src = intel_path / "seg_train" 
        test_src = intel_path / "seg_test" 
        if not train_src.exists(): train_src = intel_path / "seg_train"
        if not test_src.exists(): test_src = intel_path / "seg_test"
        
        if not train_src.exists():
            raise FileNotFoundError(f"Não foi possível encontrar a pasta de treino do Intel Scenes em {intel_path}")
            
        raw_train_folder = ImageFolder(root=train_src)
        raw_test_folder = ImageFolder(root=test_src)
        
        # Split estratificado para criar a validação a partir do treino (15% val)
        targets = raw_train_folder.targets
        indices = list(range(len(raw_train_folder)))
        train_idx, val_idx = train_test_split(indices, test_size=0.15, stratify=targets, random_state=42)
        
        splits = {
            'train': Subset(raw_train_folder, train_idx),
            'val': Subset(raw_train_folder, val_idx),
            'test': raw_test_folder
        }
        
        for split_name, subset in splits.items():
            log_info(f"Salvando partição '{split_name}' do Intel Scenes...")
            for idx in tqdm(range(len(subset))):
                # No Subset/ImageFolder, img é uma imagem PIL e target_id é o id numérico
                img, target_id = subset[idx]
                class_name = raw_train_folder.classes[target_id]
                
                # Coleta o nome original do arquivo se possível para evitar colisões
                img_name = f"{idx}.jpg"
                if hasattr(subset, 'dataset') and isinstance(subset, Subset):
                    actual_idx = subset.indices[idx]
                    img_name = Path(subset.dataset.samples[actual_idx][0]).name
                elif isinstance(subset, ImageFolder):
                    img_name = Path(subset.samples[idx][0]).name
                
                class_dir = dataset_out_dir / split_name / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Redimensiona na CPU uma única vez usando interpolação Lanczos (downsampling de alta qualidade)
                resized_img = img.resize(TARGET_SIZE, Image.LANCZOS)
                resized_img.save(class_dir / img_name)
                
        train_dataset = ImageFolder(root=dataset_out_dir / "train", transform=runtime_transforms)
        val_dataset = ImageFolder(root=dataset_out_dir / "val", transform=runtime_transforms)
        test_dataset = ImageFolder(root=dataset_out_dir / "test", transform=runtime_transforms)
        classes = train_dataset.classes

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS, prefetch_factor=PREFETCH_FACTOR)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS, prefetch_factor=PREFETCH_FACTOR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS, prefetch_factor=PREFETCH_FACTOR)

    log_info(f"Intel Scenes pronto | Treino: {len(train_dataset)} | Val: {len(val_dataset)} | Teste: {len(test_dataset)}\n")
    return train_loader, val_loader, test_loader, classes

    
##############################################################################################
##############################################################################################
################################### Funções de treinamento ###################################
##############################################################################################
##############################################################################################


def train_teacher(model_fn, weights, train_loader, val_loader, num_classes, model_name, save_dir, gpu_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_info(f"=== Iniciando Treino: {model_name} ===", gpu_id=gpu_id)
    
    # 1. Instancia o modelo e congela o encoder
    model = model_fn(weights=weights)
    
    # Identifica e substitui a última camada (o nome varia entre arquiteturas)
    if hasattr(model, 'fc'): # ResNet
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif hasattr(model, 'classifier'): # VGG/ConvNeXt
        # VGG tem classifier[6], ConvNeXt tem classifier[2]
        if isinstance(model.classifier, nn.Sequential):
            num_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_features, num_classes)
        else:
            num_features = model.classifier.in_features
            model.classifier = nn.Linear(num_features, num_classes)

    # Congela tudo
    for param in model.parameters():
        param.requires_grad = False
    
    # Descongela apenas a nova "boca"
    if hasattr(model, 'fc'):
        for param in model.fc.parameters(): param.requires_grad = True
    else:
        for param in model.classifier.parameters(): param.requires_grad = True

    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=1e-3)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Histórico para métricas
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    # Loop de épocas
    epochs = EPOCHS 
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch + 1) % INTERVAL == 0:
            check_gpu_thermal_safety(gpu_id)
        
        # Validação
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.inference_mode(): # inference_mode é ainda mais rápido que no_grad()
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # Calcula acertos direto na GPU
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        # acc = correct / total # acc é obtido como float
        acc = correct / len(val_loader.dataset)

        history['train_loss'].append(running_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(acc)
        
        log_info(f"Época {epoch+1}/{epochs} finalizada |  Modelo: {model_name} | Val Acc: {acc:.4f}", gpu_id)
        
        # Salva o melhor checkpoint
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_dir / f"{model_name}_best.pth")

    # 2. Gerar Gráfico de Diagnóstico
    plot_metrics(history, model_name, save_dir)
    return model

def plot_metrics(history, model_name, save_dir):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend(); plt.title(f"{model_name} Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Acc', color='orange')
    plt.legend(); plt.title(f"{model_name} Accuracy")
    
    plt.savefig(save_dir / f"{model_name}_performance.png")
    plt.close()
    log_info(f"Gráficos salvos em {save_dir}")


##############################################################################################
##############################################################################################
############################## Funções de controle de execução ###############################
##############################################################################################
##############################################################################################

def check_gpu_thermal_safety(gpu_id, max_temp=81, cool_temp=78):
    """
    Verifica a temperatura da GPU usando NVML. 
    Se passar de max_temp, pausa o treino até baixar para cool_temp.
    """
    try:
        gpu_id = int(gpu_id)  # Certifica que é inteiro para NVML
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        
        while True:
            # Captura a temperatura em Celsius
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            if temp >= max_temp: 
                sleep_time = (temp-cool_temp)  
                log_warning(f"ALERTA DE TEMPERATURA! GPU atingiu {temp}°C (Limite: {max_temp}°C). Pausando por {sleep_time}s para resfriamento...", gpu_id)
                pynvml.nvmlShutdown() # Libera o handle temporariamente
                time.sleep(sleep_time)
                pynvml.nvmlInit() # Reinicializa para a próxima checagem
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            else:
                # if temp > cool_temp + 1:
                log_info(f"Temperatura estável: {temp}°C", gpu_id)
                break # Temperatura está segura, pode continuar o treino
                
        pynvml.nvmlShutdown()
    except Exception as e:
        # Se falhar por algum motivo de driver, não trava o treino, apenas avisa
        logging.warning(f"Não foi possível ler a temperatura da GPU {gpu_id}: {e}", extra={'message_ctx': str(gpu_id)})


def run_full_pipeline(gpu_id, task_queue): 
    while True: 
        try:
            # Espera até 5 segundos. Se não vier nada, a fila realmente acabou.
            ds_name, model_name, model_fn, weights = task_queue.get(timeout=5)
        except mp.queues.Empty:
            log_info("Fila de tarefas vazia. Encerrando worker.", gpu_id)
            break
        except Exception as e:
            log_warning(f"Erro inesperado na fila: {e}", gpu_id)
            break
            
        log_info(f"Processando {model_name} no dataset {ds_name}", gpu_id)
        
        # Carrega os loaders (a lógica de cache continua funcionando igual)
        if ds_name == "cifar100":
            train_loader, val_loader, _, classes = get_cifar100_dataloaders()
        else:
            train_loader, val_loader, _, classes = get_intel_scenes_dataloaders("intel_scenes")
            
        save_dir = OUTPUT_FOLDER / "checkpoints_fase1" / ds_name
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_dir / f"{model_name}_best.pth"
        
        # Verifica se já existe antes de treinar
        if checkpoint_path.exists():
            log_warning(f"[SKIP] {model_name} já treinado.", gpu_id)
            task_queue.task_done()
            continue
            
        train_teacher(model_fn, weights, train_loader, val_loader, len(classes), model_name, save_dir, gpu_id)

        # Deletamos as referências dos dataloaders antigos
        del train_loader
        del val_loader
        
        # Forçamos o Python a destruir os processos zumbis da CPU na mesma hora
        gc.collect()

        task_queue.task_done()


def worker(gpu_id, task_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_info(f"Processo iniciado na GPU {gpu_id}", gpu_id)
    
    # Chama o pipeline que agora consome a fila
    run_full_pipeline(gpu_id=str(gpu_id), task_queue=task_queue)


def paralel_pipeline():
    mp.set_start_method('spawn', force=True)
    
    # 1. Cria a fila e popula com as tarefas (Dataset + Modelo)
    task_queue = mp.JoinableQueue()
    for ds_name in DATASETS:
        for model_name, (model_fn, weights) in MODELS.items():
            task_queue.put((ds_name, model_name, model_fn, weights))
    
    # 2. Sobe os processos passando a mesma fila para todos
    device_count = torch.cuda.device_count()
    processes = []
    for gpu_id in range(device_count):
        p = mp.Process(target=worker, args=(gpu_id, task_queue))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


def sequential_pipeline():
    """
    Executa o pipeline completo de forma tradicional e sequencial
    utilizando a GPU padrão configurada (ou CPU caso CUDA não esteja disponível).
    """
    # Define a GPU padrão do sistema (geralmente indexada como '0' se não houver isolamento)
    gpu_id = "0" if torch.cuda.is_available() else "cpu"
    log_info("Iniciando pipeline em modo sequencial...", gpu_id)
    
    # Criamos a lista manual de tarefas da mesma forma que a fila faria
    DATASETS = ["cifar100", "intel_scenes"]
    
    for ds_name in DATASETS:
        log_info(f"==================== INICIANDO PIPELINE SEQUENCIAL PARA: {ds_name.upper()} ====================", gpu_id)
        
        # Carrega os loaders sob demanda (com verificação interna de CACHE HIT / MISS)
        if ds_name == "cifar100":
            train_loader, val_loader, _, classes = get_cifar100_dataloaders()
        else:
            try:
                train_loader, val_loader, _, classes = get_intel_scenes_dataloaders("intel_scenes")
            except Exception as e:
                log_warning(f"[Erro Intel Scenes]: Não foi possível rodar este dataset. Mensagem: {e}", gpu_id)
                continue # Pula para o próximo dataset se houver erro crítico na carga
                
        num_classes = len(classes)
        save_dir = OUTPUT_FOLDER / "checkpoints_fase1" / ds_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Itera sobre todos os modelos definidos no dicionário GLOBAL
        for model_name, (model_fn, weights) in MODELS.items():
            checkpoint_path = save_dir / f"{model_name}_best.pth"
            
            # Se o modelo já foi treinado em execuções anteriores, pula
            if checkpoint_path.exists():
                log_warning(f"[SKIP] {model_name} já treinado para {ds_name}.", gpu_id)
                continue
                
            log_info(f"[SEQUENCIAL] Iniciando treino de {model_name} para {ds_name}", gpu_id)
            
            # Treina o professor na instância atual de forma bloqueante (um após o outro)
            train_teacher(
                model_fn=model_fn,
                weights=weights,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=num_classes,
                model_name=model_name,
                save_dir=save_dir,
                gpu_id=gpu_id
            )
            
            # Coleta de lixo explícita para evitar vazamento de memória RAM/VRAM 
            # entre a troca de arquiteturas pesadas (ex: VGG para ConvNeXt)
            torch.cuda.empty_cache()
            
    log_info("Pipeline sequencial concluído com sucesso!", gpu_id)

##############################################################################################
##############################################################################################
################################### Execução em pipeline #####################################
##############################################################################################
##############################################################################################


if __name__ == "__main__":
    # 1. Garante a criação física das pastas essenciais antes de tudo
    INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    (OUTPUT_FOLDER / "checkpoints_fase1").mkdir(parents=True, exist_ok=True)
    (OUTPUT_FOLDER / "checkpoints_fase2").mkdir(parents=True, exist_ok=True)
    
    # 2. Verifica/Baixa os pesos originais dos professores no processo pai
    # (O PyTorch gerencia o cache de downloads de forma segura)
    download_pretrained_models()
    
    # 3. Decisão de execução do Pipeline
    if PARALEL:
        # Chama a função que agora cria a Queue e gerencia as GPUs dinamicamente
        paralel_pipeline()
    else:
        log_info("Executando em modo sequencial padrão...", "PAI")
        # No modo sequencial (uma única GPU), ele roda a lógica antiga direta
        sequential_pipeline()

