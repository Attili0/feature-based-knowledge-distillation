import os
import csv
import gc
import torch
import torch.nn as nn
import multiprocessing as mp
from pathlib import Path
import pandas as pd 

# Importamos configurações da Fase 1
from train import (
    OUTPUT_FOLDER, MODELS, PARALEL,
    get_cifar100_dataloaders, get_intel_scenes_dataloaders,
    log_info, log_warning
)

# Importamos as funções que construímos na Fase 2 para não precisar reescrevê-las!
from distil import PostGAPStudent, split_teacher_encoder_classifier

# ---------------------------------------------------------
# AJUSTE AQUI: Coloque os estudantes treinados
STUDENTS_USED = ["mobilenet_v2", "shufflenet_v2"] 
# ---------------------------------------------------------

def evaluate_worker(gpu_id, task_queue, results_list):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    log_info(f"Worker de Avaliação iniciado na GPU {gpu_id}", gpu_id)
    
    # Cache para não recarregar o dataset a cada iteração
    loaders_cache = {}

    while not task_queue.empty():
        try:
            task = task_queue.get_nowait()
        except mp.queues.Empty:
            break
            
        ds_name, model_name, model_fn, weights, task_type, student_name = task

        # 1. Carrega o DataLoader de Teste sob demanda
        if ds_name not in loaders_cache:
            if ds_name == "cifar100":
                _, _, test_loader, classes = get_cifar100_dataloaders()
            else:
                _, _, test_loader, classes = get_intel_scenes_dataloaders("intel_scenes")
            loaders_cache[ds_name] = (test_loader, len(classes))
            
        test_loader, num_classes = loaders_cache[ds_name]

        try:
            # 2. Reconstrói o Professor (Sempre necessário, pois o estudante precisa do classificador dele)
            fase1_path = OUTPUT_FOLDER / "checkpoints_fase1" / ds_name / f"{model_name}_best.pth"
            # print(fase1_path)
            if not fase1_path.exists():
                log_warning(f"Professor {model_name} não encontrado em {ds_name}. Pulando...", gpu_id)
                task_queue.task_done()
                continue

            teacher_model = model_fn(weights=weights)
            
            # Adapta a última camada
            if hasattr(teacher_model, 'fc'):
                teacher_model.fc = nn.Linear(teacher_model.fc.in_features, num_classes)
            elif hasattr(teacher_model, 'classifier'):
                if isinstance(teacher_model.classifier, nn.Sequential):
                    teacher_model.classifier[-1] = nn.Linear(teacher_model.classifier[-1].in_features, num_classes)
                else:
                    teacher_model.classifier = nn.Linear(teacher_model.classifier.in_features, num_classes)
            
            # Carrega pesos
            teacher_model.load_state_dict(torch.load(fase1_path, map_location="cpu"))
            teacher_model = teacher_model.to(device).eval()

            requires_4d = "convnext" in model_name.lower()
            correct = 0

            # -------------------------------------------------------------
            # AVALIAÇÃO DO PROFESSOR ORIGINAL
            # -------------------------------------------------------------
            if task_type == "teacher":
                log_info(f"Testando PROFESSOR {model_name} no {ds_name}...", gpu_id)
                with torch.inference_mode():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        preds = teacher_model(inputs).argmax(dim=1)
                        correct += int((preds == labels).sum())
                        
                acc = correct / len(test_loader.dataset)
                results_list.append({
                    "Dataset": ds_name, "Teacher": model_name, 
                    "Student_Arch": "-", "Role": "Teacher", "Test_Accuracy": round(acc, 4)
                })

            # -------------------------------------------------------------
            # AVALIAÇÃO DO ESTUDANTE DESTILADO
            # -------------------------------------------------------------
            elif task_type == "student":
                log_info(f"Testando ESTUDANTE {student_name} (Prof: {model_name}) no {ds_name}...", gpu_id)
                student_path = OUTPUT_FOLDER / "checkpoints_fase2" / student_name / ds_name / f"student_of_{model_name}_best.pth"
                # print(student_path)

                if not student_path.exists():
                    log_warning(f"Checkpoint de {student_name} (Prof: {model_name}) não encontrado. Pulando...", gpu_id)
                    task_queue.task_done()
                    continue
                
                # Desacopla o classificador do professor
                teacher_encoder, classifier, feature_dim = split_teacher_encoder_classifier(teacher_model)
                classifier = classifier.to(device).eval()

                del teacher_encoder
                torch.cuda.empty_cache()
                
                # Monta e carrega o estudante
                student_model = PostGAPStudent(teacher_feature_dim=feature_dim, student_name=student_name)
                student_model.load_state_dict(torch.load(student_path, map_location="cpu"))
                student_model = student_model.to(device).eval()
                
                with torch.inference_mode():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        features = student_model(inputs)
                        if requires_4d:
                            features = features.unsqueeze(2).unsqueeze(3)
                            
                        logits = classifier(features)
                        preds = logits.argmax(dim=1)
                        correct += int((preds == labels).sum())
                        
                acc = correct / len(test_loader.dataset)
                results_list.append({
                    "Dataset": ds_name, "Teacher": model_name, 
                    "Student_Arch": student_name, "Role": "Student", "Test_Accuracy": round(acc, 4)
                })
                
                del student_model, classifier

        except Exception as e:
            log_warning(f"Erro ao avaliar {task_type} {model_name} em {ds_name}: {e}", gpu_id)
            
        finally:
            if 'teacher_model' in locals(): del teacher_model
            gc.collect()
            torch.cuda.empty_cache()
            task_queue.task_done()


def run_evaluation_pipeline():
    mp.set_start_method('spawn', force=True)
    task_queue = mp.JoinableQueue()
    manager = mp.Manager()
    results_list = manager.list()
    
    DATASETS = ["cifar100", "intel_scenes"]
    
    # Preenche a fila de testes
    for ds_name in DATASETS:
        for model_name, (model_fn, weights) in MODELS.items():
            # 1. Adiciona a tarefa de testar o Professor original
            task_queue.put((ds_name, model_name, model_fn, weights, "teacher", None))
            
            # 2. Adiciona as tarefas de testar os Estudantes
            for std_name in STUDENTS_USED:
                task_queue.put((ds_name, model_name, model_fn, weights, "student", std_name))
                
    device_count = torch.cuda.device_count()
    if device_count == 0: device_count = 2 
    
    processes = []
    log_info(f"Iniciando Avaliação Final em {device_count} GPUs. Total de modelos: {task_queue.qsize()}", "PAI")
    
    for gpu_id in range(device_count):
        p = mp.Process(target=evaluate_worker, args=(gpu_id, task_queue, results_list))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    # --- SALVANDO OS RESULTADOS ---
    log_info("Consolidando resultados e gerando arquivo", "PAI")
    df = pd.DataFrame(list(results_list))
    
    # Ordena a tabela para ficar bem organizada por Dataset e Professor
    df = df.sort_values(by=["Dataset", "Teacher", "Role"], ascending=[True, True, False])
    
    excel_path = OUTPUT_FOLDER / "resultados_finais_teste.xlsx"
    df.to_excel(excel_path, index=False)
    
    log_info(f"Avaliação concluída com sucesso! Resultados salvos em: {excel_path}", "PAI")
    
    # Opcional: Imprime os 10 primeiros na tela para você já ter um gostinho
    print("\n--- AMOSTRA DOS RESULTADOS ---")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    run_evaluation_pipeline()