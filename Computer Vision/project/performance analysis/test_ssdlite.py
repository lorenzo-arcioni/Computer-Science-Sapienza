import os
import time
import csv
import psutil
import torch
import numpy as np
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from PIL import Image
import platform

model_name = "ssd320_mobilenet_v3"

def count_parameters(model):
    """Conta i parametri del modello"""
    return sum(p.numel() for p in model.parameters())

def get_model_size_mb(model):
    """Calcola la dimensione del modello in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def get_system_info():
    """Ottiene informazioni sul sistema"""
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    cpu_freq_ghz = cpu_freq.current / 1000 if cpu_freq else "N/A"
    ram_gb = psutil.virtual_memory().total / (1024**3)
    return cpu_count, cpu_freq_ghz, ram_gb

def create_dummy_images(num_images=20, size=(320, 320)):
    """Crea immagini dummy per i test - identiche per tutti i modelli"""
    # Seed fisso per garantire immagini identiche tra tutti i test
    np.random.seed(42)
    
    images = []
    for i in range(num_images):
        # Crea un'immagine RGB con seed specifico per ogni immagine
        np.random.seed(42 + i)  # Seed diverso per ogni immagine ma riproducibile
        img_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        # Converti in tensor e normalizza
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        images.append(img_tensor)
    return images

def estimate_gflops_torchvision(model, input_size=(320, 320)):
    """Stima GFLOPS per modelli torchvision usando fvcore se disponibile"""
    try:
        # Prova a usare fvcore per calcolo preciso (se installato)
        from fvcore.nn import FlopCountMode, flop_count
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
        flops, _ = flop_count(model, dummy_input, supported_ops=None)
        return flops / 1e9
    except ImportError:
        
        if 'mobilenet' in model_name:
            # Valore approssimativo per SSDLite320 (basato su paper e benchmarks)
            return 0.6  # circa 600M FLOPs
        elif 'ssd300_vgg16' in model_name:
            # Valore approssimativo per SSD300 VGG16
            return 31.0  # circa 31G FLOPs
        else:
            # Fallback: indica che il valore non è disponibile
            return -1.0
    except Exception as e:
        print(f"Warning: Could not calculate GFLOPs accurately: {e}")
        return -1.0

def test_model():
    
    print(f"Testing {model_name}...")
    
    # Carica il modello
    model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    model.eval()  # Modalità valutazione
    
    # Ottieni informazioni sul sistema
    cpu_count, cpu_freq, ram_total = get_system_info()
    
    # Crea immagini dummy per il test
    test_images = create_dummy_images(50, size=(320, 320))
    
    # Misura memoria iniziale
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calcola metriche del modello
    num_params = count_parameters(model)
    model_size_mb = get_model_size_mb(model)
    gflops = estimate_gflops_torchvision(model, input_size=(320, 320))
    
    # Test di velocità
    print("Warming up...")
    # Warm-up runs
    with torch.no_grad():
        for i in range(3):
            _ = model([test_images[0]])
    
    print("Running inference tests...")
    start_time = time.time()
    
    with torch.no_grad():
        for img_tensor in test_images:
            _ = model([img_tensor])
    
    end_time = time.time()
    
    # Misura memoria dopo inferenza
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = final_memory - initial_memory
    
    # Calcola metriche
    total_time = end_time - start_time
    avg_inference_time = total_time / len(test_images)
    fps = 1.0 / avg_inference_time
    
    # Risultati
    results = {
        'modello': model_name,
        'num_parametri': num_params,
        'dimensione_mb': round(model_size_mb, 2),
        'gflops': round(gflops, 2),
        'inference_speed_ms': round(avg_inference_time * 1000, 2),
        'fps': round(fps, 2),
        'memoria_utilizzata_mb': round(memory_used, 2),
        'immagini_testate': len(test_images),
        'cpu_count': cpu_count,
        'ram_totale_gb': round(ram_total, 2),
        'cpu_freq_ghz': round(cpu_freq, 2) if isinstance(cpu_freq, (int, float)) else cpu_freq
    }
    
    print(f"Results for {model_name}:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    # Salva risultati in CSV
    csv_file = 'results.csv'
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(results)
    
    print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    test_model()
