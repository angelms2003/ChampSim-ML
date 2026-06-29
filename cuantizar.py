import torch
import torch.nn as nn
from torch.quantization import float_qparams_weight_only_qconfig
import os
import io

origin_directory = "/scratch/nas/3/angelm/vulkan-sim/models-mini-lstm-only-miss-softmax-lookahead0/"
destination_directory = "/scratch/nas/3/angelm/vulkan-sim/models-quant-mini-lstm-only-miss-softmax-lookahead0/"
    
for file in os.listdir(origin_directory):
    filename = os.fsdecode(file)
    print(f"Cuantizando {os.path.join(origin_directory, filename)}")

    # 1. Cuantizar (igual que antes)
    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.LSTM, nn.Linear},
        dtype=torch.qint8
    )
    quantized.page_delta_embedding.qconfig  = float_qparams_weight_only_qconfig
    quantized.block_offset_embedding.qconfig = float_qparams_weight_only_qconfig

    # 2. Convertir a TorchScript (siempre después de cuantizar, nunca antes)
    scripted = torch.jit.script(quantized)

    # 3. Guardar — este archivo es todo lo que necesita C++
    scripted.save(os.path.join(destination_directory, filename))

    print(f"Modelo cuantizado guardado en {os.path.join(destination_directory, filename)}")

