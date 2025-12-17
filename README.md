# PyTorch GPU Profiling: buenas y malas prácticas

Este repositorio muestra **cómo perfilar código en PyTorch** y compara
implementaciones **ineficientes vs eficientes** cuando se trabaja con GPU.

El objetivo es entender:
- Por qué **mezclar NumPy y GPU es mala idea**
- El impacto de **float64 vs float32**
- Cómo evitar **copias CPU ↔ GPU**
- Cómo usar correctamente `torch.autograd.profiler`

---

## 🚀 Qué se aprende

✔ Uso de `torch.autograd.profiler`  
✔ Etiquetado de secciones con `record_function`  
✔ Coste oculto de `.cpu()`, `.numpy()` y `.item()`  
✔ Diferencias entre `float64` y `float32` en GPU  
✔ Uso correcto de `nonzero` en PyTorch  
✔ Por qué los tensores grandes pueden causar **CUDA out of memory**

---

🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
