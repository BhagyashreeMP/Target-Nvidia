# Day 1 – Python 

## 1. What are OOP principles? How did you use them in an ML project?
**Answer:**
I use OOP to keep ML systems modular and reusable.  
Encapsulation → separate data loading, training, inference logic.  
Abstraction → expose simple APIs like train() or predict().  
Inheritance → base ModelTrainer class extended for CNN/LLM trainers.  
Polymorphism → same predict() interface for different models.  
This helps me swap models without changing the rest of the pipeline.

---

## 2. How would you structure a production Python AI project?
**Answer:**
I usually structure like:  
src/ → models/, training/, inference/, api/, utils/  
configs/ → yaml configs  
tests/ → unit tests  
scripts/ → train/run scripts  
This keeps training, serving, and utilities independent and easy to scale.

---

## 3. Difference between instance, classmethod, staticmethod?
**Answer:**
Instance → works with object state (predict()).  
Classmethod → works with class-level logic (load_from_checkpoint()).  
Staticmethod → utility/helper functions (metrics/calculations).  
In ML pipelines, predict is instance, loader is classmethod, helpers are static.

---

## 4. How does Python memory management work?
**Answer:**
Python uses reference counting and garbage collection.  
Objects are deleted when reference count becomes zero, and GC handles cyclic references.  
For ML, I manually delete large tensors and call torch.cuda.empty_cache() to avoid GPU memory leaks.

---

## 5. Why virtual environments?
**Answer:**
Different projects need different PyTorch/CUDA versions.  
Virtual environments isolate dependencies and prevent conflicts.  
I use venv/conda + requirements.txt for reproducibility.

---

## 6. How do you manage configs in production?
**Answer:**
I never hardcode values.  
I use YAML/ENV configs for model paths, batch size, keys.  
This allows changing settings without code changes and supports multiple environments.

---

## 7. Multithreading vs multiprocessing vs async?
**Answer:**
Multithreading → IO tasks but limited by GIL  
Multiprocessing → CPU heavy tasks  
Async → best for API calls and concurrent inference requests  
For FastAPI inference, I prefer async + batching.

---

## 8. Explain async/await with example.
**Answer:**
Async lets the server handle multiple requests without blocking.  
Example: while waiting for model inference or DB call, other requests are processed.  
This improves throughput for chatbots or RAG APIs.

---

## 9. What should you log in production inference?
**Answer:**
Request ID, latency, input size, model version, errors.  
Not full user data for privacy.  
Helps debugging slow or failed requests.

---

## 10. How do you handle failures or retries?
**Answer:**
Use try/except with retries and timeouts.  
For external APIs, I use exponential backoff.  
Never let one failure crash the whole pipeline.

---

## 11. What are decorators? Example in ML?
**Answer:**
Decorators wrap functions to add behavior.  
I use them for timing inference latency or retry logic.  
Example: @timer to log model prediction time.

---

## 12. Generators vs lists?
**Answer:**
Lists load everything into memory.  
Generators yield one item at a time.  
For large datasets or streaming logs, generators reduce memory usage.

---

## 13. Shallow vs deep copy?
**Answer:**
Shallow copy shares references, deep copy creates independent objects.  
In preprocessing, shallow copy can accidentally modify original data.  
So I use deep copy for safety.

---

## 14. How do you write modular training code?
**Answer:**
Separate classes: Dataset, Model, Trainer, Evaluator.  
Each has one responsibility.  
Makes testing and debugging easier.

---

## 15. How do you make code reusable?
**Answer:**
Convert common logic into packages/utils and install locally with pip.  
Avoid duplicate scripts.  
Follow DRY principle.

---

## 16. Why type hints?
**Answer:**
Improves readability and IDE support.  
Catches bugs early.  
Important when teams collaborate on large ML systems.

---

## 17. How would you create a CLI for training/inference?
**Answer:**
Use argparse/typer.  
Example: python main.py --mode train or --mode infer.  
Makes automation and CI easier.

---

## 18. If inference is slow, what do you check first?
**Answer:**
Batching, CPU/GPU usage, unnecessary preprocessing, blocking code, logging overhead.  
Then profile using torch profiler or Nsight.

---

## 19. Design classes for a RAG chatbot.
**Answer:**
DocumentLoader → load docs  
Embedder → create embeddings  
Retriever → search  
LLMService → generate answers  
APIService → serve requests  
This keeps each part independent.

---

## 20. How do you ensure code quality?
**Answer:**
Unit tests, linting, type checks, PR reviews, CI pipelines, documentation.  
For ML, I also track experiments for reproducibility.

---

# Basic Quick Questions

21. What is __init__? → constructor for initializing object state  
22. What is __name__ == "__main__"? → run file as script  
23. What is pip freeze? → export dependencies  
24. What is list comprehension? → concise list creation  
25. What is lambda? → small inline function  
26. What is context manager? → safe resource handling (with open())  
27. What is GIL? → limits true threading in Python  
28. What is requirements.txt? → dependency list  
29. What is virtualenv? → isolated environment  
30. Why clean code? → maintainability and team scalability  

---

Tip: In interviews, always answer with:
👉 concept + practical example + why it matters in production
