# any_rag
Speculative RAG (Rationale-Augmented Generation) using different API vendors

inspiration: https://replit.com/@MartinBowling/Speculative-RAG-with-Groq?v=1#main.py

### Groq test

```

rag_models:
  generalist:
      name: "llama-3.1-8b-instant"
      api: "groq"
      user_prompt: |        
        {user_prompt}
        
  specialist:
    - name: "mixtral-8x7b-32768"
      api: "groq"
    - name: "llama-3.1-70b-versatile"
      api: "groq"      

  evaluator:
      name: "mixtral-8x7b-32768"
      api: "groq"

  final:
      name: "mixtral-8x7b-32768"
      api: "groq"


rag_prompt:
  generalist: |
  ...
  ```


  ### Execute:

  ```
  python bot.py config\groq\doc.yaml
  ```


  ### Hugging face test

  ```
rag_models:
  generalist:
      name: "mistralai/Mistral-Nemo-Instruct-2407"
      api: "hugging_face"
      user_prompt: |        
        {user_prompt}
        
  specialist:
      name: "mistralai/Mistral-Nemo-Instruct-2407"
      api: "hugging_face"

  evaluator:
      name: "mistralai/Mistral-Nemo-Instruct-2407"
      api: "hugging_face"

  final:
      name: "mistralai/Mistral-Nemo-Instruct-2407"
      api: "hugging_face"

is_complex: true
num_of_drafts: 1

rag_prompt:
  generalist: |
  ...
  ```


  ### Execute:

  ```
  python bot.py config\hugging_face\test.yaml
  ```

### Mixed test
Groq + HF (HF as generalist)

```

rag_models:
  generalist:
      name: "mistralai/Mistral-Nemo-Instruct-2407"
      api: "hugging_face"
      user_prompt: |        
        {user_prompt}
        return a detailed description of updated  image prompt in <fused_image> tags.  
  specialist:
      name: "mixtral-8x7b-32768"
      api: "groq"

  evaluator:
      name: "mixtral-8x7b-32768"
      api: "groq"

  final:
      name: "mixtral-8x7b-32768"
      api: "groq"


rag_prompt:
  generalist: |
  ...
  ```


  ### Execute:

  ```
  python bot.py config\mixed\mixed_test.yaml
  ```

  Medium: https://medium.com/p/590fc51fa14e 
  <br>
  Podcast: https://www.youtube.com/watch?v=vtEwH2NGqtg
  <br>
  Previous art: [Any_COT](https://github.com/myaichat/any_cot)
