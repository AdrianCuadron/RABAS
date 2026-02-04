import json
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from vllm import LLM, SamplingParams

class GemmaVLLM:
    def __init__(self):
            
        self.model = LLM(model="google/gemma-3-27b-it", gpu_memory_utilization=0.85, tensor_parallel_size=2)
        self.params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=4096,
            repetition_penalty=1.0,
            seed=42,
        )

    def generate_responses(self, messages):
        try:
            outs = self.model.chat(messages, self.params, use_tqdm=True)
        except Exception as e:
            print(f"[ERROR] Error durante la inferencia: {e}")
            return [None for _ in messages]

        results = []
        for i, out in enumerate(outs):
            text = out.outputs[0].text.replace("```json", "").replace("```", "").strip()
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                print(f"[WARN] No se pudo parsear JSON en la respuesta {i}")
                parsed = text
            results.append(parsed)
        return results

    def format_json(self, text):
        out = self.model.chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that converts text into valid JSON. Output only the JSON without any additional text."
                },
                {
                    "role": "user",
                    "content": f"Convert the following text into valid JSON:\n{text}"
                }
            ],
            sampling_params=self.params
        )

        try:
            json_data = json.loads(out[0].outputs[0].text.strip().replace("```json", "").replace("```", "").strip())
            print(json_data)
            return json_data
        except json.JSONDecodeError:
            return None
        

class Gemma:
    def __init__(self):
        self.client = OpenAI(base_url="http://localhost:8004/v1", api_key="EMPTY")
        self.MODEL_NAME = "google/gemma-3-27b-it"

    def generate_responses(self, messages):
        responses = []
        for message in messages:
            response = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=message,
                max_tokens=1000,
                temperature=0.1,
                top_p=0.9,
                stream=False
            )
            responses.append(response.choices[0].message.content.replace("```json", "").replace("```", "").strip())
        return responses
    
class GemmaHF:
    def __init__(self):

        batch_size = 4

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side='left')
        tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto")

        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=1000, 
            do_sample=True,
            dtype="auto",
            model_kwargs={
                'device_map': 'auto',
                'batch_size':batch_size,
                "temperature": 0 
                }, 
            batch_size=batch_size,
             
        )

        self.model = HuggingFacePipeline(pipeline=pipe, batch_size=batch_size, model_kwargs={"temperature": 0})

    def generate_responses(self, messages):
        responses = self.model.batch(messages)

        output = []
    
        # Iteramos sobre cada item del dataset original
        for i, response in tqdm(enumerate(responses), desc="Generating output", total=len(responses)):

            try:
                # Extraemos el JSON del modelo
                json_text = (
                    response.split("Feedback:::")[1]
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )
                judge_data = json.loads(json_text)
            except Exception as e:
                print(f"[⚠️] JSON inválido en la respuesta {i}: {str(e)}")
                judge_data = {"error": "JSON Decode Error"}

            output.append(judge_data)

        return output