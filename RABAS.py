from collections import defaultdict
import json
from tqdm import tqdm

import time

from models import GemmaHF, Gemma, GemmaVLLM

class RABAS:
    def __init__(self, model_id: str = None, data_path: str = None, config_path: str = None, metrics: list = None, eval_data_path: str = None):
        self.model_id = model_id
        self.model = None
        self.data = None
        self.config = None
        self.prompts = {}
        self.metrics = None
        self.data_path = data_path
        self.config_path = config_path
        self.eval_data_path = eval_data_path
        self.metric_list = metrics

    def evaluate(self):
        # LOAD DATA
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # LOAD CONFIG (not used currently)
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # LOAD METRICS PROMPTS
        for metric in self.metric_list:
            with open(f"prompts/{metric}.json", 'r', encoding='utf-8') as f:
                self.prompts[metric] = json.load(f)

        # (not used currently)
        metric_config_path = "metrics.json"
        with open(metric_config_path, 'r', encoding='utf-8') as f:
            self.metrics_config = json.load(f)

        # LOAD MODEL
        #self.model = GemmaHF()
        self.model = GemmaVLLM()

        results = {}

        for metric in self.metric_list:
            results[metric] = self._evaluate_metric(metric)


        output_judge = []
        for i, item in enumerate(self.data):
            item['metrics'] = {}
            for metric in self.metric_list:
                item['metrics'][metric] = results[metric][i]

            output_judge.append(item)

        # Guardamos todo en un archivo JSON
        output_path = self.data_path.replace(".json", "_judge_output.json")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(output_judge, file, ensure_ascii=False, indent=4)

        print(f"[✅] Resultados guardados en {output_path}")
        return results
    
    def generate_prompts(self, metric):
        prompts = []

        for item in tqdm(self.data, desc="Generating prompts", total=len(self.data)):
            # add examples + input
            if metric == "context_precision":
                for context in item.get("retrieved_contexts", []):
                    prompt = self.get_prompt(metric, item, context)
                    prompts.append(prompt)
            else:
                prompt = self.get_prompt(metric, item)
                prompts.append(prompt)
            

        return prompts

    def generate_prompts_context_precision(self, metric, retrieved_contexts):
        prompts = []

        for item in retrieved_contexts:
            # add examples + input
            prompt = self.get_prompt(metric, item)
            prompts.append(prompt)

        return prompts
    
    def generate_responses(self, prompts):
        print(f"[ℹ️] Generating responses... {len(prompts)}")

        start_time = time.perf_counter()
        output = self.model.generate_responses(prompts)
        print(f"[✅] Responses generated.")        

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        # Formatear a HH:MM:SS
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        print(f"⏱️ Tiempo total: {formatted_time}")
            
        return output
    

    def get_prompt(self, metric, item, context=None):
        metric_examples = self.prompts[metric].get("examples", "")
        user_prompt = f"Here are some examples you have to follow for the formatting your answer:\n\n{metric_examples}\n\nNow provide your answer in JSON format for the following input:"
        
        
        if metric == "faithfulness":
            context_text = "\n".join(item.get("retrieved_contexts", []))
            user_prompt += (
                f"\n\nQuestion: {item['user_input']}\n"
                f"Context: {context_text}\n"
                f"Answer: {item['response']}\n"
            )
        elif metric == "context_recall":
            context_text = "\n".join(item.get("retrieved_contexts", []))
            user_prompt += (
                f"\n\nQuestion: {item['user_input']}\n"
                f"Context: {context_text}\n"
                f"Answer: {item['reference']}\n"
            )
        elif metric == "context_precision":
            user_prompt += (
                f"\n\nQuestion: {item['user_input']}\n"
                f"Context: {context}\n"
                f"Answer: {item['reference']}\n"
            )


        prompt = [
            {"role": "system", "content": self.prompts[metric]['instruction']},
            {"role": "user", "content": user_prompt}
        ]
            
        return prompt

    def _evaluate_metric(self, metric: str):
        # Placeholder for actual metric evaluation logic
        
        if metric == "faithfulness":
            # generate all prompts for faithfulness
            prompts = self.generate_prompts(metric)
            responses = self.generate_responses(prompts)
            return responses
        elif metric == "context_recall":
            # generate all prompts for context_recall
            prompts = self.generate_prompts(metric)
            responses = self.generate_responses(prompts)
            return responses
        
        elif metric == "context_precision":
            prompts = self.generate_prompts(metric)
            responses_context = self.generate_responses(prompts)
            # Agrupar cada 5 respuestas
            responses = [
                responses_context[i:i + 5]
                for i in range(0, len(responses_context), 5)
            ]

            return responses

    def get_results(self):

        #self.model = GemmaVLLM()

        parse_errors = ""
        with open(self.eval_data_path, 'r', encoding='utf-8') as f:
            evaluation_data = json.load(f)

        results = defaultdict(list)
        for i, item in tqdm(enumerate(evaluation_data), desc="Calculating metrics", total=len(evaluation_data)):
            #print(item)
            for metric in self.metric_list:
                result, parse_errors_ret = self.get_metric_result(item, metric, i)
                parse_errors += parse_errors_ret
                results[metric].append(result)

        # sacar la media de cada metrica y guardar en un archivo los resultados
        output_path = self.eval_data_path.replace(".json", "_metrics_results.txt")
        with open(f"{output_path}", "w", encoding="utf-8") as f:
            for metric in self.metric_list:
                metric_scores = results[metric]
                avg_score = sum(metric_scores) / len(metric_scores)
                print(f"[📊] Average {metric}: {avg_score}")
                f.write(f"Average {metric}: {avg_score}\n")
            if parse_errors:
                f.write(f"Parse errors found in items: \n{parse_errors}\n")
        
        print(f"[✅] Metric results saved in {output_path}")

    def get_metric_result(self, item, metric, index):
        parse_errors = ""
        if isinstance(item['metrics'][metric], str):
            try:
                parsed = json.loads(item['metrics'][metric])
            except json.JSONDecodeError as e:
                print(f"[WARN] No se pudo parsear JSON en {metric}, ({index}): {e}")
                parse_errors = (f"{metric}, ({index}): {e}\n")
                return (0.0, parse_errors)
                
        if metric == "faithfulness":
            json_result = item['metrics'][metric]
            
            # score = num-verdict-1 / len(statements)
            # comprobar si existe statements
            if 'statements' not in json_result or not json_result['statements']:
                print(f"[⚠️] No statements found for item {index}")
                if isinstance(json_result, list):
                    print(f"[✅] List found\n")
                    statements = json_result
                else:
                    statements = []
                    print(f"[❌] Empty statements\n")
                    parse_errors = (f"{metric}, ({index}): No statements found\n")
            else:
                statements = json_result['statements']

            num_verdict_1 = 0
            for statement in statements:
                if statement['verdict'] == 1:
                    num_verdict_1 += 1
            score = num_verdict_1 / len(statements) if len(statements) > 0 else 0
            return (score, parse_errors)
        
        elif metric == "context_precision":
            json_result = item['metrics'][metric]
            num_verdict_1 = 0
            for context in json_result:
                if isinstance(context,str):
                    try:
                        parsed = json.loads(context)
                        
                    except json.JSONDecodeError as e:
                        print(f"[WARN] No se pudo parsear JSON en {metric}, ({index}): {e}")
                        parse_errors = (f"{metric}, ({index}): {e}\n")
                        continue

                if context['verdict'] == 1:
                    num_verdict_1 += 1
            score = num_verdict_1 / len(json_result)
            return (score, parse_errors)

        elif metric == "context_recall":
            json_result = item['metrics'][metric]
            #print(json_result)
            # score = num-attributed-1 / len(classifications)
            # comprobar si existe classifications
            if 'classifications' not in json_result or not json_result['classifications']:
                print(f"[⚠️] No classifications found for item {index}")
                if isinstance(json_result, list):
                    print(f"[✅] List found")
                    classifications = json_result
                else:
                    classifications = []
                    print(f"[❌] Empty classifications\n")
                    parse_errors = (f"{metric}, ({index}): No classifications found\n")
            else:
                classifications = json_result['classifications']

            num_attributed_1 = 0
            for statement in classifications:
                if statement['attributed'] == 1:
                    num_attributed_1 += 1
            score = num_attributed_1 / len(classifications) if len(classifications) > 0 else 0
            return (score, parse_errors)