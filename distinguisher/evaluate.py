from guidance import models
from distinguisher.models import AggressiveSimple, AggressiveReasoning, SimpleGPT, SimpleDistinguisher, SimpleInstructDistinguisher, ReasoningDistinguisher, ReasoningGPT
from mutators import DocumentMutator, SentenceMutator, SpanMutator, WordMutator
from watermarkers import SemStampWatermarker, UMDWatermarker
from oracles.guidance.relative import RelativeOracle

from attack import Attack
import pandas as pd
import os
import datasets
import hydra
from utils import get_prompt_or_output, get_watermarked_text, get_nth_successful_perturbation
from dotenv import load_dotenv, find_dotenv



INITIAL_PROMPTS_FILE = './distinguisher/entropy_prompts.csv'
WATERMARKER = SemStampWatermarker
WATERMARKED_RESPONSES_FILE = lambda x : f'./distinguisher/watermarked_responses_{WATERMARKER.__name__}_{x}.csv'
ORACLE = RelativeOracle
ORACLE_LLM = "/data2/.shared_models/llama.cpp_models/Meta-Llama-3-8B-Instruct-q8_0.gguf"

RERUN_WATERMARKER = False
RERUN_MUTATOR = RERUN_WATERMARKER or False
NUM_STEPS = 500


class AttackParser():
    def __init__(self, file):
        self.file = file
        self.response = get_watermarked_text(self.file)
    
    def get_response(self):
        return self.response
    
    def get_nth(self, n):
        text = get_nth_successful_perturbation(self.file, n)
        # clean up the text
        text = text.replace("Revised Text:\n\n", "")
        return text
        
    
def get_file(entropy, output_num, attack_id):
    csv_file_directory = f"results/stationary_distribution/robustness_analysis/entropy_{entropy}/"
    first_perturbed_csv_filename = f"output_{output_num}/corpuses/attack_{attack_id}.csv"
    csv_file_path = os.path.join(csv_file_directory, first_perturbed_csv_filename)
    return csv_file_path

model_id = "TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ"

# Load the model
llm = models.Transformers(
    model_id, 
    echo=False,
    cache_dir="/data2/.shared_models/", 
    device_map='auto',
)

# load_dotenv(find_dotenv())
# chatgpt = models.OpenAI("gpt-4o-mini")

distinguisher_persona = \
"""
You are an expert in analyzing the similarity of responses.
You specialize in detecting whether one response has been derived from another by carefully analyzing the content and structure of the text.
Your expertise in linguistic analysis allows you to distinguish which responses are the most closely related.
Your goal is to provide a clear, concise, and accurate assessment of the provided instructions.
"""

params_list = [
      # {"mutator": DocumentMutator, "type": "document", "distinguisher": AggressiveSimple},
      # {"mutator": DocumentMutator, "type": "document", "distinguisher": AggressiveReasoning},
      # {"mutator": DocumentMutator, "type": "document", "distinguisher": ReasoningGPT},
      # {"mutator": DocumentMutator, "type": "document", "distinguisher": ReasoningDistinguisher},
      # {"mutator": DocumentMutator, "type": "document", "distinguisher": SimpleGPT},
      # {"mutator": DocumentMutator, "type": "document", "distinguisher": SimpleInstructDistinguisher},
      # {"mutator": DocumentMutator, "type": "document", "distinguisher": SimpleDistinguisher},
      
			# {"mutator": SentenceMutator, "type": "sentence", "distinguisher": AggressiveSimple},
      # {"mutator": SentenceMutator, "type": "sentence", "distinguisher": AggressiveReasoning},
      {"mutator": SentenceMutator, "type": "sentence", "distinguisher": ReasoningGPT},
      # {"mutator": SentenceMutator, "type": "sentence", "distinguisher": ReasoningDistinguisher},
      # {"mutator": SentenceMutator, "type": "sentence", "distinguisher": SimpleGPT},
      # {"mutator": SentenceMutator, "type": "sentence", "distinguisher": SimpleInstructDistinguisher},
      # {"mutator": SentenceMutator, "type": "sentence", "distinguisher": SimpleDistinguisher},
      
			# {"mutator": SpanMutator, "type": "span", "distinguisher": AggressiveSimple},
      # {"mutator": SpanMutator, "type": "span", "distinguisher": AggressiveReasoning},
      # {"mutator": SpanMutator, "type": "span", "distinguisher": ReasoningGPT},
      # {"mutator": SpanMutator, "type": "span", "distinguisher": ReasoningDistinguisher},
      # {"mutator": SpanMutator, "type": "span", "distinguisher": SimpleGPT},
      # {"mutator": SpanMutator, "type": "span", "distinguisher": SimpleInstructDistinguisher},
      # {"mutator": SpanMutator, "type": "span", "distinguisher": SimpleDistinguisher},
      
			# {"mutator": WordMutator, "type": "word", "distinguisher": AggressiveSimple},
      # {"mutator": WordMutator, "type": "word", "distinguisher": AggressiveReasoning},
      # {"mutator": WordMutator, "type": "word", "distinguisher": ReasoningGPT},
      # {"mutator": WordMutator, "type": "word", "distinguisher": ReasoningDistinguisher},
      # {"mutator": WordMutator, "type": "word", "distinguisher": SimpleGPT},
      # {"mutator": WordMutator, "type": "word", "distinguisher": SimpleInstructDistinguisher},
      # {"mutator": WordMutator, "type": "word", "distinguisher": SimpleDistinguisher},
                  ]

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
  if RERUN_WATERMARKER:
      print("Running Watermarker: ")
      watermarker = WATERMARKER(cfg, only_detect=False)
      prompts_df = pd.read_csv(INITIAL_PROMPTS_FILE)
      texts_a = []
      texts_b = []
      for entropy in range(1,11):
          p = prompts_df.loc[prompts_df['entropy'] == entropy].iloc[0]["prompt"]
          a = watermarker.generate(p)
          b = watermarker.generate(p)
          texts_a.append({"entropy": entropy, "prompt": p, "response": a})
          texts_b.append({"entropy": entropy, "prompt": p, "response": b})
          print(f"Watermarked entropy level {entropy}: ")
          print(f"\tPrompt: {p} ")
          print(f"\tResponse_A: {a} ")
          print(f"\tResponse_B: {b} ")
          
          df_a = pd.DataFrame(texts_a)
          df_a.to_csv(WATERMARKED_RESPONSES_FILE("a"))  
          
          df_b = pd.DataFrame(texts_b)
          df_b.to_csv(WATERMARKED_RESPONSES_FILE("b"))  

	# entropy: 1 (least restrictive) to 10 (most restrictive)
	# mutator: DocumentMutator, SentenceMutator, SpanMutator, WordMutator
	# distinguisher: Aggressive (AggressiveSimple, AggressiveReasoning)
	#                Reasoning  (ReasoningGPT, ReasoningDistinguisher)
	#                Simple     (SimpleGPT, SimpleInstructDistinguisher, SimpleDistinguisher)
  #params_list = []

  for params in params_list:
      Mutator = params["mutator"]
      Distinguisher = params["distinguisher"]
                
      for entropy in range(1,11):
          mutations_path = lambda x : f"./distinguisher/mutations/{Mutator.__name__}_entropy-{entropy}_{x}.csv"
          
          if RERUN_MUTATOR:
              oracle_llm = models.LlamaCpp(
                model=ORACLE_LLM,
                echo=False,
                n_gpu_layers=-1,
                n_ctx=2048
							)
              oracle = ORACLE(oracle_llm)
              #mutator = Mutator()
              cfg.mutator_args.type = params["type"]
              cfg.watermark_args.name = WATERMARKER.__name__.lower()
              attack = Attack(cfg, param_oracle=oracle)
              df_a = pd.read_csv(WATERMARKED_RESPONSES_FILE("a"))
              df_b = pd.read_csv(WATERMARKED_RESPONSES_FILE("b"))
              
              cfg.attack_args.max_steps = NUM_STEPS
              cfg.attack_args.log_csv_path = mutations_path("a")
              print("Running Mutator:")
              attack.attack(df_a.loc[df_a['entropy'] == entropy].iloc[0]["prompt"], df_a.loc[df_a['entropy'] == entropy].iloc[0]["response"])
              cfg.attack_args.log_csv_path = mutations_path("b")
              attack.attack(df_b.loc[df_b['entropy'] == entropy].iloc[0]["prompt"], df_b.loc[df_b['entropy'] == entropy].iloc[0]["response"])
              print("Finished mutations.")
              

          csv_path = f"./distinguisher/results/{Distinguisher.__name__}_{Mutator.__name__}_entropy-{entropy}.csv"
				
          response_A = AttackParser(mutations_path("a"))
          response_B = AttackParser(mutations_path("b"))
		  		# response_A = AttackParser(get_file(4, 1, "2_1"))
			  	# response_B = AttackParser(get_file(4, 2, "1_1"))
          #sd = Distinguisher(chatgpt, distinguisher_persona, response_A.get_response(), response_B.get_response())  
          sd2 = Distinguisher(llm, distinguisher_persona, response_A.get_response(), response_B.get_response())

          

          dataset = []
          for n in range(NUM_STEPS):
              dataset.append({
			  					"P": response_A.get_nth(n),
				  				"Num": n,
					  			"Origin": "A",
						  })
              dataset.append({
	  							"P": response_B.get_nth(n),
		  						"Num": n,
			  					"Origin": "B",
				  		})

          dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset))
          #dataset = sd.distinguish_majority(dataset, 5, "gpt_")
          dataset = sd2.distinguish_majority(dataset, 5, "llama_")
			  	# dataset = sd2.distinguish(dataset, "llama_")

          df = dataset.to_pandas()
          df["Response_A"] = response_A.get_response()
          df["Response_B"] = response_B.get_response() 
          df.to_csv(csv_path)

				  # ./impossibility-watermark> CUDA_VISIBLE_DEVICES=0 python -m distinguisher.evaluate




                  
if __name__ == "__main__":
    main()