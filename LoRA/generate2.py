import torch
from peft import PeftModel
from transformers import pipeline, GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd
from utils.prompter import Prompter
from tqdm import tqdm


device = "cuda"
prompt_template = "C1_polyglot"
base_model = 'EleutherAI/polyglot-ko-5.8b'
lora_weights = './51216100.0001c1_polyglot'
prompter = Prompter(prompt_template)
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
        device_map={'':0}
    )
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

data = load_dataset("json", data_files="filtered_C1_prefixstrip.json")


i = 0 

pred = []
grade =[]
with torch.autocast("cuda"):

    for Essay_prompt, Essay in tqdm(zip(data['train']['Essay_prompt'][i:], data['train']['Essay'][i:])):
        prompt = prompter.generate_prompt(Essay_prompt=Essay_prompt, Essay=Essay)  
        
        ans = pipe(
            prompt,
            do_sample=True,
            max_length=2048,  
            temperature=0.1,
            top_p=0.75,
            top_k=40, 
            return_full_text=False,
            eos_token_id=2,
        )
  
        response= ans[0]["generated_text"].strip()
        grade = ans[0]["generated_text"].strip()


        pred.append(response)
        grade.append(grade)
        
        response2 = pd.Series(pred)
        grade2 = pd.Series(grade)

        new_df = pd.concat([pd.Series(data['train']['Essay_prompt']), pd.Series(data['train']['Essay']), response2, grade2], ignore_index = True, axis=1)
        new_df.columns = ['Essay_prompt', 'Essay', 'Response','Grade']
        new_df.to_json('./51216100.0001c1_polyglot.json', force_ascii=False, orient = 'records', indent=4)

        torch.cuda.empty_cache()









