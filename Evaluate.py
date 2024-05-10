from datasets import Dataset, load_dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import re
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training,PeftConfig,PeftModel
import os

class Evaluate_NLP:
    def __init__(self, model_name, model_ft_name=None, quantize=True, lora=True):
        """
        Initializes the Evaluate_NLP class for evaluating NLP models.

        Args:
            model_name (str): Name of the base model from Hugging Face.
            model_ft_name (str, optional): Name of the fine-tuned model from Hugging Face. Defaults to None.
            quantize (bool, optional): Whether to use quantization (BitsAndBytesConfig). Defaults to True.
            lora (bool, optional): Whether to use LoRA (PEFT). Defaults to True.
        """

        self.model_name = model_name
        self.model_ft_name = model_ft_name
        self.quantize = quantize
        self.lora = lora

        # Define configurations
        if self.quantize:
            # Define your quantization configuration
            self.quantization_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16
                                    )
        else:
            self.quantization_config = None

        if self.lora:
            # Define your LoRA configuration
            self.lora_config = LoraConfig(
                                    r = 16,
                                    lora_alpha=16,
                                    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head'],
                                    lora_dropout=0.1,
                                    bias="none",
                                    task_type=TaskType.CAUSAL_LM
                                    )
        else:
            self.lora_config = None

        # Load tokenizer (always needed)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        # Load model (base or fine-tuned)
        if model_ft_name is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=self.quantization_config,device_map = 'auto' , trust_remote_code=True)
        else:
            # Handle fine-tuned model loading using Peft if necessary
            config = PeftConfig.from_pretrained(self.model_ft_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=self.quantization_config,device_map = 'auto' , trust_remote_code=True)
            self.model = PeftModel.from_pretrained(self.model, self.model_ft_name)

        # Apply LoRA if specified
        if self.lora:
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, self.lora_config)

    def mmlu_eval(self) :
        # # load dataset
        # dataset = load_dataset("cais/mmlu", 'all', trust_remote_code=True)
        # test = pd.DataFrame(dataset['test'])
        # dev = pd.DataFrame(dataset['dev'])

        # # choices
        # choices = ["A", "B", "C", "D"]
        pass
    
    # Fill your path to Story_cloze dataset
    def story_cloze(self,output_path, dataset_path) :
        # read story_cloze dataset
        df = pd.read_csv(dataset_path)
        ans = []
        story_ids = []
        # iteration data
        for idx in tqdm(range(len(df))) : 
            doc = df.iloc[idx]
            # Generate prompt
            story = doc['InputSentence1'] + ' '
            story += doc['InputSentence2'] + ' '
            story += doc['InputSentence2'] + ' '
            story += doc['InputSentence2']
            text_choice = 'A. ' + doc['RandomFifthSentenceQuiz1'] + '\nB. ' + doc['RandomFifthSentenceQuiz2']
            prompt = "<|user|>\nChoose one of two following ending for this story, return A with the ending A and B with the ending B:\n" \
                        + story \
                        + "\n\n" \
                        + text_choice \
                        + "\n" \
                        + "<|assistant|>\n" \
                        + "Answer: " 
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            # Evaluate your model
            with torch.no_grad():
                    outputs = self.model(input_ids=input_ids)
            logits = outputs.logits  # shape (batch_size, sequence_length, vocab_size)
            next_token_logits = logits[:, -1, :]  # shape (batch_size, vocab_size)
        #         print(logits)

            next_token_logits = next_token_logits.flatten()
            next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1).cpu()
            tokens_of_interest = [
                self.tokenizer("A", add_special_tokens=False).input_ids[-1],
                self.tokenizer("B", add_special_tokens=False).input_ids[-1],
            ]
            probs = next_token_probs[tokens_of_interest].tolist()
            pred = {0: "A", 1: "B"}[np.argmax(probs)]
            ans.append(pred)
            story_ids.append(doc['InputStoryid'])
        # Save to csv file
        result_df = pd.DataFrame({"InputStoryid": story_ids, "Answer": ans})

        result_df.to_csv(output_path, index=False)
        return ans