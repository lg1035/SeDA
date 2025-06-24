
import torch
import transformers
import openai

class ChatGLM2():
    def __init__(self, args) -> None:
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained("../../../.cache/chatglm2-6b", trust_remote_code=True, revision="v1.0")
        self.model = AutoModel.from_pretrained("../../../.cache/chatglm2-6b", trust_remote_code=True, revision="v1.0").to("cuda:" + args.cuda)
        self.args = args
        #print("✅ Using ChatGLM2 with generate() instead of .chat()")

    def qa(self, prompt):
        with torch.no_grad():
            if isinstance(prompt, list):
                answers = []
                for p in prompt:
                    response = self._chat_single(p)
                    answers.append(response)
                return answers
            else:
                return self._chat_single(prompt)

    def _chat_single(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.args.max_length,
            temperature=self.args.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace('\n', ' ').replace('\t', ' ')

    def generate_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        embedding = torch.mean(embeddings, dim=1)
        return embedding

class DeepSeek():
    def __init__(self, args) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_path = '../../../.cache/deepseek-llm-7b-chat'
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).to("cuda:" + args.cuda)

    def qa(self, prompt):
        def clean_response(p, r):
            return r[len(p):].strip() if r.startswith(p) else r.strip()

        if isinstance(prompt, list):
            return [clean_response(p, self._chat_single(p)) for p in prompt]
        else:
            return clean_response(prompt, self._chat_single(prompt))

    def _chat_single(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.args.max_length,
            temperature=self.args.temperature,
            do_sample=True,
            top_k=10,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace('\n', ' ').replace('\t', ' ')



class GPT():
    def __init__(self,args) -> None:  
        
        openai.api_key = "Your API Key"
        self.args = args
    def qa(self,prompt):
        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=self.args.temperature,
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],)
        ans = completion.choices[0].message.content
        ans = ans.replace('\n', ' ')
        ans = ans.replace('\t', ' ')
        return ans

class LLAMA2():
    def __init__(self,args) -> None:
        from transformers import AutoTokenizer
        model_path = '../../../.cache/LLaMA2-7b'
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            torch_dtype=torch.float16,
            # device_map="auto",#"auto",
            device="cuda:"+args.cuda,
        )

    def qa(self, prompt):
        sequences = self.pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=self.args.max_length,
            temperature=self.args.temperature,
        )

        if isinstance(prompt, list):  # batch
            answers = []
            for p, seq in zip(prompt, sequences):
                ans = seq[0]['generated_text']
                if ans.startswith(p):
                    ans = ans[len(p):].strip()
                ans = ans.replace('\n', ' ').replace('\t', ' ')
                answers.append(ans)
            return answers
        else:
            ans = sequences[0]['generated_text']
            if ans.startswith(prompt):
                ans = ans[len(prompt):].strip()
            return ans.replace('\n', ' ').replace('\t', ' ')


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='description of the program')
    parser.add_argument( '--cuda',default='1', type=str, help='help message for arg1')
    parser.add_argument('--split',default=1, type=int, help='help message for arg2')
    parser.add_argument('--max_length',default=200, type=int, help='help message for arg2')
    parser.add_argument('--temperature', default=0.2,type=float, help='help message for arg2')
    args = parser.parse_args()

    llm = DeepSeek(args)
    ans = llm.qa(['Tell me all you know about trump:',
                  'Tell me all you know about Biden:',])
    print(ans)

    ans = llm.qa('Tell me all the information about Biden:',)
    print(ans)