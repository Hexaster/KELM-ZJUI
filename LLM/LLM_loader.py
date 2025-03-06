import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLM_loader:
    def __init__(self, model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"):
        # Load the model
        self.tokeniser = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     device_map="auto")
        # Set up the system prompt and assistant marker first
        self.system_prompt = "<|im_start|>system\nYou are a helpful AI assistant to answer questions for people\n<|im_end|>"
        self.assistant_marker = "<|im_start|>assistant\n"
    def generate_response(self, prompt, max_new_tokens = 50, temperature = 0.6):
        """
        Generates a response based on the provided input prompt using a language model.
        :param prompt: The prompt from the user.
        :type prompt: str
        :param max_new_tokens: The maximum number of new tokens to generate in the
            response. Defaults to 50.
        :type max_new_tokens: int, optional
        :param temperature: Sampling temperature to control randomness in the
            generation. Lower values make the output more deterministic, while higher
            values increase variety. Defaults to 0.6.
        :type temperature: float, optional
        :return: The generated textual response from the model.
        :rtype: str
        """
        # Format the user prompt with the appropriate tokens
        user_prompt = f"<|im_start|>user\n{prompt}\n<|im_end|>"
        # Combine everything into one complete prompt
        full_prompt = self.system_prompt + "\n" + user_prompt + "\n" + self.assistant_marker

        inputs = self.tokeniser(full_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True
            )
        generated_text = self.tokeniser.decode(outputs[0], skip_special_tokens=True)
        return generated_text


def main():
    print("Initializing the language model...")
    llm = LLM_loader()

    print("\nLanguage model initialized successfully!")
    print("-------------------------------------------")

    while True:
        user_input = input("\nEnter your prompt (or 'exit' to quit): ")

        if user_input.lower() == "exit":
            break

        print("\nGenerating response...")
        response = llm.generate_response(user_input)
        print(f"\nResponse: {response}")

if __name__ == "__main__":
    main()