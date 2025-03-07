import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# This class loads a llm from hugging face, in order to interact with Django
class LLM_loader:
    def __init__(self, model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"):
        # Load the model
        self.tokeniser = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     device_map="auto")
        # Ensure that the history always contain the prompt from the system
        self.default_system_prompt = {
        "role": "system",
        "text": "You are KELM, a knowledge-enhanced language model that can help you answer questions about your knowledge base."
    }
        self.chat_history = [self.default_system_prompt]
        pass

    def set_chat_history(self, messages):
        """
        If we already have messages, we can directly set it.
        :param messages: A list of dictionaries, where each dictionary represents a
            message. Each message must contain a "role" key which specifies its type,
            such as "system", "user", or "assistant". The content of the messages
            defines the interaction flow.
        :return: None
        """
        self.messages = [self.default_system_prompt] + messages
        pass

    def apply_chat_template(self):
        conversation = ""
        #NEW_RESPONSE_MARKER = "[NEW_RESPONSE]"
        for message in self.messages:
            if message["role"] == "system":
                conversation += f"<|im_start|>System: {message['text']}<|im_end|>\n"
            elif message["role"] == "user":
                conversation += f"<|im_start|>User: {message['text']}<|im_end|>\n"
            elif message["role"] == "assistant":
                conversation += f"<|im_start|>Assistant: {message['text']}<|im_end|>\n"
        if self.messages[-1]["role"] == "user":
            #conversation += f"Assistant: {NEW_RESPONSE_MARKER}"
            conversation += "<|im_start|>Assistant: "
        return conversation

    def generate_response(self, prompt, max_new_tokens = 128, temperature = 0.6):
        """
        Generates a response based on the provided input prompt using a language model.
        :param prompt: The prompt from the user.
        :type prompt: str
        :param max_new_tokens: The maximum number of new tokens to generate in the
            response.
        :type max_new_tokens: int, optional
        :param temperature: Sampling temperature to control randomness in the
            generation. Lower values make the output more deterministic, while higher
            values increase variety. Defaults to 0.6.
        :type temperature: float, optional
        :return: The generated textual response from the model.
        :rtype: str
        """
        # Append user's prompt to self.message so the history is updated
        self.messages.append({"role": "user", "text": prompt})
        # Apply the chat template
        input_text = self.apply_chat_template()
        #input_text = self.tokeniser.apply_chat_template(self.messages, tokenize=False)
        inputs = self.tokeniser(input_text, return_tensors="pt").to(self.model.device)
        # Get the length of the prompt so we can trim the generated text later
        prompt_length = inputs.input_ids.shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True
            )
        generated_text = self.tokeniser.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        # Append AI's response to messages
        self.messages.append({"role": "assistant", "text": generated_text})
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