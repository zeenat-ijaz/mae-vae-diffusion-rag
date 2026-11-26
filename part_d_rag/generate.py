import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rag import GENERATOR_NAME, device

NOTICE = "For educational use only. Not medical advice."


def load_generator():
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_NAME)
    model = AutoModelForCausalLM.from_pretrained(GENERATOR_NAME)
    if device == "cuda":
        model = model.half()
    model = model.to(device)
    model.eval()
    return tokenizer, model


def build_messages(question, passages):
    numbered = ""
    for i, passage in enumerate(passages):
        numbered = numbered + "[" + str(i + 1) + "] " + passage + "\n\n"

    system = ("You answer biomedical research questions. Use only the numbered passages given by the user. "
              "Cite the passage numbers in square brackets, for example [1]. "
              "If the passages do not contain the answer, say that you do not know.")
    user = "Passages:\n\n" + numbered + "Question: " + question
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_answer(question, passages, tokenizer, model):
    messages = build_messages(question, passages)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
