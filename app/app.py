import sys
sys.path.append(".")
sys.path.append("starter")
sys.path.append("part_a_mae")
sys.path.append("part_b_vae")
sys.path.append("part_c_diffusion")
sys.path.append("part_d_rag")

import numpy as np
import torch
import torch.nn as nn
import gradio as gr
from common import *
from eurosat import load_eurosat_split, patchify, unpatchify, CLASS_NAMES
from mae_model import MAE
from vae_model import VAE
from unet import make_unet
from sample import sample_images, to_01
from rag import load_retriever, embed_texts, Index
from generate import load_generator, generate_answer, NOTICE

store = {}


def get(name, loader):
    if name not in store:
        store[name] = loader()
    return store[name]


def to_picture(images, ncols, scale):
    images = images.detach().cpu().clamp(0, 1)
    n, c, h, w = images.shape
    rows = (n + ncols - 1) // ncols
    canvas = np.zeros((rows * h, ncols * w, 3), dtype=np.float32)
    for i in range(n):
        r = i // ncols
        col = i % ncols
        tile = images[i].permute(1, 2, 0).numpy()
        if c == 1:
            tile = np.repeat(tile, 3, axis=2)
        canvas[r * h:(r + 1) * h, col * w:(col + 1) * w] = tile
    canvas = np.repeat(np.repeat(canvas, scale, axis=0), scale, axis=1)
    return (canvas * 255).astype(np.uint8)


def load_mae_parts():
    model = MAE().to(device)
    model.load_state_dict(torch.load("weights/mae.pt", map_location=device))
    model.eval()
    classifier = nn.Linear(192, 10).to(device)
    classifier.load_state_dict(torch.load("weights/mae_classifier.pt", map_location=device))
    images, labels = load_eurosat_split("test")
    return model, classifier, images, labels


def run_mae(tile_number):
    model, classifier, images, labels = get("mae", load_mae_parts)
    tile_number = int(tile_number)
    image = images[tile_number:tile_number + 1]
    patches = patchify(image)
    with torch.no_grad():
        loss, pred, mask = model(patches.to(device))
        features = model.features(patches.to(device))
        predicted_class = classifier(features).argmax(dim=1).item()
    pred = pred.cpu()
    mask3 = mask.cpu().unsqueeze(-1)
    masked_input = unpatchify(patches * (1 - mask3) + 0.5 * mask3)
    reconstruction = unpatchify(patches * (1 - mask3) + pred * mask3)
    text = "Predicted class: " + CLASS_NAMES[predicted_class] + "  |  True class: " + CLASS_NAMES[labels[tile_number].item()]
    return to_picture(masked_input, 1, 4), to_picture(reconstruction, 1, 4), to_picture(image, 1, 4), text


def load_vae_parts():
    model = VAE().to(device)
    model.load_state_dict(torch.load("weights/vae.pt", map_location=device))
    model.eval()
    x_train, y_train, x_test, y_test, names = load_uhat()
    return model, x_test, y_test, names


def new_characters():
    model, x_test, y_test, names = get("vae", load_vae_parts)
    z = torch.randn(64, 16).to(device)
    with torch.no_grad():
        samples = torch.sigmoid(model.decode(z))
    return to_picture(samples, 8, 3)


def interpolate(first_name, second_name, amount):
    model, x_test, y_test, names = get("vae", load_vae_parts)
    first = names.index(first_name)
    second = names.index(second_name)
    a = x_test[(y_test == first).nonzero()[0].item()].unsqueeze(0).to(device)
    b = x_test[(y_test == second).nonzero()[0].item()].unsqueeze(0).to(device)
    with torch.no_grad():
        mu_a, _ = model.encode(a)
        mu_b, _ = model.encode(b)
        z = (1 - amount) * mu_a + amount * mu_b
        result = torch.sigmoid(model.decode(z))
    return to_picture(torch.cat([a.cpu(), result.cpu(), b.cpu()], dim=0), 3, 8)


def load_unet():
    unet = make_unet().to(device)
    unet.load_state_dict(torch.load("weights/unet.pt", map_location=device))
    unet.eval()
    return unet


def write_character(name):
    unet = get("unet", load_unet)
    names = load_json("results/diffusion_train.json")["class_names"]
    label = torch.full((8,), names.index(name), device=device, dtype=torch.long)
    samples, saved = sample_images(unet, label)
    return to_picture(to_01(samples), 8, 4)


def load_rag_parts():
    retriever_tokenizer, retriever_model = load_retriever()
    index = Index()
    generator_tokenizer, generator_model = load_generator()
    return retriever_tokenizer, retriever_model, index, generator_tokenizer, generator_model


def answer_question(question):
    retriever_tokenizer, retriever_model, index, generator_tokenizer, generator_model = get("rag", load_rag_parts)
    vector = embed_texts([question], retriever_tokenizer, retriever_model)[0]
    chunk_ids = index.top_chunks(vector, 5)
    passages = []
    for chunk_id in chunk_ids:
        passages.append(index.chunk_texts[chunk_id])
    answer = generate_answer(question, passages, generator_tokenizer, generator_model)

    shown = ""
    for number, chunk_id in enumerate(chunk_ids):
        shown = shown + "**[" + str(number + 1) + "]** abstract id " + str(index.chunk_abstract_ids[chunk_id]) + "\n\n"
        shown = shown + passages[number] + "\n\n"
    return answer + "\n\n*" + NOTICE + "*", shown


vae_names = load_json("results/vae_train.json")["class_names"] if os.path.exists("results/vae_train.json") else ["alif"]
diffusion_names = vae_names

with gr.Blocks(title="Four generative models") as demo:
    gr.Markdown("# Four generative models in one app")

    with gr.Tab("MAE (satellite images)"):
        gr.Markdown("Pick a test tile. The model sees only 25% of the patches and rebuilds the rest.")
        tile = gr.Slider(0, 5399, value=0, step=1, label="Test tile number")
        mae_button = gr.Button("Run")
        with gr.Row():
            masked_out = gr.Image(label="Masked input")
            recon_out = gr.Image(label="Reconstruction")
            original_out = gr.Image(label="Original")
        mae_text = gr.Textbox(label="Land cover class")
        mae_button.click(run_mae, inputs=tile, outputs=[masked_out, recon_out, original_out, mae_text])

    with gr.Tab("VAE (Urdu characters)"):
        vae_button = gr.Button("New random characters")
        vae_out = gr.Image(label="64 new characters")
        vae_button.click(new_characters, outputs=vae_out)
        with gr.Row():
            first_choice = gr.Dropdown(vae_names, value=vae_names[0], label="First character")
            second_choice = gr.Dropdown(vae_names, value=vae_names[1], label="Second character")
        amount = gr.Slider(0, 1, value=0.5, step=0.1, label="Morph amount")
        morph_out = gr.Image(label="First, morph, second")
        amount.change(interpolate, inputs=[first_choice, second_choice, amount], outputs=morph_out)

    with gr.Tab("Diffusion (write a character)"):
        character = gr.Dropdown(diffusion_names, value=diffusion_names[0], label="Character to write")
        diffusion_button = gr.Button("Generate 8 samples")
        diffusion_out = gr.Image(label="8 new samples")
        diffusion_button.click(write_character, inputs=character, outputs=diffusion_out)

    with gr.Tab("RAG (biomedical questions)"):
        gr.Markdown("**" + NOTICE + "**")
        question_box = gr.Textbox(label="Your question")
        rag_button = gr.Button("Answer")
        answer_out = gr.Markdown()
        passages_out = gr.Markdown()
        rag_button.click(answer_question, inputs=question_box, outputs=[answer_out, passages_out])

if __name__ == "__main__":
    demo.launch()
