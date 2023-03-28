import csv
import gradio as gr
import open_clip
import os
import torch

from PIL import Image

import clip_interrogator
from clip_interrogator import Config, Interrogator

from modules import devices, lowvram, script_callbacks, shared

__version__ = '0.1.4'

ci = None
low_vram = False

BATCH_OUTPUT_MODES = [
    'Text file for each image',
    'Single text file with all prompts',
    'csv file with columns for filenames and prompts',
]

class BatchWriter:
    def __init__(self, folder, mode):
        self.folder = folder
        self.mode = mode
        self.csv, self.file = None, None
        if mode == BATCH_OUTPUT_MODES[1]:
            self.file = open(os.path.join(folder, 'batch.txt'), 'w', encoding='utf-8')
        elif mode == BATCH_OUTPUT_MODES[2]:
            self.file = open(os.path.join(folder, 'batch.csv'), 'w', encoding='utf-8', newline='')
            self.csv = csv.writer(self.file, quoting=csv.QUOTE_MINIMAL)
            self.csv.writerow(['filename', 'prompt'])

    def add(self, file, prompt):
        if self.mode == BATCH_OUTPUT_MODES[0]:
            txt_file = os.path.splitext(file)[0] + ".txt"
            with open(os.path.join(self.folder, txt_file), 'w', encoding='utf-8') as f:
                f.write(prompt)
        elif self.mode == BATCH_OUTPUT_MODES[1]:
            self.file.write(f"{prompt}\n")
        elif self.mode == BATCH_OUTPUT_MODES[2]:
            self.csv.writerow([file, prompt])

    def close(self):
        if self.file is not None:
            self.file.close()


def load(clip_model_name):
    global ci
    if ci is None:
        print(f"Loading CLIP Interrogator {clip_interrogator.__version__}...")

        config = Config(
            device=devices.get_optimal_device(), 
            cache_path = 'models/clip-interrogator',
            clip_model_name=clip_model_name,
            blip_model=shared.interrogator.load_blip_model().float()
        )
        if low_vram:
            config.apply_low_vram_defaults()
        ci = Interrogator(config)

    if clip_model_name != ci.config.clip_model_name:
        ci.config.clip_model_name = clip_model_name
        ci.load_clip_model()

def unload():
    global ci
    if ci is not None:
        print("Offloading CLIP Interrogator...")
        ci.blip_model = ci.blip_model.to(devices.cpu)
        ci.clip_model = ci.clip_model.to(devices.cpu)
        ci.blip_offloaded = True
        ci.clip_offloaded = True
        devices.torch_gc()

def image_analysis(image, clip_model_name):
    load(clip_model_name)

    image = image.convert('RGB')
    image_features = ci.image_to_features(image)

    top_mediums = ci.mediums.rank(image_features, 5)
    top_artists = ci.artists.rank(image_features, 5)
    top_movements = ci.movements.rank(image_features, 5)
    top_trendings = ci.trendings.rank(image_features, 5)
    top_flavors = ci.flavors.rank(image_features, 5)

    medium_ranks = {medium: sim for medium, sim in zip(top_mediums, ci.similarities(image_features, top_mediums))}
    artist_ranks = {artist: sim for artist, sim in zip(top_artists, ci.similarities(image_features, top_artists))}
    movement_ranks = {movement: sim for movement, sim in zip(top_movements, ci.similarities(image_features, top_movements))}
    trending_ranks = {trending: sim for trending, sim in zip(top_trendings, ci.similarities(image_features, top_trendings))}
    flavor_ranks = {flavor: sim for flavor, sim in zip(top_flavors, ci.similarities(image_features, top_flavors))}
    
    return medium_ranks, artist_ranks, movement_ranks, trending_ranks, flavor_ranks

def interrogate(image, mode, caption=None):
    if mode == 'best':
        prompt = ci.interrogate(image, caption=caption)
    elif mode == 'caption':
        prompt = ci.generate_caption(image) if caption is None else caption
    elif mode == 'classic':
        prompt = ci.interrogate_classic(image, caption=caption)
    elif mode == 'fast':
        prompt = ci.interrogate_fast(image, caption=caption)
    elif mode == 'negative':
        prompt = ci.interrogate_negative(image)
    else:
        raise Exception(f"Unknown mode {mode}")
    return prompt

def image_to_prompt(image, mode, clip_model_name):
    shared.state.begin()
    shared.state.job = 'interrogate'

    try: 
        if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
            lowvram.send_everything_to_cpu()
            devices.torch_gc()

        load(clip_model_name)
        image = image.convert('RGB')
        prompt = interrogate(image, mode)
    except torch.cuda.OutOfMemoryError as e:
        prompt = "Ran out of VRAM"
        print(e)
    except RuntimeError as e:
        prompt = f"Exception {type(e)}"
        print(e)

    shared.state.end()
    return prompt


def about_tab():
    gr.Markdown("## 🕵️‍♂️ CLIP Interrogator 🕵️‍♂️")
    gr.Markdown("*Want to figure out what a good prompt might be to create new images like an existing one? The CLIP Interrogator is here to get you answers!*")
    gr.Markdown("## Notes")
    gr.Markdown(
        "CLIP models:\n"
        "* For best prompts with Stable Diffusion 1.* choose the **ViT-L-14/openai** model.\n"
        "* For best prompts with Stable Diffusion 2.* choose the **ViT-H-14/laion2b_s32b_b79k** model.\n"
        "\nOther:\n"
        "* When you are done click the **Unload** button to free up memory."
    )
    gr.Markdown("## Github")
    gr.Markdown("If you have any issues please visit [CLIP Interrogator on Github](https://github.com/pharmapsychotic/clip-interrogator) and drop a star if you like it!")
    gr.Markdown(f"<br><br>CLIP Interrogator version: {clip_interrogator.__version__}<br>Extension version: {__version__}")

    if torch.cuda.is_available():
        device = devices.get_optimal_device()
        vram_total_mb = torch.cuda.get_device_properties(device).total_memory / (1024**2)
        vram_info = f"GPU VRAM: **{vram_total_mb:.2f}MB**"
        if low_vram:
            vram_info += "<br>Using low VRAM configuration"
        gr.Markdown(vram_info)

def get_models():
    return ['/'.join(x) for x in open_clip.list_pretrained()]

def analyze_tab():
    with gr.Column():
        with gr.Row():
            image = gr.Image(type='pil', label="Image")
            model = gr.Dropdown(get_models(), value='ViT-L-14/openai', label='CLIP Model')
        with gr.Row():
            medium = gr.Label(label="Medium", num_top_classes=5)
            artist = gr.Label(label="Artist", num_top_classes=5)        
            movement = gr.Label(label="Movement", num_top_classes=5)
            trending = gr.Label(label="Trending", num_top_classes=5)
            flavor = gr.Label(label="Flavor", num_top_classes=5)
    button = gr.Button("Analyze", variant='primary')
    button.click(image_analysis, inputs=[image, model], outputs=[medium, artist, movement, trending, flavor])

def batch_tab():
    def batch_process(folder, clip_model, mode, output_mode):
        if not os.path.exists(folder):
            return f"Folder {folder} does not exist"
        if not os.path.isdir(folder):
            return "{folder} is not a directory"

        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            return "Folder has no images"

        shared.state.begin()
        shared.state.job = 'batch interrogate'

        try: 
            if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
                lowvram.send_everything_to_cpu()
                devices.torch_gc()

            load(clip_model)

            shared.total_tqdm.updateTotal(len(files))
            ci.config.quiet = True

            # generate captions in first pass
            captions = []
            for file in files:
                if shared.state.interrupted:
                    break
                image = Image.open(os.path.join(folder, file)).convert('RGB')
                captions.append(ci.generate_caption(image))
                shared.total_tqdm.update()

            # interrogate in second pass
            writer = BatchWriter(folder, output_mode)
            shared.total_tqdm.clear()
            shared.total_tqdm.updateTotal(len(files))
            for idx, file in enumerate(files):
                if shared.state.interrupted:
                    break
                image = Image.open(os.path.join(folder, file)).convert('RGB')
                prompt = interrogate(image, mode, caption=captions[idx])
                writer.add(file, prompt)
                shared.total_tqdm.update()

            writer.close()
            ci.config.quiet = False
            unload()
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            print("Ran out of VRAM!")
        except RuntimeError as e:
            print(e)
        shared.state.end()
        shared.total_tqdm.clear()

    with gr.Column():
        with gr.Row():
            folder = gr.Text(label="Images folder", value="", interactive=True)
        with gr.Row():
            clip_model = gr.Dropdown(get_models(), value='ViT-L-14/openai', label='CLIP Model')
            mode = gr.Radio(['caption', 'best', 'fast', 'classic', 'negative'], label='Prompt Mode', value='fast')
            output_mode = gr.Dropdown(BATCH_OUTPUT_MODES, value=BATCH_OUTPUT_MODES[0], label='Output Mode')
        with gr.Row():        
            button = gr.Button("Go!", variant='primary')
            interrupt = gr.Button('Interrupt', visible=True)
            interrupt.click(fn=lambda: shared.state.interrupt(), inputs=[], outputs=[])

    button.click(batch_process, inputs=[folder, clip_model, mode, output_mode], outputs=[])

def prompt_tab():
    with gr.Column():
        with gr.Row():
            image = gr.Image(type='pil', label="Image")
            with gr.Column():
                mode = gr.Radio(['best', 'fast', 'classic', 'negative'], label='Mode', value='best')
                clip_model = gr.Dropdown(get_models(), value='ViT-L-14/openai', label='CLIP Model')
        prompt = gr.Textbox(label="Prompt", lines=3)
    with gr.Row():
        button = gr.Button("Generate", variant='primary')
        unload_button = gr.Button("Unload")
    button.click(image_to_prompt, inputs=[image, mode, clip_model], outputs=prompt)
    unload_button.click(unload)


def add_tab():
    global low_vram
    low_vram = shared.cmd_opts.lowvram or shared.cmd_opts.medvram
    if not low_vram and torch.cuda.is_available():
        device = devices.get_optimal_device()
        vram_total = torch.cuda.get_device_properties(device).total_memory
        if vram_total <= 12*1024*1024*1024:
            low_vram = True

    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tab("Prompt"):
            prompt_tab()
        with gr.Tab("Analyze"):
            analyze_tab()
        with gr.Tab("Batch"):
            batch_tab()
        with gr.Tab("About"):
            about_tab()

    return [(ui, "Interrogator", "interrogator")]

script_callbacks.on_ui_tabs(add_tab)
