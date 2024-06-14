import os
import torch
import folder_paths

from torchvision.transforms.functional import to_pil_image
from .models.nextdit import NextDiT

import comfy.supported_models_base
import comfy.latent_formats
import comfy.model_patcher
import comfy.model_base
import comfy.utils
import comfy.conds
import comfy.model_management
import numpy as np
import math
from .transport import Sampler, create_transport

nextdit = None
model_name = None

tokenizer = None
text_encoder = None

def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts):
    captions = []
    for caption in prompt_batch:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding=True,
            pad_to_multiple_of=8,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask.cuda()

        prompt_embeds = text_encoder(
            input_ids=text_input_ids.cuda(),
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks

class NextDiTInfer:
	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
				"llm_name": (folder_paths.get_filename_list("clip"),),
				"height": ("INT", {"default": 1024, "min": 512, "max": 2048}),
				"width": ("INT", {"default": 1024, "min": 512, "max": 2048}),
				"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
				"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
				"cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
				"time_shift": ("INT", {"default": 4, "min": 1, "max": 20}),
				"solver": (["euler", "midpoint", "rk4"], ),
				"scaling_method": (["Time-aware", "None"], ),
				"scaling_watershed": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
				"proportional_attn": ("BOOLEAN", {"default": True}),
				"keep_model_on": ("BOOLEAN", {"default": False}),
				"model_dtype": (["bf16", "float32", "float16"], ),
				"positive": ("STRING", {"multiline": True, "dynamicPrompts": True}),
				"negative": ("STRING", {"multiline": True, "dynamicPrompts": True}),
			}
		}
	RETURN_TYPES = ("LATENT",)
	RETURN_NAMES = ("latent",)
	FUNCTION = "load_checkpoint"
	CATEGORY = "ExtraModels/NextDiT"
	TITLE = "NextDiT Text to Image"

	def load_checkpoint(self, ckpt_name, llm_name, height, width, seed, steps, cfg, time_shift, solver, scaling_method, scaling_watershed, proportional_attn, keep_model_on, model_dtype, positive, negative):
		model_type = comfy.model_management.unet_dtype()
		if model_dtype == 'bf16':
			model_dtype = torch.bfloat16
		elif model_dtype == 'float16':
			model_dtype = torch.float16
		else:
			model_dtype = torch.float32
		load_device = comfy.model_management.get_torch_device()
		offload_device = comfy.model_management.unet_offload_device()

		clip_path = folder_paths.get_full_path("clip", llm_name)
		llm_folder = os.path.dirname(clip_path)
		global tokenizer, text_encoder
		from transformers import AutoModel, AutoTokenizer
		if tokenizer is None:
			tokenizer = AutoTokenizer.from_pretrained(llm_folder)
			tokenizer.padding_side = "right"
		if text_encoder is None:
			text_encoder = AutoModel.from_pretrained(llm_folder, torch_dtype=model_type, device_map=load_device)
		cap_feats, cap_mask = encode_prompt([positive]+[negative], text_encoder, tokenizer, 0)
		if not keep_model_on:
			text_encoder = None
			comfy.model_management.get_free_memory()
		ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
		global nextdit, model_name
		if nextdit is None:
			nextdit = NextDiT(
				patch_size=2, 
				dim=2304, 
				n_layers=24, 
				n_heads=32, 
				n_kv_heads=8, 
				qk_norm=True,
				cap_feat_dim=2048
			)
			state_dict = comfy.utils.load_torch_file(ckpt_path)
			model_name = ckpt_path
			nextdit.load_state_dict(state_dict)
			nextdit.to(load_device).to(model_type)
		model_kwargs = dict(
			cap_feats=cap_feats,
			cap_mask=cap_mask,
			cfg_scale=cfg,
		)
		if proportional_attn:
			model_kwargs["proportional_attn"] = True
			model_kwargs["base_seqlen"] = (64) ** 2
		else:
			model_kwargs["proportional_attn"] = False
			model_kwargs["base_seqlen"] = None
		
		do_extrapolation = (width * height > 1024 * 1024 * 1.25)

		if do_extrapolation and scaling_method == "Time-aware":
			model_kwargs["scale_factor"] = math.sqrt(width * height / 1024**2)
			model_kwargs["scale_watershed"] = scaling_watershed
		else:
			model_kwargs["scale_factor"] = 1.0
			model_kwargs["scale_watershed"] = 1.0
		
		transport = create_transport(
			"Linear",
			"velocity",
			None,
			None,
			None,
		)
		sampler = Sampler(transport)
		sample_fn = sampler.sample_ode(
			sampling_method=solver,
			num_steps=steps,
			atol=1e-6,
			rtol=1e-3,
			reverse=False,
			time_shifting_factor=time_shift,
		)
		if int(seed) != 0:
			torch.random.manual_seed(int(seed))
		with torch.no_grad():
			z = torch.randn([1, 4, height // 8, width // 8], device=load_device).to(model_type)
			z = z.repeat(2, 1, 1, 1)
			samples = sample_fn(z, nextdit.forward_with_cfg, **model_kwargs)[-1]
		samples = samples[:1] / 0.13025
		if not keep_model_on:
			nextdit = None
			comfy.model_management.get_free_memory()

		return ({"samples":samples},)

NODE_CLASS_MAPPINGS = {
	"NextDiTInfer" : NextDiTInfer,
}
