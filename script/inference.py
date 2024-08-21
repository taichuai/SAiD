"""Inference using the SAID_UNet1D model
"""
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from diffusers import DDIMScheduler
import torch
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from said.model.diffusion import SAID_UNet1D
from said.util.audio import fit_audio_unet, load_audio
from said.util.blendshape import (
    load_blendshape_coeffs,
    save_blendshape_coeffs,
    save_blendshape_coeffs_image,
)
from dataset.dataset_voca import BlendVOCADataset


def main():
    """Main function"""
    # Arguments
    parser = argparse.ArgumentParser(
        description="Inference the lipsync using the SAiD model"
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="../BlendVOCA/SAiD.pth",
        help="Path of the weights of SAiD model",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default="../BlendVOCA/audio/FaceTalk_170731_00024_TA/sentence01.wav",
        help="Path of the audio file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../out.csv",
        help="Path of the output blendshape coefficients file (csv format)",
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        default="./out.png",
        help="Path of the image of the output blendshape coefficients",
    )
    parser.add_argument(
        "--intermediate_dir",
        type=str,
        default="./interm",
        help="Saving directory of the intermediate outputs",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        help="Prediction type of the scheduler function, 'epsilon', 'sample', or 'v_prediction'",
    )
    parser.add_argument(
        "--save_image",
        type=bool,
        default=True,
        help="Save the output blendshape coefficients as an image",
    )
    parser.add_argument(
        "--save_intermediate",
        type=bool,
        default=False,
        help="Save the intermediate outputs",
    )
    parser.add_argument(
        "--num_steps", type=int, default=1000, help="Number of inference steps"
    )
    parser.add_argument("--strength", type=float, default=1.0, help="How much to paint")
    parser.add_argument(
        "--guidance_scale", type=float, default=2.0, help="Guidance scale"
    )
    parser.add_argument(
        "--guidance_rescale", type=float, default=0.0, help="Guidance scale"
    )
    parser.add_argument(
        "--eta", type=float, default=0.0, help="Eta for DDIMScheduler, between [0, 1]"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="FPS of the blendshape coefficients sequence",
    )
    parser.add_argument(
        "--divisor_unet",
        type=int,
        default=1,
        help="Length of the blendshape coefficients sequence should be divided by this number",
    )
    parser.add_argument(
        "--unet_feature_dim",
        type=int,
        default=-1,
        help="Dimension of the latent feature of the UNet",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="GPU/CPU device",
    )
    parser.add_argument(
        "--init_sample_path",
        default=None,
        type=str,
        help="Path of the initial sample file (csv format)",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        help="Path of the mask file (csv format)",
    )
    args = parser.parse_args()

    weights_path = args.weights_path
    audio_path = args.audio_path
    output_path = args.output_path
    output_image_path = args.output_image_path
    intermediate_dir = args.intermediate_dir
    prediction_type = args.prediction_type
    num_steps = args.num_steps
    strength = args.strength
    guidance_scale = args.guidance_scale
    guidance_rescale = args.guidance_rescale
    eta = args.eta
    fps = args.fps
    divisor_unet = args.divisor_unet
    unet_feature_dim = args.unet_feature_dim
    device = args.device
    save_image = args.save_image
    save_intermediate = args.save_intermediate
    show_process = True

    # Load init sample
    init_sample_path = args.init_sample_path
    init_samples = None
    if init_sample_path is not None:
        init_samples = load_blendshape_coeffs(init_sample_path).unsqueeze(0).to(device)

    # Load mask
    mask_path = args.mask_path
    mask = None
    if mask_path is not None:
        mask = load_blendshape_coeffs(mask_path).unsqueeze(0).to(device)

    # Load model
    said_model = SAID_UNet1D(
        noise_scheduler=DDIMScheduler,
        feature_dim=unet_feature_dim,
        prediction_type=prediction_type,
    )
    said_model.load_state_dict(torch.load(weights_path, map_location=device))
    print('Loaded SAID model.')
    said_model.to(device)
    said_model.eval()

    # Load data
    waveform = load_audio(audio_path, said_model.sampling_rate)
    # waveform = torch.zeros_like(waveform)

    # Process the waveform
    audio_4_len = waveform.shape[0] // (160 * 4)
    waveform = waveform[:audio_4_len * (160 * 4)]# Fit the size of waveform
    
    window_len = audio_4_len
    waveform = np.pad(waveform, (160, 160), mode='constant', constant_values=(0, 0))# Fit the size of waveform
    waveform = torch.from_numpy(waveform) # Fit the size of waveform

    # Fit the size of waveform
    # fit_output = fit_audio_unet(waveform, said_model.sampling_rate, fps, divisor_unet)
    # waveform = fit_output.waveform
    #  window_len= fit_output.window_size

    print("window_len: ", window_len)
    print('waveform', waveform.shape)
    # Process the waveform
    waveform_processed = said_model.process_audio(waveform).to(device)
    print('waveform_processed', waveform_processed.shape)# Inference

    # Inference
    with torch.no_grad():
        output = said_model.inference(
            waveform_processed=waveform_processed,
            init_samples=init_samples,
            mask=mask,
            num_inference_steps=num_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            eta=eta,
            save_intermediate=save_intermediate,
            show_process=show_process,
        )

    result = output.result[0, :window_len].cpu().numpy()
    
    print('result shape: ', result.shape)
    
    np.save(output_path, result)

    # save_blendshape_coeffs(
    #     coeffs=result,
    #     classes=BlendVOCADataset.default_blendshape_classes,
    #     output_path=output_path,
    # )

    # Save coeffs as an image
    if save_image:
        save_blendshape_coeffs_image(result, output_image_path)

    # Save intermediates
    # if save_intermediate:
    #     os.makedirs(intermediate_dir, exist_ok=True)
    #     for t, interm in enumerate(reversed(output.intermediates)):
    #         interm_coeffs = interm[0, :window_len].cpu().numpy()
    #         timestep = t + 1

    #         save_blendshape_coeffs_image(
    #             interm_coeffs, os.path.join(intermediate_dir, f"{timestep}.png")
    #         )

    #         save_blendshape_coeffs(
    #             coeffs=interm_coeffs,
    #             classes=BlendVOCADataset.default_blendshape_classes,
    #             output_path=os.path.join(intermediate_dir, f"{timestep}.csv"),
    #         )


if __name__ == "__main__":
    main()
