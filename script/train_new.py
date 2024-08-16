import argparse
from dataclasses import dataclass
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_path, ".."))

import pathlib
from typing import Optional
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from said.model.diffusion import SAID, SAID_UNet1D
from said.model.wav2vec2 import ModifiedWav2Vec2Model
from said.util.blendshape import load_blendshape_coeffs
from dataset.dataset_voca import DataBatch, BlendVOCATrainDataset, BlendVOCAValDataset


@dataclass
class LossStepOutput:
    """
    Dataclass for the losses at each step
    """
    predict: torch.FloatTensor  # MAE loss for the predicted output
    velocity: torch.FloatTensor  # MAE loss for the velocity
    vertex: Optional[torch.FloatTensor]  # MAE loss for the reconstructed vertex


@dataclass
class LossEpochOutput:
    """
    Dataclass for the averaged losses at each epoch
    """
    total: float = 0  # Averaged total loss
    predict: float = 0  # Averaged prediction loss
    velocity: float = 0  # Averaged velocity loss
    vertex: float = 0  # Averaged vertex loss
    lr: Optional[float] = None  # Last learning rate


def random_noise_loss(
    said_model: SAID,
    data: DataBatch,
    std: Optional[torch.FloatTensor],
    device: torch.device,
    prediction_type: str = "epsilon",
) -> LossStepOutput:
    """Compute the loss with randomized noises

    Parameters
    ----------
    said_model : SAID
        SAiD model object
    data : DataBatch
        Output of the BlendVOCADataset.collate_fn
    std : Optional[torch.FloatTensor]
        (1, x_dim), Standard deviation of coefficients
    device : torch.device
        GPU device
    prediction_type: str
        Prediction type of the scheduler function, "epsilon", "sample", or "v_prediction", by default "epsilon"

    Returns
    -------
    LossStepOutput
        Computed losses
    """
    waveform = data.waveform
    blendshape_coeffs = data.blendshape_coeffs.to(device)
    cond = data.cond.to(device)

    coeff_latents = said_model.encode_samples(
        blendshape_coeffs * said_model.latent_scale
    )

    curr_batch_size = len(waveform)
    window_size = blendshape_coeffs.shape[1]

    # print('waveform: ', waveform[0].shape)
    waveform_processed = said_model.process_audio(waveform).to(device)
    random_timesteps = said_model.get_random_timesteps(curr_batch_size).to(device)

    # print('window_size: ', window_size)
    cond_embedding = said_model.get_audio_embedding(waveform_processed, window_size)
    # print('wavel shape', waveform_processed.shape, 'cond embedding shape', cond_embedding.shape, 'coeff and latent', blendshape_coeffs.shape, coeff_latents.shape)
    uncond_embedding = said_model.null_cond_emb.repeat(
        curr_batch_size, cond_embedding.shape[1], 1
    )
    cond_mask = cond.view(-1, 1, 1)

    audio_embedding = cond_embedding * cond_mask + uncond_embedding * torch.logical_not(
        cond_mask
    )
    noise_dict = said_model.add_noise(coeff_latents, random_timesteps)
    noisy_latents = noise_dict.noisy_sample
    noise = noise_dict.noise
    velocity = noise_dict.velocity

    # print('noisy_latents: ', noisy_latents.shape, 'random: ', random_timesteps, 'audio embedding: ', audio_embedding.shape)
    pred = said_model(noisy_latents, random_timesteps, audio_embedding)
    # print('pred: ', pred.shape)

    # Set answer corresponding to prediction_type
    answer = None
    if prediction_type == "epsilon":
        answer = noise
    elif prediction_type == "sample":
        answer = coeff_latents
    elif prediction_type == "v_prediction":
        answer = velocity

    criterion_pred = nn.L1Loss()
    criterion_velocity = nn.L1Loss()
    criterion_vertex = nn.L1Loss()

    answer_reweight = answer
    pred_reweight = pred
    if std is not None:
        answer_reweight /= std.view(1, 1, -1)
        pred_reweight /= std.view(1, 1, -1)

    loss_pred = criterion_pred(pred_reweight, answer_reweight)

    answer_diff = answer_reweight[:, 1:, :] - answer_reweight[:, :-1, :]
    pred_diff = pred_reweight[:, 1:, :] - pred_reweight[:, :-1, :]

    loss_vel = criterion_velocity(pred_diff, answer_diff)

    loss_vertex = None
    if data.blendshape_delta is not None:
        blendshape_delta = data.blendshape_delta.to(device)
        b, k, v, i = blendshape_delta.shape
        _, t, _ = answer.shape

        blendshape_delta_norm = torch.norm(blendshape_delta, p=1, dim=[1, 2, 3]) / (
            k * v * i
        )
        blendshape_delta_normalized = torch.div(
            blendshape_delta,
            blendshape_delta_norm.view(-1, 1, 1, 1),
        )

        be_answer = torch.bmm(answer, blendshape_delta_normalized.view(b, k, v * i))
        be_pred = torch.bmm(pred, blendshape_delta_normalized.view(b, k, v * i))

        # be_answer = torch.einsum("bkvi,btk->btvi", blendshape_delta_normalized, answer)
        # be_pred = torch.einsum("bkvi,btk->btvi", blendshape_delta_normalized, pred)

        loss_vertex = criterion_vertex(be_pred, be_answer)

    return LossStepOutput(
        predict=loss_pred,
        velocity=loss_vel,
        vertex=loss_vertex,
    )


def train_epoch(
    said_model: SAID,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler,
    std: Optional[torch.FloatTensor],
    weight_vel: float,
    weight_vertex: float,
    prediction_type: str = "epsilon",
    ema_model: Optional[EMAModel] = None,
    epoch: int = 0,
) -> LossEpochOutput:
    """Train the SAiD model one epoch.

    Parameters
    ----------
    said_model : SAID
        SAiD model object
    train_dataloader : DataLoader
        Dataloader of the BlendVOCATrainDataset
    optimizer : torch.optim.Optimizer
        Optimizer object
    lr_scheduler: torch.optim.lr_scheduler
        Learning rate scheduler object
    std : Optional[torch.FloatTensor]
        (1, x_dim), Standard deviation of coefficients
    weight_vel: float
        Weight for the velocity loss
    weight_vertex: float
        Weight for the vertex loss
    prediction_type: str
        Prediction type of the scheduler function, "epsilon", "sample", or "v_prediction", by default "epsilon"
    ema_model: Optional[EMAModel]
        EMA model of said_model, by default None

    Returns
    -------
    LossEpochOutput
        Average losses
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    said_model.to(device)

    if std is not None:
        std = std.to(device)

    said_model.train()

    train_total_losses = {
        "loss": 0,
        "loss_predict": 0,
        "loss_velocity": 0,
        "loss_vertex": 0,
    }
    train_total_num = 0
    for data in tqdm(train_dataloader, desc="Train (epoch {})".format(epoch), total=len(train_dataloader)):
        optimizer.zero_grad()
        curr_batch_size = len(data.waveform)

        losses = random_noise_loss(said_model, data, std, device, prediction_type)

        loss = losses.predict + weight_vel * losses.velocity
        if losses.vertex is not None:
            loss += weight_vertex * losses.vertex

        loss.backward()
        torch.nn.utils.clip_grad_norm_(said_model.parameters(), 1.0)
        optimizer.step()
        if ema_model:
            ema_model.step(said_model.parameters())
        lr_scheduler.step()

        train_total_losses["loss"] += loss.item() * curr_batch_size
        train_total_losses["loss_predict"] += losses.predict.item() * curr_batch_size
        train_total_losses["loss_velocity"] += losses.velocity.item() * curr_batch_size
        if losses.vertex is not None:
            train_total_losses["loss_vertex"] += losses.vertex.item() * curr_batch_size

        train_total_num += curr_batch_size

    train_avg_losses = LossEpochOutput(
        total=train_total_losses["loss"] / train_total_num,
        predict=train_total_losses["loss_predict"] / train_total_num,
        velocity=train_total_losses["loss_velocity"] / train_total_num,
        vertex=train_total_losses["loss_vertex"] / train_total_num,
        lr=lr_scheduler.get_last_lr()[0],
    )

    return train_avg_losses


def validate_epoch(
    said_model: SAID,
    val_dataloader: DataLoader,
    std: Optional[torch.FloatTensor],
    weight_vel: float,
    weight_vertex: float,
    prediction_type: str = "epsilon",
    num_repeat: int = 1,
) -> LossEpochOutput:
    """Validate the SAiD model one epoch.

    Parameters
    ----------
    said_model : SAID
        SAiD model object
    val_dataloader : DataLoader
        Dataloader of the BlendVOCAValDataset
    std : torch.FloatTensor
        (1, x_dim), Standard deviation of coefficients
    weight_vel: float
        Weight for the velocity loss
    weight_vertex: float
        Weight for the vertex loss
    prediction_type: str
        Prediction type of the scheduler function, "epsilon", "sample", or "v_prediction", by default "epsilon"
    num_repeat : int, optional
        Number of the repetition, by default 1

    Returns
    -------
    LossEpochOutput
        Average losses
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    said_model.to(device)

    if std is not None:
        std = std.to(device)

    said_model.eval()

    val_total_losses = {
        "loss": 0,
        "loss_predict": 0,
        "loss_velocity": 0,
        "loss_vertex": 0,
    }
    val_total_num = 0
    with torch.no_grad():
        for data in tqdm(val_dataloader, desc="Validation batch", total=len(val_dataloader)):
            curr_batch_size = len(data.waveform)
            losses = random_noise_loss(
                said_model, data, std, device, prediction_type
            )

            loss = losses.predict + weight_vel * losses.velocity
            if losses.vertex is not None:
                loss += weight_vertex * losses.vertex

            val_total_losses["loss"] += loss.item() * curr_batch_size
            val_total_losses["loss_predict"] += (
                losses.predict.item() * curr_batch_size
            )
            val_total_losses["loss_velocity"] += (
                losses.velocity.item() * curr_batch_size
            )
            if losses.vertex is not None:
                val_total_losses["loss_vertex"] += (
                    losses.vertex.item() * curr_batch_size
                )

            val_total_num += curr_batch_size

    val_avg_losses = LossEpochOutput(
        total=val_total_losses["loss"] / val_total_num,
        predict=val_total_losses["loss_predict"] / val_total_num,
        velocity=val_total_losses["loss_velocity"] / val_total_num,
        vertex=val_total_losses["loss_vertex"] / val_total_num,
    )

    return val_avg_losses


def main() -> None:
    """Main function"""
    default_data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"

    # Arguments
    parser = argparse.ArgumentParser(
        description="Train the SAiD model using BlendVOCA dataset"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="../BlendVOCA/audio",
        help="Directory of the audio data",
    )
    parser.add_argument(
        "--coeffs_dir",
        type=str,
        default="../BlendVOCA/blendshape_coeffs",
        help="Directory of the blendshape coefficients data",
    )
    parser.add_argument(
        "--coeffs_std_path",
        type=str,
        default="",  # default_data_dir / "coeffs_std.csv",
        help="Path of the coeffs std data",
    )
    parser.add_argument(
        "--blendshape_residuals_path",
        type=str,
        default="",  # default_data_dir / "blendshape_residuals.pickle",
        help="Path of the blendshape residuals",
    )
    parser.add_argument(
        "--landmarks_path",
        type=str,
        default="",  # default_data_dir / "FLAME_head_landmarks.txt",
        help="Path of the landmarks data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../output",
        help="Directory of the outputs",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        help="Prediction type of the scheduler function, 'epsilon', 'sample', or 'v_prediction'",
    )
    parser.add_argument(
        "--window_size_min",
        type=int,
        default=25,
        help="Minimum window size of the blendshape coefficients sequence at training",
    )
    parser.add_argument(
        "--batch_size", type=int, default=24, help="Batch size at training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10000, help="The number of epochs"
    )
    parser.add_argument(
        "--num_warmup_epochs",
        type=int,
        default=2,
        help="The number of warmup epochs",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="The number of workers"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate"
    )
    parser.add_argument(
        "--uncond_prob",
        type=float,
        default=0.1,
        help="Unconditional probability of waveform (for classifier-free guidance)",
    )
    parser.add_argument(
        "--unet_feature_dim",
        type=int,
        default=-1,
        help="Dimension of the latent feature of the UNet",
    )
    parser.add_argument(
        "--weight_vel",
        type=float,
        default=1.0,
        help="Weight for the velocity loss",
    )
    parser.add_argument(
        "--weight_vertex",
        type=float,
        default=0.02,
        help="Weight for the vertex loss",
    )
    parser.add_argument(
        "--ema",
        type=bool,
        default=True,
        help="Use Exponential Moving Average of models weights",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="Ema decay rate",
    )
    parser.add_argument(
        "--val_period", type=int, default=1, help="Period of validating model"
    )
    parser.add_argument(
        "--val_repeat", type=int, default=10, help="Number of repetition of val dataset"
    )
    parser.add_argument(
        "--save_period", type=int, default=10, help="Period of saving model"
    )
    args = parser.parse_args()

    audio_dir = args.audio_dir
    coeffs_dir = args.coeffs_dir
    coeffs_std_path = args.coeffs_std_path
    blendshape_deltas_path = args.blendshape_residuals_path
    if blendshape_deltas_path == "":
        blendshape_deltas_path = None
    landmarks_path = args.landmarks_path
    if landmarks_path == "":
        landmarks_path = None

    coeffs_std = (
        None if coeffs_std_path == "" else load_blendshape_coeffs(coeffs_std_path)
    )

    output_dir = args.output_dir
    prediction_type = args.prediction_type
    window_size_min = args.window_size_min
    batch_size = args.batch_size
    epochs = args.epochs
    num_warmup_epochs = args.num_warmup_epochs
    num_workers = args.num_workers
    learning_rate = args.learning_rate
    uncond_prob = args.uncond_prob
    unet_feature_dim = args.unet_feature_dim
    weight_vel = args.weight_vel
    weight_vertex = args.weight_vertex
    ema = args.ema
    ema_decay = args.ema_decay
    val_period = args.val_period
    val_repeat = args.val_repeat
    save_period = args.save_period

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    said_model = SAID_UNet1D(
        feature_dim=unet_feature_dim,
        prediction_type=prediction_type,
    ).to(device)
    said_model.audio_encoder = ModifiedWav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base-960h"
    ).to(device)

    print(f"Audio_encoder Model loaded successfully from facebook/wav2vec2-base-960h")

    # Load data
    train_dataset = BlendVOCATrainDataset(
        audio_dir=audio_dir,
        blendshape_coeffs_dir=coeffs_dir,
        blendshape_deltas_path=blendshape_deltas_path,
        landmarks_path=landmarks_path,
        sampling_rate=said_model.sampling_rate,
        window_size_min=window_size_min,
        uncond_prob=uncond_prob,
        preload=False,
    )
    val_dataset = BlendVOCAValDataset(
        audio_dir=audio_dir,
        blendshape_coeffs_dir=coeffs_dir,
        blendshape_deltas_path=blendshape_deltas_path,
        landmarks_path=landmarks_path,
        sampling_rate=said_model.sampling_rate,
        uncond_prob=uncond_prob,
        preload=True,
    )

    train_sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=len(train_dataset),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=train_dataset.collate_fn,
        num_workers=num_workers,
        prefetch_factor=4
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=num_workers,
    )

    # Initialize the optimzier - freeze audio encoder
    for p in said_model.audio_encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        params=filter(lambda p: p.requires_grad, said_model.parameters()),
        lr=learning_rate,
    )

    num_training_steps = len(train_dataloader) * epochs
    num_warmup_steps = len(train_dataloader) * num_warmup_epochs

    lr_scheduler = get_scheduler(
        name="constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    said_model.to(device)

    # Prepare the EMA model
    ema_model = EMAModel(said_model.parameters(), decay=ema_decay) if ema else None

    # Set the progress bar
    progress_bar = tqdm(range(1, epochs + 1), desc="Epochs")

    for epoch in range(1, epochs + 1):
        # Train the model
        train_avg_losses = train_epoch(
            said_model=said_model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            std=coeffs_std,
            weight_vel=weight_vel,
            weight_vertex=weight_vertex,
            prediction_type=prediction_type,
            ema_model=ema_model,
            epoch=epoch,
        )

        # Log
        logs = {
            "Train/Total Loss": train_avg_losses.total,
            "Train/Predict Loss": train_avg_losses.predict,
            "Train/Velocity Loss": train_avg_losses.velocity,
            "Train/Vertex Loss": train_avg_losses.vertex,
            "Train/Learning Rate": train_avg_losses.lr,
        }

        # Validate the model
        if epoch % val_period == 0:
            if ema:
                ema_model.store(said_model.parameters())
                ema_model.copy_to(said_model.parameters())

            val_avg_losses = validate_epoch(
                said_model=said_model,
                val_dataloader=val_dataloader,
                std=coeffs_std,
                weight_vel=weight_vel,
                weight_vertex=weight_vertex,
                prediction_type=prediction_type,
                num_repeat=val_repeat,
            )
            # Append the log
            logs["Validation/Total Loss"] = val_avg_losses.total
            logs["Validation/Predict Loss"] = val_avg_losses.predict
            logs["Validation/Velocity Loss"] = val_avg_losses.velocity
            logs["Validation/Vertex Loss"] = val_avg_losses.vertex

            if ema:
                ema_model.restore(said_model.parameters())

        # Print logs
        progress_bar.update(1)
        progress_bar.set_postfix(**logs)

        log_str = 'Epoch: {} |'.format(epoch) + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in logs.items()])
        print('Logs: {}'.format(log_str))

        # Save the model
        if epoch % save_period == 0:
            if ema:
                ema_model.store(said_model.parameters())
                ema_model.copy_to(said_model.parameters())

            torch.save(
                said_model.state_dict(),
                os.path.join(output_dir, f"{epoch}.pth"),
            )

            if ema:
                ema_model.restore(said_model.parameters())


if __name__ == "__main__":
    main()


# python script/train.py --output_dir /data2/liujie/train_log/said_logs
