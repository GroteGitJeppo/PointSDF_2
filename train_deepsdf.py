#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
"""
DeepSDF autodecoder training for PointSDF_2 — aligned with corepp/train_deep_sdf.py:
  - Same per-scene loop order, L1 + latent regularisation, DataParallel, checkpoints
  - Data: SDFSceneDataset (CSV splits + samples.npz) instead of deepsdf SDFSamples
  - Config: YAML (--config) instead of specs.json (--experiment)

Outputs under <output_dir>/[<run_tag>/]:
  ModelParameters/*.pth  OptimizerParameters/*.pth  LatentCodes/*.pth  Logs.pth  config.yaml
"""

import argparse
import json
import logging
import math
import os
import random
import signal
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml

from data.sdf_scene_dataset import SDFSceneDataset
from models.decoder import Decoder
from models import SDFDecoder

# ---------------------------------------------------------------------------
# Workspace layout (same names as corepp/deepsdf/deep_sdf/workspace.py)
# ---------------------------------------------------------------------------

MODEL_PARAMS_SUBDIR = "ModelParameters"
OPTIMIZER_PARAMS_SUBDIR = "OptimizerParameters"
LATENT_CODES_SUBDIR = "LatentCodes"
LOGS_FILENAME = "Logs.pth"


def get_model_params_dir(experiment_dir, create=False):
    d = os.path.join(experiment_dir, MODEL_PARAMS_SUBDIR)
    if create and not os.path.isdir(d):
        os.makedirs(d)
    return d


def get_optimizer_params_dir(experiment_dir, create=False):
    d = os.path.join(experiment_dir, OPTIMIZER_PARAMS_SUBDIR)
    if create and not os.path.isdir(d):
        os.makedirs(d)
    return d


def get_latent_codes_dir(experiment_dir, create=False):
    d = os.path.join(experiment_dir, LATENT_CODES_SUBDIR)
    if create and not os.path.isdir(d):
        os.makedirs(d)
    return d


def load_model_parameters(experiment_directory, checkpoint, decoder):
    filename = os.path.join(
        experiment_directory, MODEL_PARAMS_SUBDIR, checkpoint + ".pth"
    )
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'model state dict "{filename}" does not exist')
    data = torch.load(filename, map_location="cpu")
    decoder.load_state_dict(data["model_state_dict"])
    return data["epoch"]


# ---------------------------------------------------------------------------
# Repelling loss (unused in main loop; kept for parity with corepp)
# ---------------------------------------------------------------------------


def repelling_loss(latents, thresh=5):
    r_loss = torch.cdist(latents, latents)
    r_loss = thresh - r_loss
    r_loss = F.relu(r_loss)
    r_loss = torch.sum(r_loss) - thresh * latents.shape[0]
    return r_loss


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        raise NotImplementedError


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(schedule_specs):
    schedules = []
    for s in schedule_specs:
        t = s["Type"]
        if t == "Step":
            schedules.append(
                StepLearningRateSchedule(s["Initial"], s["Interval"], s["Factor"])
            )
        elif t == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(s["Initial"], s["Final"], s["Length"])
            )
        elif t == "Constant":
            schedules.append(ConstantLearningRateSchedule(s["Value"]))
        else:
            raise ValueError(f'Unknown learning rate schedule Type: "{t}"')
    return schedules


def save_model(experiment_directory, filename, decoder, epoch):
    model_params_dir = get_model_params_dir(experiment_directory, True)
    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):
    optimizer_params_dir = get_optimizer_params_dir(experiment_directory, True)
    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):
    full_filename = os.path.join(get_optimizer_params_dir(experiment_directory), filename)
    if not os.path.isfile(full_filename):
        raise FileNotFoundError(f'optimizer state dict "{full_filename}" does not exist')
    data = torch.load(full_filename, map_location="cpu")
    optimizer.load_state_dict(data["optimizer_state_dict"])
    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):
    latent_codes_dir = get_latent_codes_dir(experiment_directory, True)
    all_latents = latent_vec.state_dict()
    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


def load_latent_vectors(experiment_directory, filename, lat_vecs):
    full_filename = os.path.join(get_latent_codes_dir(experiment_directory), filename)
    if not os.path.isfile(full_filename):
        raise FileNotFoundError(f'latent state file "{full_filename}" does not exist')
    data = torch.load(full_filename, map_location="cpu")
    if isinstance(data["latent_codes"], torch.Tensor):
        if lat_vecs.num_embeddings != data["latent_codes"].size(0):
            raise ValueError("num latent codes mismatched")
        if lat_vecs.embedding_dim != data["latent_codes"].size(2):
            raise ValueError("latent code dimensionality mismatch")
        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec
    else:
        lat_vecs.load_state_dict(data["latent_codes"])
    return data["epoch"]


def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):
    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, LOGS_FILENAME),
    )


def load_logs(experiment_directory):
    full_filename = os.path.join(experiment_directory, LOGS_FILENAME)
    if not os.path.isfile(full_filename):
        raise FileNotFoundError(f'log file "{full_filename}" does not exist')
    data = torch.load(full_filename, map_location="cpu")
    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):
    iters_per_epoch = len(loss_log) // len(lr_log)
    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]
    return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_spec_with_default(cfg, key, default):
    return cfg[key] if key in cfg and cfg[key] is not None else default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log:
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def build_decoder(cfg):
    """Match corepp NetworkSpecs when use_facebook_decoder_specs is true."""
    if cfg.get("use_facebook_decoder_specs", False):
        ns = cfg["network_specs"]
        return Decoder(cfg["latent_size"], **ns).cuda()
    return SDFDecoder(
        latent_size=cfg["latent_size"],
        num_layers=cfg["num_layers"],
        inner_dim=cfg["inner_dim"],
        skip_connections=cfg.get("skip_connections", True),
        dropout_prob=float(cfg.get("dropout_prob", 0.2)),
        weight_norm=cfg.get("weight_norm", True),
    ).cuda()


def main_function(cfg, continue_from, batch_split):
    experiment_directory = cfg["_experiment_directory"]

    seed = cfg.get("seed", 133)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    logging.debug("running %s", experiment_directory)

    lr_specs = cfg["learning_rate_schedule"]
    if len(lr_specs) != 2:
        raise ValueError("learning_rate_schedule must have exactly 2 entries (model, latent)")
    lr_schedules = get_learning_rate_schedules(lr_specs)

    num_epochs = int(cfg["epochs"])
    snapshot_frequency = int(cfg.get("snapshot_frequency", 50))
    checkpoints = list(
        range(snapshot_frequency, num_epochs + 1, snapshot_frequency)
    )
    for ep in cfg.get("additional_snapshots", []):
        checkpoints.append(ep)
    checkpoints = sorted(set(checkpoints))

    grad_clip = get_spec_with_default(cfg, "gradient_clip_norm", None)
    if grad_clip is not None:
        grad_clip = float(grad_clip)
        logging.debug("clipping gradients to max norm %s", grad_clip)

    num_samp_per_scene = int(cfg["samples_per_scene"])

    clamp_dist = float(cfg.get("clamp_value", cfg.get("clamping_distance", 0.1)))
    min_t = -clamp_dist
    max_t = clamp_dist
    enforce_minmax = bool(cfg.get("enforce_minmax", False))

    do_code_regularization = get_spec_with_default(cfg, "code_regularization", True)
    do_code_regularization_sphere = get_spec_with_default(
        cfg, "code_regularization_sphere", False
    )
    code_reg_lambda = float(get_spec_with_default(cfg, "code_regularization_lambda", 1e-4))
    reg_ramp_epochs = int(get_spec_with_default(cfg, "reg_ramp_epochs", 100))

    decoder = build_decoder(cfg)
    logging.info(decoder)
    logging.info("training with %s GPU(s)", torch.cuda.device_count())

    decoder = torch.nn.DataParallel(decoder)

    log_frequency = int(cfg.get("log_frequency", 10))

    stage_splits = cfg.get("stage1_splits", ["train", "val"])
    clamp_val = cfg["clamp_value"] if cfg.get("clamp", False) else None

    sdf_dataset = SDFSceneDataset(
        cfg["sdf_data_dir"],
        cfg["splits_csv"],
        split=stage_splits,
        samples_per_scene=num_samp_per_scene,
        clamp_value=clamp_val,
    )

    num_scenes = len(sdf_dataset)
    logging.info("There are %s scenes", num_scenes)

    latent_size = int(cfg["latent_size"])
    code_bound = cfg.get("code_bound", None)
    max_norm = float(code_bound) if code_bound is not None else None

    lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=max_norm)
    init_std = float(cfg.get("code_init_std_dev", 1.0))
    torch.nn.init.normal_(lat_vecs.weight.data, 0.0, init_std / math.sqrt(latent_size))

    logging.debug(
        "initialized with mean magnitude %s",
        get_mean_latent_vector_magnitude(lat_vecs),
    )

    loss_l1 = torch.nn.L1Loss(reduction="sum")

    optimizer_all = optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )

    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    start_epoch = 1

    if continue_from is not None:
        logging.info('continuing from "%s"', continue_from)

        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + ".pth", lat_vecs
        )
        model_epoch = load_model_parameters(
            experiment_directory, continue_from, decoder
        )
        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + ".pth", optimizer_all
        )

        loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = load_logs(
            experiment_directory
        )

        if log_epoch != model_epoch:
            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
            )

        if not (model_epoch == optimizer_epoch == lat_epoch):
            raise RuntimeError(
                f"epoch mismatch: {model_epoch} vs {optimizer_epoch} vs {lat_epoch} vs {log_epoch}"
            )

        start_epoch = model_epoch + 1
        logging.debug("loaded")

    logging.info("starting from epoch %s", start_epoch)

    logging.info(
        "Number of decoder parameters: %s",
        sum(p.data.nelement() for p in decoder.parameters()),
    )
    logging.info(
        "Number of shape code parameters: %s (# codes %s, code dim %s)",
        lat_vecs.num_embeddings * lat_vecs.embedding_dim,
        lat_vecs.num_embeddings,
        lat_vecs.embedding_dim,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DeepSDF training expects CUDA (corepp uses .cuda() throughout).")

    lat_vecs = lat_vecs.to(device)

    def save_latest(epoch):
        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)

    def save_checkpoints(epoch):
        save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)

    def adjust_learning_rate(schedules, optimizer, epoch):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = schedules[i].get_learning_rate(epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        decoder.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        for scene_idx in range(len(sdf_dataset)):
            sample = sdf_dataset[scene_idx]

            sdf_data = sample["sdf_data"]
            latent_idx = int(sample["latent_idx"])

            sdf_data = sdf_data.reshape(-1, 4)
            num_sdf_samples = sdf_data.shape[0]
            sdf_data.requires_grad_(False)

            xyz = sdf_data[:, 0:3].to(device, dtype=torch.float32)
            sdf_gt = sdf_data[:, 3].unsqueeze(1).to(device, dtype=torch.float32)

            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, min_t, max_t)

            xyz = torch.chunk(xyz, batch_split)

            indices = torch.full(
                (num_sdf_samples,), latent_idx, dtype=torch.long, device=device
            )
            indices = torch.chunk(indices, batch_split)
            sdf_gt = torch.chunk(sdf_gt, batch_split)

            batch_loss = 0.0
            optimizer_all.zero_grad()

            for bi in range(batch_split):
                batch_vecs = lat_vecs(indices[bi]).to(device, dtype=torch.float32)
                net_input = torch.cat([batch_vecs, xyz[bi]], dim=1)
                pred_sdf = decoder(net_input)

                if enforce_minmax:
                    pred_sdf = torch.clamp(pred_sdf, min_t, max_t)

                chunk_loss = loss_l1(pred_sdf, sdf_gt[bi]) / num_sdf_samples

                if do_code_regularization:
                    l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                    reg_loss = (
                        code_reg_lambda * min(1, epoch / reg_ramp_epochs) * l2_size_loss
                    ) / num_sdf_samples
                    chunk_loss = chunk_loss + reg_loss

                if do_code_regularization_sphere:
                    sphere_loss_reg = torch.abs(1 - torch.norm(batch_vecs, dim=1)).sum()
                    reg_loss = (
                        code_reg_lambda * min(1, epoch / reg_ramp_epochs) * sphere_loss_reg
                    ) / num_sdf_samples
                    chunk_loss = chunk_loss + reg_loss

                chunk_loss.backward()
                batch_loss += chunk_loss.item()

            lr = lr_schedules[0].get_learning_rate(epoch)
            logging.info("epoch %s, loss = %s, lr = %s", epoch, batch_loss, lr)
            loss_log.append(batch_loss)

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            optimizer_all.step()

        elapsed = time.time() - t0
        timing_log.append(elapsed)

        lr_log.append([s.get_learning_rate(epoch) for s in lr_schedules])

        lat_mag_log.append(get_mean_latent_vector_magnitude(lat_vecs))

        append_parameter_magnitudes(param_mag_log, decoder)

        if epoch in checkpoints:
            save_checkpoints(epoch)

        if log_frequency > 0 and epoch % log_frequency == 0:
            save_latest(epoch)
            save_logs(
                experiment_directory,
                loss_log,
                lr_log,
                timing_log,
                lat_mag_log,
                param_mag_log,
                epoch,
            )

        logging.info(
            "Epoch %s/%s finished in %.1fs",
            epoch,
            num_epochs,
            elapsed,
        )

    # Per-label tensors for Stage 2 (PointCloudLatentDataset / train_encoder)
    latent_save_dir = os.path.join(experiment_directory, "latent_codes")
    os.makedirs(latent_save_dir, exist_ok=True)
    for label, idx in sdf_dataset.label_to_idx.items():
        torch.save(
            lat_vecs.weight.data[idx].cpu(),
            os.path.join(latent_save_dir, f"{label}.pth"),
        )
    logging.info("Per-label latents for Stage 2: %s", latent_save_dir)

    print("latent: \n", lat_vecs.weight)


def _configure_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a DeepSDF autodecoder (PointSDF_2, corepp-compatible loop)"
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML config (see configs/train_deepsdf.yaml)",
    )
    parser.add_argument(
        "--continue-from",
        default=None,
        metavar="NAME",
        help="Resume from checkpoint NAME (e.g. latest or 500), without .pth",
    )
    parser.add_argument(
        "--batch_split",
        type=int,
        default=1,
        help="Split each scene's samples into this many chunks (gradient accumulation).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()
    _configure_logging(args.verbose)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if cfg.get("timestamp_run_dir", True):
        run_tag = datetime.now().strftime("%d_%m_%H%M%S")
        experiment_directory = os.path.join(cfg["output_dir"], run_tag)
    else:
        experiment_directory = cfg["output_dir"]

    os.makedirs(experiment_directory, exist_ok=True)
    cfg["_experiment_directory"] = experiment_directory

    with open(os.path.join(experiment_directory, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    # Optional copy for interoperability with tools that expect specs.json
    specs_json = {
        "Description": cfg.get("description", "PointSDF_2 DeepSDF"),
        "CodeLength": cfg["latent_size"],
        "NumEpochs": cfg["epochs"],
        "SamplesPerScene": cfg["samples_per_scene"],
        "LearningRateSchedule": cfg["learning_rate_schedule"],
    }
    with open(os.path.join(experiment_directory, "specs.json"), "w") as f:
        json.dump(specs_json, f, indent=2)

    main_function(cfg, args.continue_from, int(args.batch_split))
