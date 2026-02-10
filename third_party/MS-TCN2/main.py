#!/usr/bin/python2.7

import os

# CUDA allocator knobs: helps long-running training avoid fragmentation OOM.
# Keep to options supported by the installed torch version.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch
from model import Trainer
from batch_gen import BatchGenerator
import argparse
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NOTE: we expose seed via CLI to support multi-seed ensembling.
seed = 1538574472
# Prefer throughput over strict determinism.
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="gtea")
parser.add_argument('--split', default='1')
parser.add_argument('--seed', default=str(seed), type=int, help='Random seed for python/torch.')
parser.add_argument(
    '--exp',
    default='',
    type=str,
    help='Optional experiment subdir name. If set, overrides model/results subdir (under models/<dataset>/ and results/<dataset>/).',
)

parser.add_argument('--features_dim', default='2048', type=int)
parser.add_argument('--bz', default='1', type=int)
parser.add_argument('--lr', default='0.0005', type=float)
parser.add_argument('--preload', action='store_true', help='Preload all features/labels into RAM (faster).')
parser.add_argument('--amp', action='store_true', help='Use mixed precision training on CUDA.')
parser.add_argument('--grad_accum', default='1', type=int, help='Gradient accumulation steps.')
parser.add_argument('--resume', action='store_true', help='Resume from latest epoch in model_dir (if present).')
parser.add_argument('--chunk_len', default='0', type=int, help='Train on random chunks of this length (0 disables).')
parser.add_argument('--fg_min_ratio', default='0.0', type=float, help='Min foreground ratio in sampled chunk.')
parser.add_argument('--smooth_weight', default='0.15', type=float, help='Temporal smoothing regularizer weight.')
parser.add_argument('--save_every', default='1', type=int, help='Save model checkpoint every N epochs.')
parser.add_argument('--save_opt', default='1', type=int, help='Whether to save optimizer state (1/0).')
parser.add_argument('--grad_clip', default='1.0', type=float, help='Gradient clipping norm (0 disables).')
parser.add_argument('--save_best', default='1', type=int, help='Save best checkpoint on test split (1/0).')
parser.add_argument('--save_final', default='1', type=int, help='Save final checkpoint (1/0).')
parser.add_argument('--val_every', default='1', type=int, help='Validate on test split every N epochs (0 disables).')
parser.add_argument(
    '--best_metric',
    default='val_acc',
    choices=['val_acc', 'f1_50'],
    help='Metric used to select best checkpoint on test split.',
)


parser.add_argument('--num_f_maps', default='64', type=int)

# Need input
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--num_layers_PG', type=int)
parser.add_argument('--num_layers_R', type=int)
parser.add_argument('--num_R', type=int)

args = parser.parse_args()

seed = int(args.seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

num_epochs = args.num_epochs
features_dim = args.features_dim
bz = args.bz
lr = args.lr

num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R
num_f_maps = args.num_f_maps

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

vid_list_file = "./data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = "./data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
features_path = "./data/"+args.dataset+"/features/"
gt_path = "./data/"+args.dataset+"/groundTruth/"

mapping_file = "./data/"+args.dataset+"/mapping.txt"

exp_name = (str(args.exp).strip() or ("split_" + str(args.split)))
model_dir = "./models/" + args.dataset + "/" + exp_name
results_dir = "./results/" + args.dataset + "/" + exp_name

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)
if args.action == "train":
    background_id = int(actions_dict.get("background", 0))
    batch_gen = BatchGenerator(
        num_classes,
        actions_dict,
        gt_path,
        features_path,
        sample_rate,
        preload=args.preload,
        chunk_len=int(args.chunk_len),
        fg_min_ratio=float(args.fg_min_ratio),
        background_id=background_id,
    )
    batch_gen.read_data(vid_list_file)

    # Compute class weights on the training split to counter background dominance.
    # Keep background down-weighted to improve foreground recall (segment-F1).
    import numpy as np
    counts = np.zeros((num_classes,), dtype=np.int64)
    if args.preload:
        for vid in batch_gen.list_of_examples:
            y = batch_gen._tgt_cache.get(vid)
            if y is None:
                continue
            y = y[y >= 0]
            if y.size:
                binc = np.bincount(y.astype(np.int64, copy=False), minlength=num_classes)
                counts[: binc.shape[0]] += binc
    else:
        for vid in batch_gen.list_of_examples:
            fp = open(gt_path + vid, 'r')
            content = fp.read().split('\n')[:-1]
            fp.close()
            ys = [int(actions_dict.get(x, background_id)) for x in content]
            if ys:
                binc = np.bincount(np.asarray(ys, dtype=np.int64), minlength=num_classes)
                counts[: binc.shape[0]] += binc
    total = int(counts.sum())
    w = np.ones((num_classes,), dtype=np.float32)
    for i in range(num_classes):
        if counts[i] > 0:
            w[i] = float(total / (num_classes * counts[i]))
    # Background downweight.
    w[background_id] = min(float(w[background_id]), 0.4)
    w = np.clip(w, 0.05, 10.0)
    class_weights = torch.tensor(w, dtype=torch.float32, device=device)

    trainer = Trainer(
        num_layers_PG,
        num_layers_R,
        num_R,
        num_f_maps,
        features_dim,
        num_classes,
        args.dataset,
        args.split,
        class_weights=class_weights,
        smooth_weight=float(args.smooth_weight),
    )
    # Keep CLI stable: do not auto-enable AMP. Use --amp explicitly if desired.

    start_epoch = 0
    opt_state = None
    if args.resume:
        # Find latest epoch-N.model / epoch-N.opt
        latest = -1
        for fn in os.listdir(model_dir):
            if fn.startswith("epoch-") and fn.endswith(".model"):
                try:
                    n = int(fn.split("-")[1].split(".")[0])
                except Exception:
                    continue
                latest = max(latest, n)
        if latest > 0:
            mpath = os.path.join(model_dir, f"epoch-{latest}.model")
            opath = os.path.join(model_dir, f"epoch-{latest}.opt")
            trainer.model.load_state_dict(torch.load(mpath, map_location="cpu"))
            if os.path.exists(opath):
                opt_state = torch.load(opath, map_location="cpu")
            start_epoch = latest

    # Prepare validation list for save_best.
    file_ptr = open(vid_list_file_tst, 'r')
    val_vids = file_ptr.read().split('\n')[:-1]
    file_ptr.close()

    trainer.train(
        model_dir,
        batch_gen,
        num_epochs=num_epochs,
        batch_size=bz,
        learning_rate=lr,
        device=device,
        amp=bool(args.amp),
        grad_accum=int(args.grad_accum),
        start_epoch=int(start_epoch),
        resume_optimizer_state=opt_state,
        save_every=int(args.save_every),
        save_opt=bool(int(args.save_opt)),
        grad_clip=float(args.grad_clip),
        save_best=bool(int(args.save_best)),
        save_final=bool(int(args.save_final)),
        val_every=int(args.val_every),
        best_metric=str(args.best_metric),
        val_vids=val_vids,
        actions_dict=actions_dict,
        features_path=features_path,
        gt_path=gt_path,
        sample_rate=sample_rate,
    )

if args.action == "predict":
    trainer = Trainer(
        num_layers_PG,
        num_layers_R,
        num_R,
        num_f_maps,
        features_dim,
        num_classes,
        args.dataset,
        args.split,
    )
    trainer.predict(
        model_dir,
        results_dir,
        features_path,
        vid_list_file_tst,
        num_epochs,
        actions_dict,
        device,
        sample_rate,
    )
