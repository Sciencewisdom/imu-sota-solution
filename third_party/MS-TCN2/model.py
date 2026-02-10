#!/usr/bin/python2.7

import sys
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from loguru import logger

from eval import edit_score, f_score


class MS_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes)) for _ in range(num_R)]
        )

    def forward(self, x):
        out = self.PG(x)
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            out = R(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs

class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**i, dilation=2**i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
             nn.Conv1d(2*num_f_maps, num_f_maps, 1)
             for i in range(num_layers)

            ))


        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)

        return out

class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out
    
class MS_TCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MS_TCN, self).__init__()
        self.stage1 = SS_TCN(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SS_TCN(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SS_TCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SS_TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class Trainer:
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, dataset, split, class_weights=None, smooth_weight=0.15):
        self.model = MS_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes)
        if class_weights is not None:
            self.ce = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.smooth_weight = float(smooth_weight)

        logger.add('logs/' + dataset + "_" + split + "_{time}.log")
        logger.add(sys.stdout, colorize=True, format="{message}")

    def train(
        self,
        save_dir,
        batch_gen,
        num_epochs,
        batch_size,
        learning_rate,
        device,
        amp=False,
        grad_accum=1,
        start_epoch=0,
        resume_optimizer_state=None,
        save_every=1,
        save_opt=True,
        grad_clip=1.0,
        save_best=True,
        save_final=True,
        val_every=1,
        best_metric="val_acc",
        val_vids=None,
        actions_dict=None,
        features_path=None,
        gt_path=None,
        sample_rate=1,
    ):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        if resume_optimizer_state is not None:
            optimizer.load_state_dict(resume_optimizer_state)
        scaler = torch.cuda.amp.GradScaler(enabled=bool(amp) and str(device).startswith("cuda"))
        best_metric = str(best_metric)
        if best_metric not in ("val_acc", "f1_50"):
            best_metric = "val_acc"
        best = {"epoch": -1, "metric": best_metric, "value": -1.0}

        def _eval_full_metrics():
            if not val_vids or not actions_dict or not features_path or not gt_path:
                return None
            self.model.eval()
            correct = 0.0
            total = 0.0
            edit = 0.0
            overlap = [0.1, 0.25, 0.5]
            tp_s = np.zeros(3, dtype=np.float64)
            fp_s = np.zeros(3, dtype=np.float64)
            fn_s = np.zeros(3, dtype=np.float64)
            id_to_action = {int(v): str(k) for k, v in actions_dict.items()}
            with torch.no_grad():
                for vid in val_vids:
                    stem = vid.split(".")[0]
                    feats = np.load(str(features_path) + stem + ".npy").astype(np.float32, copy=False)
                    feats = feats[:, :: int(sample_rate)]
                    # GT labels are strings per time step
                    fptr = open(str(gt_path) + vid, "r")
                    content = fptr.read().split("\n")[:-1]
                    fptr.close()
                    y = np.asarray([int(actions_dict.get(x, 0)) for x in content], dtype=np.int64)
                    y = y[:: int(sample_rate)]
                    T = min(int(feats.shape[1]), int(y.shape[0]))
                    if T <= 0:
                        continue
                    x = torch.from_numpy(feats[:, :T]).unsqueeze(0).to(device, non_blocking=True)
                    out = self.model(x)
                    pred = torch.argmax(out[-1, 0, :, :T], dim=0).detach().cpu().numpy()
                    correct += float((pred == y[:T]).sum())
                    total += float(T)
                    recog = [id_to_action.get(int(i), "background") for i in pred.tolist()]
                    gt = content[:: int(sample_rate)][:T]
                    edit += float(edit_score(recog, gt))
                    for s in range(len(overlap)):
                        tp1, fp1, fn1 = f_score(recog, gt, overlap[s])
                        tp_s[s] += tp1
                        fp_s[s] += fp1
                        fn_s[s] += fn1
            self.model.train()
            if not total:
                return {"val_acc": 0.0, "edit": 0.0, "f1_10": 0.0, "f1_25": 0.0, "f1_50": 0.0}
            val_acc = float(correct / total)
            edit = float(edit / max(1, len(val_vids)))
            f1s = []
            for s in range(len(overlap)):
                denom_p = float(tp_s[s] + fp_s[s])
                denom_r = float(tp_s[s] + fn_s[s])
                precision = float(tp_s[s] / denom_p) if denom_p > 0 else 0.0
                recall = float(tp_s[s] / denom_r) if denom_r > 0 else 0.0
                denom = float(precision + recall)
                f1 = float(2.0 * (precision * recall) / denom) if denom > 0 else 0.0
                f1s.append(float(np.nan_to_num(f1) * 100.0))
            return {"val_acc": val_acc, "edit": edit, "f1_10": f1s[0], "f1_25": f1s[1], "f1_50": f1s[2]}

        for epoch in range(int(start_epoch), int(num_epochs)):
            epoch_loss = 0
            n_batches = 0
            correct = 0
            total = 0
            step = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input = batch_input.to(device, non_blocking=True)
                batch_target = batch_target.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                if step % int(grad_accum) == 0:
                    optimizer.zero_grad(set_to_none=True)

                # AMP is used only for the forward pass; keep loss in float32 for stability.
                with torch.cuda.amp.autocast(enabled=bool(amp) and str(device).startswith("cuda")):
                    predictions = self.model(batch_input)

                loss = 0.0
                n_stages = int(predictions.shape[0]) if hasattr(predictions, "shape") else len(predictions)
                for p in predictions:
                    p32 = p.float()
                    loss = loss + self.ce(
                        p32.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1)
                    )
                    loss = loss + float(self.smooth_weight) * torch.mean(
                        torch.clamp(
                            self.mse(
                                F.log_softmax(p32[:, :, 1:], dim=1),
                                F.log_softmax(p32.detach()[:, :, :-1], dim=1),
                            ),
                            min=0,
                            max=16,
                        )
                        * mask[:, :, 1:]
                    )
                loss = loss / float(max(1, int(grad_accum)))

                # Log a scale-stable loss (per stage, per grad_accum) without changing optimization dynamics.
                # MS-TCN2 has multiple refinement stages; summing can look "too large" early on.
                denom = float(max(1, int(n_stages)))
                epoch_loss += float(loss.detach().item()) / denom
                if not torch.isfinite(loss):
                    logger.info("non_finite_loss: skipping step")
                    step += 1
                    continue
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % int(grad_accum) == 0:
                    if float(grad_clip) and float(grad_clip) > 0:
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()
                step += 1
                n_batches += 1

            batch_gen.reset()
            if str(device).startswith("cuda"):
                # Reduce allocator fragmentation on long runs.
                torch.cuda.empty_cache()

            val_acc = None
            val_metrics = None
            if int(val_every) > 0 and ((epoch + 1) % int(val_every) == 0 or (epoch + 1) == int(num_epochs)):
                val_metrics = _eval_full_metrics()
                val_acc = float(val_metrics.get("val_acc", 0.0)) if val_metrics else None

            if bool(save_best) and val_metrics is not None:
                cur = float(val_metrics.get(best_metric, -1.0))
                if cur > float(best["value"]):
                    best = {"epoch": int(epoch + 1), "metric": best_metric, "value": float(cur)}
                    torch.save(self.model.state_dict(), save_dir + "/best.model")
                    Path(save_dir + "/best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

            if int(save_every) > 0 and ((epoch + 1) % int(save_every) == 0 or (epoch + 1) == int(num_epochs)):
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                if bool(save_opt):
                    torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            acc = (float(correct) / total) if total else 0.0
            avg_loss = (epoch_loss / n_batches) if n_batches else float("nan")
            if val_metrics is None:
                logger.info("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, avg_loss, acc))
            else:
                logger.info(
                    "[epoch %d]: epoch loss = %f,   acc = %f,   val_acc = %f,   edit = %.4f,   F1@0.10 = %.2f,   F1@0.25 = %.2f,   F1@0.50 = %.2f"
                    % (
                        epoch + 1,
                        avg_loss,
                        acc,
                        float(val_metrics.get("val_acc", 0.0)),
                        float(val_metrics.get("edit", 0.0)),
                        float(val_metrics.get("f1_10", 0.0)),
                        float(val_metrics.get("f1_25", 0.0)),
                        float(val_metrics.get("f1_50", 0.0)),
                    )
                )

        if bool(save_final):
            torch.save(self.model.state_dict(), save_dir + "/final.model")

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                #print vid
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
