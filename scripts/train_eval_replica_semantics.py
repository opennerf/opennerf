import sys
import os
import subprocess
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datasets import replica

CONDA_DIR = '/home/fengelmann/miniconda3/envs/opennerf'
PREFIX = '/home/fengelmann/Programming/opennerf'

def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

def train_scene(scene, experiment_name):
    cmd = [os.path.join(CONDA_DIR, "bin/python"),
           os.path.join(CONDA_DIR, "lib/python3.10/site-packages/nerfstudio/scripts/train.py"),
           "opennerf",
            "--vis=wandb",  # viewer+wandb
            f"--experiment-name=replica_{scene}",
            "--viewer.num-rays-per-chunk=2048",
            "--steps-per-eval-batch=500000",
            "--steps-per-eval-image=500000",
            "--steps-per-eval-all-images=500000",
            "--max-num-iterations=30000",  # 30000
            "--pipeline.datamanager.train-num-rays-per-batch=2048",
            f"--data={PREFIX}/data/nerfstudio/replica_{scene}",
            "--output-dir={PREFIX}/outputs",
            f"--timestamp={experiment_name}"]
    subprocess.run(cmd)


def eval_scene(scene, experiment_name):
    cmd = ["/home/fengelmann/miniconda3/envs/opennerf/bin/python",
        "/home/fengelmann/Programming/opennerf/datasets/replica_semantics.py",
        "interpolate",
        "--interpolation-steps=1",
        "--pose_source=train",
        f"--load-config={PREFIX}/outputs/replica_{scene}/opennerf/{experiment_name}/config.yml",
        "--colormap-options.colormap=pca",
        f"--output_path={PREFIX}/outputs/replica_{scene}/opennerf/{experiment_name}/",
        "--rendered-output-names=rgb",
        "--eval-num-rays-per-chunk=500",
        "--downscale-factor=2",]
    subprocess.run(cmd)


def eval_semantics(experiment_name):

    pr_files = []  # predicted files
    gt_files = []  # ground truth files
    for scene in replica.scenes:
        pr_files.append(f'{PREFIX}/outputs/replica_{scene}/opennerf/{experiment_name}/semantics_{scene}.txt')
        gt_files.append(f'{PREFIX}/datasets/replica_gt_semantics/semantic_labels_{scene}.txt')
    
    confusion = np.zeros([replica.num_classes, replica.num_classes], dtype=np.ulonglong)

    print('evaluating', len(pr_files), 'scans...')
    for i in range(len(pr_files)):
        evaluate_scan(pr_files[i], gt_files[i], confusion)
        sys.stdout.write("\rscans processed: {}".format(i+1))
        sys.stdout.flush()

    class_ious = {}
    for i in range(replica.num_classes):
        label_name = replica.class_names_reduced[i]
        label_id = i
        class_ious[label_name] = get_iou(label_id, confusion)

    print('classes \t IoU \t Acc')
    print('----------------------------')
    for i in range(replica.num_classes):
        label_name = replica.class_names_reduced[i]
        print('{0:<14s}: {1:>5.2%}   {2:>6.2%}'.format(label_name, class_ious[label_name][0], class_ious[label_name][1]))

    iou_values = np.array([i[0] for i in class_ious.values()])
    acc_values = np.array([i[1] for i in class_ious.values()])
    print()
    print(f'mIoU: \t {np.mean(iou_values):.2%}')
    print(f'mAcc: \t {np.mean(acc_values):.2%}')
    print()
    for i, split in enumerate(['head', 'comm', 'tail']):
        print(f'{split}: \t {np.mean(iou_values[17 * i:17 * (i + 1)]):.2%}')
        print(f'{split}: \t {np.mean(acc_values[17 * i:17 * (i + 1)]):.2%}')
        print('---')

def evaluate_scan(pr_file, gt_file, confusion):

    pr_ids = np.array(process_txt(pr_file), dtype=np.int64)
    gt_file_contents = np.array(process_txt(gt_file)).astype(np.int64)
    gt_ids = np.vectorize(replica.map_to_reduced.get)(gt_file_contents)

    # sanity checks
    if not pr_ids.shape == gt_ids.shape:
        print(f'number of predicted values does not match number of vertices: {pr_file}')
    for (gt_val, pr_val) in zip(gt_ids, pr_ids):
        if gt_val == replica.num_classes:
            continue
        confusion[gt_val][pr_val] += 1


def get_iou(label_id, confusion):
    tp = np.longlong(confusion[label_id, label_id])
    fn = np.longlong(confusion[label_id, :].sum()) - tp
    fp = np.longlong(confusion[:, label_id].sum()) - tp
    denom = float(tp + fp + fn)
    if denom == 0:
        return float('nan')
    iou = tp / denom
    acc = tp / float(tp + fn)
    return (iou, acc)


def main():
    experiment_name = 'run_0'
    for scene in replica.scenes:
        train_scene(scene, experiment_name)
        eval_scene(scene, experiment_name)
    eval_semantics(experiment_name)

if __name__ == "__main__":
    main()
