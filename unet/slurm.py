"""
For submit jobs on Rice Univ. research computing cluster.
"""
import os


class Config:
    def __init__(self):
        self.job_id = 0
        self.learning_rate = 0.001
        self.unet_channels = 32
        self.dropout = 0
        self.optimizer = 'adam'
        self.print_every = 10
        self.save_model_every = 100
        self.crop_size = 224
        self.train_set = 'purple_108'
        self.valid_set = 'valid1_ids_gray2_43'
        self.batch_size = 1

    def __str__(self):
        rep = "a" + jobid_to_string(self.job_id) + \
              "_2018-3-21" + \
              "_lr" + str(self.learning_rate) + \
              "_ch" + str(self.unet_channels) + \
              "_dropout" + str(self.dropout) + \
              "_optimizer" + self.optimizer + \
              "_size" + str(self.crop_size) + \
              "_" + self.train_set
        return rep


def submit(job_dir, config):
    os.makedirs(job_dir, exist_ok=True)
    slurm_file_name = os.path.join(job_dir, "a" + str(config.job_id) + ".slurm")
    with open(slurm_file_name, 'a') as f:
        print("#!/bin/bash", file=f)
        print("#SBATCH --partition=commons", file=f)
        print("#SBATCH --nodes=1", file=f)
        print("#SBATCH --ntasks-per-node=1", file=f)
        print("#SBATCH --gres=gpu:1", file=f)
        print("#SBATCH --mem-per-cpu=4000m", file=f)
        print("#SBATCH --time=1-00:00:00", file=f)
        print("#SBATCH --mail-user=li.yang.pbs@gmail.com", file=f)
        print("#SBATCH --mail-type=ALL", file=f)
        print("#SBATCH --export=ALL", file=f)
        print("", file=f)
        print("echo 'My job ran on:'", file=f)
        print("echo $SLURM_NODELIST", file=f)
        print("echo 'submission dir:' $SLURM_SUBMIT_DIR", file=f)
        print("echo modules: ", file=f)
        print("module list", file=f)
        print("cd $SLURM_SUBMIT_DIR", file=f)
        print("", file=f)
        print("module purge", file=f)
        print("module load GCC/6.4.0  CUDA/7.5.18  OpenMPI/2.1.1 PyTorch/0.1.12", file=f)
        print("", file=f)
        print("srun python main.py " +
              "--not_debug " +
              "--use_gpu " +
              "--print_every " + str(config.print_every) + " " +
              "--save_model_every " + str(config.save_model_every) + " " +
              "--unet_batch_norm " +
              "--crop_size " + str(config.crop_size) + " ",
              "--train_set " + config.train_set + " ",
              file=f)

    os.chmod(slurm_file_name, mode=777)
    os.system("sbatch -D " + job_dir + " " + slurm_file_name)


def jobid_to_string(job_id):
    if job_id < 10:
        JID = '00' + str(job_id)
    elif job_id < 100:
        JID = '0' + str(job_id)
    elif job_id < 1000:
        JID = str(job_id)
    else:
        raise Exception
    return JID


def main():
    cfg = Config()
    cfg.job_id = 1
    cfg.learning_rate = 0.001
    cfg.print_every = 10
    cfg.save_model_every = 100
    cfg.optimizer = 'adam'
    cfg.crop_size = 256
    cfg.train_set = 'purple_108'
    cfg.batch_size = 4
    cfg.unet_channels = 64
    submit(os.path.join("rice_cluster_result", str(cfg)), cfg)

    print("number of jobs: ", cfg.job_id)


if __name__ == '__main__':
    main()
