"""
For submit jobs on Rice Univ. research computing cluster.
"""
import os


class Config:
    def __init__(self):
        self.job_id = 0
        self.learning_rate = 0.001
        self.input_width = 256
        self.input_height = 256
        self.train_split = 'train1_ids_gray2_500'
        self.val_split = 'valid1_ids_gray2_43'
        self.batch_size = 1

    def __str__(self):
        rep = 'a' + jobid_to_string(self.job_id) + \
              '_2018-3-21' + \
              '_lr' + str(self.learning_rate) + \
              '_W' + str(self.input_width) + \
              '_H' + str(self.input_height) + \
              '_bc' + str(self.batch_size) + \
              '_' + self.train_split + \
              '_' + self.val_split
        return rep


def submit(job_dir, config):
    os.makedirs(job_dir, exist_ok=True)
    slurm_file_name = os.path.join(job_dir, "a" + str(config.job_id) + ".slurm")
    with open(slurm_file_name, 'a') as f:
        print("#!/bin/bash", file=f)
        print("#SBATCH --partition=commons", file=f)
        print("#SBATCH --nodes=1", file=f)
        print("#SBATCH --ntasks-per-node=1", file=f)
        print("#SBATCH --gres=gpu:1", file=f)  # use gpu
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
        print("cd.. /../net/lib/", file=f)
        print("./build_lib.sh", file=f)
        print("cd - ")
        print("")

        print("srun python main.py " +
              "--learning_rate " + str(config.learning_rate) + " ",
              "--input_width " + str(config.input_width) + " ",
              "--input_height " + str(config.input_height) + " ",
              "--train_split " + config.train_split + " ",
              "--val_split " + config.val_split + " ",
              "--batch_size " + str(config.batch_size) + " ",
              "--is_validation ",
              file=f)

    os.chmod(slurm_file_name, mode=777)
    os.system("sbatch -D " + job_dir + " " + slurm_file_name)
    config.job_id += 1  # every time submit, job id plus 1


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
    cfg.job_id = 0
    cfg.learning_rate = 0.001
    cfg.input_height = 256
    cfg.input_width = 256

    cfg.train_split = 'train1_ids_gray2_500'
    submit(os.path.join("rice_cluster_result", str(cfg)), cfg)

    cfg.train_split = 'purple_108'
    submit(os.path.join("rice_cluster_result", str(cfg)), cfg)

    print("number of jobs: ", cfg.job_id)


if __name__ == '__main__':
    main()
