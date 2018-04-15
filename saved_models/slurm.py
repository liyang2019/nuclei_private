import os


class Config:
    def __init__(self):
        self.job_id = 0
        self.model = None
        self.model_folder = None
        self.test_augment_mode = 'none'
        self.identifier = None

    def __str__(self):
        rep = "a" + jobid_to_string(self.job_id) + \
              "_2018-4-15" + \
              "_" + self.model_folder + \
              "_" + self.model + \
              "_" + self.identifier
        return rep


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


def submit(job_dir, config):
    os.makedirs(job_dir, exist_ok=True)
    slurm_file_name = os.path.join(job_dir, "a" + str(config.job_id) + ".slurm")
    with open(slurm_file_name, 'a') as f:
        print("#!/bin/bash", file=f)
        print("#SBATCH --partition=commons", file=f)
        print("#SBATCH --nodes=1", file=f)
        print("#SBATCH --ntasks-per-node=1", file=f)
        # print("#SBATCH --gres=gpu:1", file=f)  # use gpu
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
        print("cd ../mask_rcnn", file=f)
        print("./build_lib.sh", file=f)
        print("cd - ", file=f)
        print("", file=f)

        print("srun python main.py " +
              "--initial_checkpoint " + os.path.join(config.model_folder, config.model) + " " +
              "--test_augment_mode " + config.test_augment_mode,
              file=f)

    os.chmod(slurm_file_name, mode=777)
    os.system("sbatch -D " + job_dir + " " + slurm_file_name)
    config.job_id += 1  # every time submit, job id plus 1


def main():
    cfg = Config()
    cfg.job_id = 0
    cfg.model_folder = 'models_gray690'
    for cfg.model in ['00027500_model.pth', '00029000_model.pth', '00031500_model.pth', '00032500_model.pth', '00035000_model.pth']:
        for cfg.test_augment_mode in ['scaleup', 'scaledown', 'hflip', 'vflip', 'none', 'blur']:
            submit(str(cfg), cfg)
            cfg.job_id += 1

    print("number of jobs: ", cfg.job_id)


if __name__ == '__main__':
    main()