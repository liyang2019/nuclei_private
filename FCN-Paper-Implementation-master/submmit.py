"""
For submit jobs on Rice Univ. research computing cluster.
"""
import numpy as np
import os


class Config:
    def __init__(self):
        self.job_id = 0
        self.n_sites = 80
        self.h = 1.0
        self.delta = 5
        self.alpha = 2
        self.n_sample = 1000
        self.learning_rate = 0.0001
        self.hamiltonian = 'TFI'
        self.dtype = 'float'


def submit(job_dir, config):
    os.makedirs(job_dir, exist_ok=True)
    slurm_file_name = os.path.join(job_dir, "b" + str(config.job_id) + ".slurm")
    with open(slurm_file_name, 'a') as f:
        print("#!/bin/bash", file=f)
        print("#SBATCH --partition=commons", file=f)
        print("#SBATCH --nodes=1", file=f)
        print("#SBATCH --ntasks-per-node=1", file=f)
        print("#SBATCH --mem-per-cpu=4000m", file=f)
        print("#SBATCH --time=1-00:00:00", file=f)
        print("#SBATCH --mail-user=li.yang.pbs@gmail.com", file=f)
        print("#SBATCH --mail-type=ALL", file=f)
        print("#SBATCH --export=ALL", file=f)
        print("", file=f)
        print("echo 'My job ran on:'", file=f)
        print("echo $SLURM_NODELIST", file=f)
        print("echo 'submission dir:' $SLURM_SUBMIT_DIR", file=f)
        print("cd $SLURM_SUBMIT_DIR", file=f)
        print("", file=f)
        print("python main.py " +
              "--n_sites " + str(config.n_sites) + " " +
              "--output_dir " + job_dir + " " +
              "--transverse_field " + str(config.h) + " " +
              "--pbc "
              "--delta " + str(config.delta) + " " +
              "--alpha " + str(config.alpha) + " " +
              "--n_iter 10000 " +
              "--n_sample " + str(config.n_sample) + " " +
              "--learning_rate " + str(config.learning_rate) + " " +
              "--lambda 0.0 " +
              "--hamiltonian " + config.hamiltonian + " " +
              "--dtype " + config.dtype, file=f)
    os.chmod(slurm_file_name, mode=777)
    os.system("sbatch -D " + job_dir + " " + slurm_file_name)


def parse_dir(config):
    if config.job_id < 10:
        JID = '00' + str(config.job_id)
    elif config.job_id < 100:
        JID = '0' + str(config.job_id)
    elif config.job_id < 1000:
        JID = str(config.job_id)
    else:
        raise Exception
    return os.path.join("result",
                        "a" + JID + "_2018-2-26" +
                        "_" + config.hamiltonian +
                        "_N" + str(config.n_sites) +
                        # "_h{:.2f}".format(config.h) +
                        "_delta" + str(config.delta) +
                        "_alpha" + str(config.alpha) +
                        "_Nsam" + str(config.n_sample) +
                        "_lr" + str(config.learning_rate) +
                        "_" + config.dtype +
                        "_SR")


def main():
    cfg = Config()
    cfg.hamiltonian = 'Heisenberg'
    # 2018-3-7
    # n_sample
    for cfg.n_sites in [10, 50, 100]:
        for cfg.delta in [5]:
            for cfg.alpha in [2]:
                for cfg.n_sample in [1000]:
                    for cfg.learning_rate in [0.01, 0.001, 0.0001]:
                        for cfg.dtype in ['float', 'complex']:
                            submit(parse_dir(cfg), cfg)
                            cfg.job_id += 1
    # # n_sites
    # for cfg.n_sites in [5, 10, 50, 100, 200, 250]:
    #     for cfg.delta in [5]:
    #         for cfg.alpha in [2]:
    #             for cfg.n_sample in [1000]:
    #                 for cfg.learning_rate in [0.001]:
    #                     submit(parse_dir(cfg), cfg)
    #                     cfg.job_id += 1
    #
    # # delta
    # for cfg.n_sites in [100]:
    #     for cfg.delta in [3, 5, 7, 9, 11]:
    #         for cfg.alpha in [2]:
    #             for cfg.n_sample in [1000]:
    #                 for cfg.learning_rate in [0.001]:
    #                     submit(parse_dir(cfg), cfg)
    #                     cfg.job_id += 1
    #
    # # alpha
    # for cfg.n_sites in [100]:
    #     for cfg.delta in [5]:
    #         for cfg.alpha in [1, 2, 4, 8]:
    #             for cfg.n_sample in [1000]:
    #                 for cfg.learning_rate in [0.001]:
    #                     submit(parse_dir(cfg), cfg)
    #                     cfg.job_id += 1
    #
    # # n_sample
    # for cfg.n_sites in [100]:
    #     for cfg.delta in [5]:
    #         for cfg.alpha in [2]:
    #             for cfg.n_sample in [1000, 2000, 5000]:
    #                 for cfg.learning_rate in [0.001]:
    #                     submit(parse_dir(cfg), cfg)
    #                     cfg.job_id += 1
    #
    #     # n_sample
    #     for cfg.n_sites in [100]:
    #         for cfg.delta in [5]:
    #             for cfg.alpha in [2]:
    #                 for cfg.n_sample in [1000]:
    #                     for cfg.learning_rate in [0.1, 0.01, 0.001, 0.0001]:
    #                         submit(parse_dir(cfg), cfg)
    #                         cfg.job_id += 1

    # # n_sites
    # for cfg.n_sites in [50, 100, 200, 250]:
    #     for cfg.h in [1.5]:
    #         for cfg.delta in [5]:
    #             for cfg.alpha in [2]:
    #                 for cfg.n_sample in [1000]:
    #                     for cfg.learning_rate in [0.0001]:
    #                         submit(parse_dir(cfg), cfg)
    #                         cfg.job_id += 1

    # # h
    # for cfg.n_sites in [100]:
    #     for cfg.h in np.linspace(0, 2, 11):
    #         for cfg.delta in [5]:
    #             for cfg.alpha in [2]:
    #                 for cfg.n_sample in [1000]:
    #                     for cfg.learning_rate in [0.0001]:
    #                         submit(parse_dir(cfg), cfg)
    #                         cfg.job_id += 1
    # # delta
    # for cfg.n_sites in [100]:
    #     for cfg.h in [1.5]:
    #         for cfg.delta in [3, 5, 7, 9, 11]:
    #             for cfg.alpha in [2]:
    #                 for cfg.n_sample in [1000]:
    #                     for cfg.learning_rate in [0.0001]:
    #                         submit(parse_dir(cfg), cfg)
    #                         cfg.job_id += 1
    #
    # # alpha
    # for cfg.n_sites in [100]:
    #     for cfg.h in [1.5]:
    #         for cfg.delta in [5]:
    #             for cfg.alpha in [1, 2, 4, 8]:
    #                 for cfg.n_sample in [1000]:
    #                     for cfg.learning_rate in [0.0001]:
    #                         submit(parse_dir(cfg), cfg)
    #                         cfg.job_id += 1
    #
    # # n_sample
    # for cfg.n_sites in [100]:
    #     for cfg.h in [1.5]:
    #         for cfg.delta in [5]:
    #             for cfg.alpha in [2]:
    #                 for cfg.n_sample in [1000, 2000, 5000]:
    #                     for cfg.learning_rate in [0.0001]:
    #                         submit(parse_dir(cfg), cfg)
    #                         cfg.job_id += 1

    # # learning rate
    # for cfg.n_sites in [100]:
    #     for cfg.h in [1.5]:
    #         for cfg.delta in [5]:
    #             for cfg.alpha in [2]:
    #                 for cfg.n_sample in [1000]:
    #                     for cfg.learning_rate in [0.001, 0.0001]:
    #                         submit(parse_dir(cfg), cfg)
    #                         cfg.job_id += 1

    print("number of jobs: ", cfg.job_id)


if __name__ == '__main__':
    main()
