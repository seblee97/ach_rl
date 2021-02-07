def create_job_script(run_command: str,
                      save_path: str,
                      num_cpus: int,
                      conda_env_name: str,
                      memory: int,
                      error_path: str,
                      output_path: str,
                      walltime: str = "24:0:0") -> None:
    """Create a job script for use on HPC.

    Args:
            run_command: main script command, e.g. 'python run.py'
            save_path: path to save the job script to
            num_cpus: number of cores for job
            conda_env_name: name of conda environment to activate for job
            memory: number of gb memory to allocate to node.
            walltime: time to give job--1 day by default
    """
    with open(save_path, 'w') as file:
        file.write(f"#PBS -lselect=1:ncpus={num_cpus}:mem={memory}gb\n")
        file.write(f"#PBS -lwalltime={walltime}\n")
        # output/error file paths
        file.write(f"#PBS -e {error_path}\n")
        file.write(f"#PBS -o {output_path}\n")
        # initialise conda env
        file.write("module load anaconda3/personal\n")
        file.write(f"source activate {conda_env_name}\n")
        # change to dir where job was submitted from
        file.write("cd $PBS_O_WORKDIR\n")
        # job script
        file.write(f"{run_command}\n")
        # move files back to permanent
        #file.write("mkdir $WORK/$PBS_JOBID\n")
        #file.write("cp * $WORK/$PBS_JOBID\n")
