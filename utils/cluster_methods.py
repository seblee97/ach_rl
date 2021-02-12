def create_job_script(run_command: str,
                      save_path: str,
                      num_cpus: int,
                      conda_env_name: str,
                      memory: int,
                      error_path: str,
                      output_path: str,
                      checkpoint_path: str,
                      array_job_length: int = 0,
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
        if array_job_length:
            file.write(f"#PBS -J 1-{array_job_length}\n")
        # define job id variable without array index
        file.write('JOBID=${PBS_JOBID%"[$PBS_ARRAY_INDEX].pbs"}\n')
        # output/error file paths
        # file.write(f"#PBS -e {checkpoint_path}/error.txt\n")
        # file.write(f"#PBS -o {checkpoint_path}/output.txt\n")
        # log job id info
        file.write('echo "PBS_JOBID is ${PBS_JOBID}"\n')
        file.write('echo "PBS_ARRAY_INDEX value is ${PBS_ARRAY_INDEX}"\n')
        file.write('echo "PBS_JOBID without ARRAY_INDEX is $JOBID"\n')
        # initialise conda env
        file.write("module load anaconda3/personal\n")
        file.write(f"source activate {conda_env_name}\n")
        # change to dir where job was submitted from
        file.write("cd $PBS_O_WORKDIR\n")
        # job script
        file.write(f"{run_command}\n")
        # copy error/output files to permanent
        file.write(f'ls $PBS_O_WORKDIR/\n')
        file.write(f'ls \n')
        file.write(f'ls $TMPDIR/\n')
        file.write(
            f"mv $PBS_O_WORKDIR/$PBS_JOBNAME.e$JOBID.$PBS_ARRAY_INDEX {checkpoint_path}/\n"
        )
        file.write(
            f"mv $PBS_O_WORKDIR/$PBS_JOBNAME.o$JOBID.$PBS_ARRAY_INDEX {checkpoint_path}/\n"
        )
