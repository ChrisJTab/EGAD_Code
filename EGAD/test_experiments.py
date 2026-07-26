import os
import sys
import subprocess

benchmark = "tpcc"
database = "epic"
num_warehouses = 1
skew_factor = 0.0
fullread = "false"
cpu_exec_num_threads = 32
num_epochs = 300
num_txns = 100000
split_fields = "false"
commutative_ops = "false"
num_records = 20000000 # 20000000
exec_device = "gpu"
exec_mode = "hybrid_staging" # gpu_only, cpu_only, hybrid_staging
num_repeat = 1
overlap_flush = "false"

epic_driver_path = "./build/epic_driver"
epic_micro_driver_path = "./build/micro_driver"
output_path = "./epic_output"
cmd_template = "{} -b {} -d {} -w {} -a {} -r {} -c {} -e {} -s {} -f {} -m {} -n {} -x {} -y {} -z {}"
output_file_template = "output__b{}__d{}__w{}__a{}__r{}__c{}__e{}__s{}__f{}__m{}__n{}__x{}__y{}__z{}__r{}.txt"
err_file_template = "output__b{}__d{}__w{}__a{}__r{}__c{}__e{}__s{}__f{}__m{}__n{}__x{}__y{}__z{}__r{}.err"

micro_cmd_template = "{} -b {} -d {} -w {} -a {} -r {} -c {} -e {} -s {} -f {} -m {} -n {} -x {} -y {} -z {} -p {}"
micro_output_file_template = "output__b{}__d{}__w{}__a{}__r{}__c{}__e{}__s{}__f{}__m{}__n{}__x{}__y{}__z{}__p{}__r{}.txt"
micro_err_file_template = "output__b{}__d{}__w{}__a{}__r{}__c{}__e{}__s{}__f{}__m{}__n{}__x{}__y{}__z{}__p{}__r{}.err"




def run_with_numa(cmd, *, node=0, cores_only=True, threads=12, extra_env=None):
    # Drop page cache before each run for consistent memory bandwidth
    subprocess.run("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'", shell=True, capture_output=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OMP_DYNAMIC"]   = "false"
    env["OMP_NUM_THREADS"] = str(threads)

    if cores_only:
        env["OMP_PLACES"]    = "cores"
        env["OMP_PROC_BIND"] = "close"
        # node0 physical cores are 0-11 on your machine
        cpu_list = "0-11" if node == 0 else "12-23"
    else:
        env["OMP_PLACES"]    = "threads"
        env["OMP_PROC_BIND"] = "spread"
        # node0 threads are 0-11,24-35 ; node1 threads are 12-23,36-47
        cpu_list = "0-11,24-35" if node == 0 else "12-23,36-47"

    if extra_env:
        env.update(extra_env)

    numa_prefix = f"numactl --physcpubind={cpu_list} --membind={node}" # should be --preferred=0  instead of membind
    full_cmd = f"{numa_prefix} {cmd}"

    return subprocess.run(full_cmd, capture_output=True, text=True, shell=True, env=env)


def print_experiment_count():
    print_experiment_count.count += 1
    print("experiment count: ", print_experiment_count.count)


print_experiment_count.count = 0


def epic_ycsb_experiment():
    database = "epic"

    for split_fields in ["false"]:
        for benchmark in ["ycsbf"]:
            for skew_factor in [0.99]:
                for repeat in range(0, num_repeat):
                    print_experiment_count()
                    cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                              fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                              commutative_ops, num_records, exec_device, exec_mode, overlap_flush)
                    print(cmd)
                    command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                    print(command_output.stdout)
                    print(command_output.stderr, file=sys.stderr)
                    output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, overlap_flush, repeat)
                    err_filename = err_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, overlap_flush, repeat)
                    output_filepath = os.path.join(output_path, output_filename)
                    with open(output_filepath, "w") as output_file:
                        output_file.write(command_output.stdout)
                    err_filepath = os.path.join(output_path, err_filename)
                    with open(err_filepath, "w") as err_file:
                        err_file.write(command_output.stderr)

def epic_ycsb_hybrid_stager_experiment():
    database = "epic"

    for split_fields in ["false"]:
        for benchmark in ["ycsbf"]:
            for skew_factor in [0.01, 0.25, 0.50, 0.75, 0.99]:
                for repeat in range(0, num_repeat):
                    print_experiment_count()
                    cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                              fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                              commutative_ops, num_records, exec_device, exec_mode, overlap_flush)
                    print(cmd)
                    command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                    print(command_output.stdout)
                    print(command_output.stderr, file=sys.stderr)
                    output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, overlap_flush, repeat)
                    err_filename = err_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                            fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                            split_fields, commutative_ops, num_records,
                                                            exec_device, exec_mode, overlap_flush, repeat)
                    output_filepath = os.path.join(output_path, output_filename)
                    with open(output_filepath, "w") as output_file:
                        output_file.write(command_output.stdout)
                    err_filepath = os.path.join(output_path, err_filename)
                    with open(err_filepath, "w") as err_file:
                        err_file.write(command_output.stderr)

def epic_ycsb_hybrid_stager_experiment_split():
    database = "epic"

    for split_fields in ["true"]:
        for benchmark in ["ycsbf"]:
            for skew_factor in [0.25]:
                for repeat in range(0, num_repeat):
                    print_experiment_count()
                    cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                              fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                              commutative_ops, num_records, exec_device, exec_mode, overlap_flush)
                    print(cmd)
                    command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                    print(command_output.stdout)
                    print(command_output.stderr, file=sys.stderr)
                    output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, overlap_flush, repeat)
                    err_filename = err_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                            fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                            split_fields, commutative_ops, num_records,
                                                            exec_device, exec_mode, overlap_flush, repeat)
                    output_filepath = os.path.join(output_path, output_filename)
                    with open(output_filepath, "w") as output_file:
                        output_file.write(command_output.stdout)
                    err_filepath = os.path.join(output_path, err_filename)
                    with open(err_filepath, "w") as err_file:
                        err_file.write(command_output.stderr)

def epic_ycsb_hybrid_stager_experiment_split_numa1():
    database = "epic"

    for split_fields in ["true"]:
        for benchmark in ["ycsbf"]:
            for skew_factor in [0.25]:
                for repeat in range(0, num_repeat):
                    print_experiment_count()
                    cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                              fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                              commutative_ops, num_records, exec_device, exec_mode, overlap_flush)
                    print(cmd)

                    command_output = run_with_numa(cmd, node=0, cores_only=False, threads=24)
                    print(command_output.stdout)
                    print(command_output.stderr, file=sys.stderr)
                    output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, overlap_flush, repeat)
                    err_filename = err_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                            fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                            split_fields, commutative_ops, num_records,
                                                            exec_device, exec_mode, overlap_flush, repeat)
                    output_filepath = os.path.join(output_path, output_filename)
                    with open(output_filepath, "w") as output_file:
                        output_file.write(command_output.stdout)
                    err_filepath = os.path.join(output_path, err_filename)
                    with open(err_filepath, "w") as err_file:
                        err_file.write(command_output.stderr)

def epic_ycsb_hybrid_stager_experiment_split_numa2():
    database = "epic"

    for split_fields in ["true"]:
        for benchmark in ["ycsbf"]:
            for skew_factor in [0.25]:
                for repeat in range(0, num_repeat):
                    print_experiment_count()
                    cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                              fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                              commutative_ops, num_records, exec_device, exec_mode, overlap_flush)
                    print(cmd)

                    command_output = run_with_numa(cmd, node=1, cores_only=True, threads=12)
                    print(command_output.stdout)
                    print(command_output.stderr, file=sys.stderr)
                    output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, overlap_flush, repeat)
                    err_filename = err_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                            fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                            split_fields, commutative_ops, num_records,
                                                            exec_device, exec_mode, overlap_flush, repeat)
                    output_filepath = os.path.join(output_path, output_filename)
                    with open(output_filepath, "w") as output_file:
                        output_file.write(command_output.stdout)
                    err_filepath = os.path.join(output_path, err_filename)
                    with open(err_filepath, "w") as err_file:
                        err_file.write(command_output.stderr)

def epic_ycsb_hybrid_stager_experiment_split_numa3():
    database = "epic"

    for split_fields in ["true"]:
        for benchmark in ["ycsbf"]:
            for skew_factor in [0.25]:
                for repeat in range(0, num_repeat):
                    print_experiment_count()
                    cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                              fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                              commutative_ops, num_records, exec_device, exec_mode, overlap_flush)
                    print(cmd)

                    command_output = run_with_numa(cmd, node=0, cores_only=True, threads=12)
                    print(command_output.stdout)
                    print(command_output.stderr, file=sys.stderr)
                    output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, overlap_flush, repeat)
                    err_filename = err_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                            fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                            split_fields, commutative_ops, num_records,
                                                            exec_device, exec_mode, overlap_flush, repeat)
                    output_filepath = os.path.join(output_path, output_filename)
                    with open(output_filepath, "w") as output_file:
                        output_file.write(command_output.stdout)
                    err_filepath = os.path.join(output_path, err_filename)
                    with open(err_filepath, "w") as err_file:
                        err_file.write(command_output.stderr)

def epic_ycsb_hybrid_stager_experiment_split_numa3_overlap():
    database = "epic"
    overlap_flush = "true"

    for split_fields in ["true"]:
        for benchmark in ["ycsbf"]:
            for skew_factor in [0.25]:
                for repeat in range(0, num_repeat):
                    print_experiment_count()
                    cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                              fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                              commutative_ops, num_records, exec_device, exec_mode, overlap_flush)
                    print(cmd)

                    command_output = run_with_numa(cmd, node=0, cores_only=True, threads=12,
                                                  )
                    print(command_output.stdout)
                    print(command_output.stderr, file=sys.stderr)
                    output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, overlap_flush, repeat)
                    err_filename = err_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                            fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                            split_fields, commutative_ops, num_records,
                                                            exec_device, exec_mode, overlap_flush, repeat)
                    output_filepath = os.path.join(output_path, output_filename)
                    with open(output_filepath, "w") as output_file:
                        output_file.write(command_output.stdout)
                    err_filepath = os.path.join(output_path, err_filename)
                    with open(err_filepath, "w") as err_file:
                        err_file.write(command_output.stderr)

def epic_ycsb_scatter_delay_sweep():
    database = "epic"
    overlap_flush = "true"

    for delay_us in [0, 2000, 3500, 4500, 5000, 5500, 6000, 7000]:
        for split_fields in ["true"]:
            for benchmark in ["ycsbf"]:
                for skew_factor in [0.25]:
                    for repeat in range(0, num_repeat):
                        print_experiment_count()
                        print(f"=== SCATTER_DELAY_US={delay_us} ===")
                        cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                                  commutative_ops, num_records, exec_device, exec_mode, overlap_flush)
                        print(cmd)

                        command_output = run_with_numa(cmd, node=0, cores_only=True, threads=12,
                                                       extra_env={"SCATTER_DELAY_US": str(delay_us)})
                        print(command_output.stdout)
                        print(command_output.stderr, file=sys.stderr)
                        output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                      fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                      split_fields, commutative_ops, num_records,
                                                                      exec_device, exec_mode, overlap_flush, repeat)
                        output_filepath = os.path.join(output_path, output_filename)
                        with open(output_filepath, "w") as output_file:
                            output_file.write(command_output.stdout)

def epic_ycsb_hybrid_stager_sync_vs_async_all_skews():
    database = "epic"

    for split_fields in ["true"]:
        for benchmark in ["ycsbf"]:
            for overlap_flush in ["false", "true"]:
                for skew_factor in [0.25]:
                    for repeat in range(0, num_repeat):
                        print_experiment_count()
                        cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                                  commutative_ops, num_records, exec_device, exec_mode, overlap_flush)
                        print(cmd)

                        command_output = run_with_numa(cmd, node=0, cores_only=True, threads=12)
                        print(command_output.stdout)
                        print(command_output.stderr, file=sys.stderr)
                        output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                      fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                      split_fields, commutative_ops, num_records,
                                                                      exec_device, exec_mode, overlap_flush, repeat)
                        err_filename = err_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                split_fields, commutative_ops, num_records,
                                                                exec_device, exec_mode, overlap_flush, repeat)
                        output_filepath = os.path.join(output_path, output_filename)
                        with open(output_filepath, "w") as output_file:
                            output_file.write(command_output.stdout)
                        err_filepath = os.path.join(output_path, err_filename)
                        with open(err_filepath, "w") as err_file:
                            err_file.write(command_output.stderr)

def epic_tpcc_experiment():
    database = "epic"
    benchmark = "tpcc"

    for num_warehouses in [64]:
        for repeat in range(0, num_repeat):
            print_experiment_count()
            cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                      fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                      commutative_ops, num_records, exec_device, exec_mode, overlap_flush)
            print(cmd)
            command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            print(command_output.stdout)
            print(command_output.stderr, file=sys.stderr)
            output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                          fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                          split_fields, commutative_ops, num_records,
                                                          exec_device, exec_mode, overlap_flush, repeat)
            err_filename = err_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, overlap_flush, repeat)
            output_filepath = os.path.join(output_path, output_filename)
            with open(output_filepath, "w") as output_file:
                output_file.write(command_output.stdout)
            err_filepath = os.path.join(output_path, err_filename)
            with open(err_filepath, "w") as err_file:
                err_file.write(command_output.stderr)

def epic_tpcc_full_experiment():
    database = "epic"
    benchmark = "tpccfull"

    for num_warehouses in [64]:
        for repeat in range(0, num_repeat):
            print_experiment_count()
            cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                      fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                      commutative_ops, num_records, exec_device, exec_mode, overlap_flush)
            print(cmd)
            command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            print(command_output.stdout)
            print(command_output.stderr, file=sys.stderr)
            output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                          fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                          split_fields, commutative_ops, num_records,
                                                          exec_device, exec_mode, overlap_flush, repeat)
            err_filename = err_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, overlap_flush, repeat)
            output_filepath = os.path.join(output_path, output_filename)
            with open(output_filepath, "w") as output_file:
                output_file.write(command_output.stdout)
            err_filepath = os.path.join(output_path, err_filename)
            with open(err_filepath, "w") as err_file:
                err_file.write(command_output.stderr)


def gacco_ycsb_experiment():
    database = "gacco"
    num_txns = 32768

    for benchmark in ["ycsbf"]:
        for skew_factor in [0.99]:
            for repeat in range(0, num_repeat):
                print_experiment_count()
                cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                          fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                          commutative_ops, num_records, exec_device, exec_mode, overlap_flush)
                print(cmd)
                command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                print(command_output.stdout)
                print(command_output.stderr, file=sys.stderr)
                output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                              fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                              split_fields, commutative_ops, num_records,
                                                              exec_device, exec_mode, overlap_flush, repeat)
                err_filename = err_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, overlap_flush, repeat)
                output_filepath = os.path.join(output_path, output_filename)
                with open(output_filepath, "w") as output_file:
                    output_file.write(command_output.stdout)
                err_filepath = os.path.join(output_path, err_filename)
                with open(err_filepath, "w") as err_file:
                    err_file.write(command_output.stderr)


def gacco_tpcc_experiment():
    database = "gacco"
    num_txns = 32768

    for benchmark in ["tpccn", "tpccp"]:
        for commutative_ops in ["true", "false"]:
            for num_warehouses in [64]:
                for repeat in range(0, num_repeat):
                    print_experiment_count()
                    cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                              fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                              commutative_ops, num_records, exec_device, exec_mode, overlap_flush)
                    print(cmd)
                    command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                    print(command_output.stdout)
                    print(command_output.stderr, file=sys.stderr)
                    output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, overlap_flush, repeat)
                    err_filename = err_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, overlap_flush, repeat)
                    output_filepath = os.path.join(output_path, output_filename)
                    with open(output_filepath, "w") as output_file:
                        output_file.write(command_output.stdout)
                    err_filepath = os.path.join(output_path, err_filename)
                    with open(err_filepath, "w") as err_file:
                        err_file.write(command_output.stderr)


def epic_cpu_tpcc_experiment():
    database = "epic"
    benchmark = "tpcc"
    exec_device = "cpu"
    exec_mode = "cpu_only"

    for num_warehouses in [64]:
        for repeat in range(0, num_repeat):
            print_experiment_count()
            cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                      fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                      commutative_ops, num_records, exec_device, exec_mode, overlap_flush)
            print(cmd)
            command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            print(command_output.stdout)
            print(command_output.stderr, file=sys.stderr)
            output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                          fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                          split_fields, commutative_ops, num_records,
                                                          exec_device, exec_mode, overlap_flush, repeat)
            err_filename = err_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, overlap_flush, repeat)
            output_filepath = os.path.join(output_path, output_filename)
            with open(output_filepath, "w") as output_file:
                output_file.write(command_output.stdout)
            err_filepath = os.path.join(output_path, err_filename)
            with open(err_filepath, "w") as err_file:
                err_file.write(command_output.stderr)


def epic_cpu_tpcc_full_experiment():
    database = "epic"
    benchmark = "tpccfull"
    exec_device = "cpu"
    exec_mode = "cpu_only"

    for num_warehouses in [64]:
        for repeat in range(0, num_repeat):
            print_experiment_count()
            cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                      fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                      commutative_ops, num_records, exec_device, exec_mode, overlap_flush)
            print(cmd)
            command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            print(command_output.stdout)
            print(command_output.stderr, file=sys.stderr)
            output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                          fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                          split_fields, commutative_ops, num_records,
                                                          exec_device, exec_mode, overlap_flush, repeat)
            err_filename = err_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, overlap_flush, repeat)
            output_filepath = os.path.join(output_path, output_filename)
            with open(output_filepath, "w") as output_file:
                output_file.write(command_output.stdout)
            err_filepath = os.path.join(output_path, err_filename)
            with open(err_filepath, "w") as err_file:
                err_file.write(command_output.stderr)


def epic_cpu_ycsb_experiment():
    database = "epic"
    exec_device = "cpu"
    exec_mode = "cpu_only"
    split_fields = "false"

    for benchmark in ["ycsbf"]:
        for skew_factor in [0.99]:
            for repeat in range(0, num_repeat):
                print_experiment_count()
                cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                          fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                          commutative_ops, num_records, exec_device, exec_mode, overlap_flush)
                print(cmd)
                command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                print(command_output.stdout)
                print(command_output.stderr, file=sys.stderr)
                output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                              fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                              split_fields, commutative_ops, num_records,
                                                              exec_device, exec_mode, overlap_flush, repeat)
                err_filename = err_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, overlap_flush, repeat)
                output_filepath = os.path.join(output_path, output_filename)
                with open(output_filepath, "w") as output_file:
                    output_file.write(command_output.stdout)
                err_filepath = os.path.join(output_path, err_filename)
                with open(err_filepath, "w") as err_file:
                    err_file.write(command_output.stderr)


def epic_ycsb_epoch_size_experiment():
    database = "epic"
    split_fields = "false"
    benchmark = "ycsbf"

    skew_factor_range = [0.99, 0.0]
    num_txns_ranges = [[70000],
                       [200000]]

    for skew_factor, num_txns_ranges in zip(skew_factor_range, num_txns_ranges):
        for num_txns in num_txns_ranges:
            for repeat in range(0, num_repeat):
                print_experiment_count()
                cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                          fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                          commutative_ops, num_records, exec_device, exec_mode, overlap_flush)
                print(cmd)
                command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                print(command_output.stdout)
                print(command_output.stderr, file=sys.stderr)
                output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                              fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                              split_fields, commutative_ops, num_records,
                                                              exec_device, exec_mode, overlap_flush, repeat)
                err_filename = err_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, overlap_flush, repeat)
                output_filepath = os.path.join(output_path, output_filename)
                with open(output_filepath, "w") as output_file:
                    output_file.write(command_output.stdout)
                err_filepath = os.path.join(output_path, err_filename)
                with open(err_filepath, "w") as err_file:
                    err_file.write(command_output.stderr)

def epic_tpcc_epoch_size_experiment():
    database = "epic"
    benchmark = "tpcc"

    num_warehouses_range = [1, 64]
    num_txns_ranges = [[40000],
                       [200000]]

    for num_warehouses, num_txns_range in zip(num_warehouses_range, num_txns_ranges):
        for num_txns in num_txns_range:
            for repeat in range(0, num_repeat):
                print_experiment_count()
                cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                          fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                          commutative_ops, num_records, exec_device, exec_mode, overlap_flush)
                print(cmd)
                command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                print(command_output.stdout)
                print(command_output.stderr, file=sys.stderr)
                output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                              fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                              split_fields, commutative_ops, num_records,
                                                              exec_device, exec_mode, overlap_flush, repeat)
                err_filename = err_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, overlap_flush, repeat)
                output_filepath = os.path.join(output_path, output_filename)
                with open(output_filepath, "w") as output_file:
                    output_file.write(command_output.stdout)
                err_filepath = os.path.join(output_path, err_filename)
                with open(err_filepath, "w") as err_file:
                    err_file.write(command_output.stderr)

def epic_microbenchmark():
    database = "epic"
    benchmark = "micro"
    skew_factor = 0.8
    for abort_rate in [50]:
        for repeat in range(0, num_repeat):
            cmd = micro_cmd_template.format(epic_micro_driver_path, benchmark, database, num_warehouses, skew_factor,
                                            fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                            commutative_ops, num_records, exec_device, exec_mode, overlap_flush, abort_rate)
            print(cmd)
            command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            print(command_output.stdout)
            print(command_output.stderr, file=sys.stderr)
            output_filename = micro_output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                split_fields, commutative_ops, num_records,
                                                                exec_device, exec_mode, overlap_flush, abort_rate, repeat)
            err_filename = micro_err_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                split_fields, commutative_ops, num_records,
                                                                exec_device, exec_mode, overlap_flush, abort_rate, repeat)
            output_filepath = os.path.join(output_path, output_filename)
            with open(output_filepath, "w") as output_file:
                output_file.write(command_output.stdout)
            err_filepath = os.path.join(output_path, err_filename)
            with open(err_filepath, "w") as err_file:
                err_file.write(command_output.stderr)

def gacco_tpcc_epoch_size_experiment():
    database = "gacco"

    num_warehouses_range = [1, 64]
    num_txns_ranges = [[4000],
                       [25000]]

    for benchmark in ["tpccn", "tpccp"]:
        for num_warehouses, num_txns_range in zip(num_warehouses_range, num_txns_ranges):
            for num_txns in num_txns_range:
                for repeat in range(0, num_repeat):
                    print_experiment_count()
                    cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                              fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                              commutative_ops, num_records, exec_device, exec_mode)
                    print(cmd)
                    command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                    print(command_output.stdout)
                    print(command_output.stderr, file=sys.stderr)
                    output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, repeat)
                    err_filename = err_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, repeat)
                    output_filepath = os.path.join(output_path, output_filename)
                    with open(output_filepath, "w") as output_file:
                        output_file.write(command_output.stdout)
                    err_filepath = os.path.join(output_path, err_filename)
                    with open(err_filepath, "w") as err_file:
                        err_file.write(command_output.stderr)

def gacco_ycsb_epoch_size_experiment():
    database = "gacco"
    split_fields = "false"
    benchmark = "ycsbf"

    skew_factor_range = [0.99, 0.0]
    num_txns_ranges = [[6000],
                       [45000]]

    for skew_factor, num_txns_ranges in zip(skew_factor_range, num_txns_ranges):
        for num_txns in num_txns_ranges:
            for repeat in range(0, num_repeat):
                print_experiment_count()
                cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                          fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                          commutative_ops, num_records, exec_device, exec_mode)
                print(cmd)
                command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                print(command_output.stdout)
                print(command_output.stderr, file=sys.stderr)
                output_filename = output_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                              fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                              split_fields, commutative_ops, num_records,
                                                              exec_device, exec_mode, repeat)
                err_filename = err_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                                  fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                                  split_fields, commutative_ops, num_records,
                                                                  exec_device, exec_mode, repeat)
                output_filepath = os.path.join(output_path, output_filename)
                with open(output_filepath, "w") as output_file:
                    output_file.write(command_output.stdout)
                err_filepath = os.path.join(output_path, err_filename)
                with open(err_filepath, "w") as err_file:
                    err_file.write(command_output.stderr)


def epic_ycsb_figure7(benchmarks=None, skews=None, repeats=3, splits=None):
    """Reproduce Figure 7 of OSDI '24 Epic paper across all YCSB workloads.

    Matches original artifact run_experiments.py config:
    - database=epic, benchmarks in {ycsba, ycsbb, ycsbc, ycsbf}
    - num_records=5,000,000 (5M — scaled from original 10M to fit 32GB GPU), num_txns=100k, num_epochs=5
    - fullread=true, exec=gpu_only, overlap_flush=false
    - split_fields in {true, false} (outer loop for both Epic and Epic+FieldSplit)

    Note: field-split (split_fields=true) currently crashes with CUDA memory errors
    in the gpuPiecewiseExecKernel on this branch — likely a regression from the
    hybrid-staging modifications. Pass splits=["false"] to skip the broken path.
    """
    database = "epic"
    _num_warehouses = 1
    _fullread = "true"
    _cpu_exec_num_threads = 32
    _num_epochs = 5
    _num_txns = 100000
    _commutative_ops = "false"
    _num_records = 5000000
    _exec_device = "gpu"
    _exec_mode = "gpu_only"
    _overlap_flush = "false"

    if benchmarks is None:
        benchmarks = ["ycsba", "ycsbb", "ycsbc", "ycsbf"]
    if skews is None:
        skews = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
    if splits is None:
        splits = ["true", "false"]

    for benchmark in benchmarks:
        for split_fields in splits:
            for skew_factor in skews:
                for repeat in range(0, repeats):
                    print_experiment_count()
                    cmd = cmd_template.format(
                        epic_driver_path, benchmark, database, _num_warehouses,
                        skew_factor, _fullread, _cpu_exec_num_threads, _num_epochs,
                        _num_txns, split_fields, _commutative_ops, _num_records,
                        _exec_device, _exec_mode, _overlap_flush)
                    print(cmd)
                    command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                    output_filename = output_file_template.format(
                        benchmark, database, _num_warehouses, skew_factor, _fullread,
                        _cpu_exec_num_threads, _num_epochs, _num_txns, split_fields,
                        _commutative_ops, _num_records, _exec_device, _exec_mode,
                        _overlap_flush, repeat)
                    err_filename = err_file_template.format(
                        benchmark, database, _num_warehouses, skew_factor, _fullread,
                        _cpu_exec_num_threads, _num_epochs, _num_txns, split_fields,
                        _commutative_ops, _num_records, _exec_device, _exec_mode,
                        _overlap_flush, repeat)
                    output_filepath = os.path.join(output_path, output_filename)
                    with open(output_filepath, "w") as output_file:
                        output_file.write(command_output.stdout)
                    err_filepath = os.path.join(output_path, err_filename)
                    with open(err_filepath, "w") as err_file:
                        err_file.write(command_output.stderr)


# Backwards-compat alias
def epic_ycsb_figure7_ycsbf(skews=None, repeats=3, splits=None):
    return epic_ycsb_figure7(benchmarks=["ycsbf"], skews=skews, repeats=repeats, splits=splits)


def _run_one_cpu_sweep(benchmark, fullread, split, num_records, num_epochs_, repeats=1, skews=None):
    """Run one benchmark on CPU (-x cpu -y cpu_only) across skews."""
    database = "epic"
    if skews is None:
        skews = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
    for skew_factor in skews:
        for repeat in range(repeats):
            print_experiment_count()
            cmd = cmd_template.format(
                epic_driver_path, benchmark, database, 1,
                skew_factor, fullread, 32, num_epochs_, 100000, split,
                "false", num_records, "cpu", "cpu_only", "false")
            print(cmd)
            command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            output_filename = output_file_template.format(
                benchmark, database, 1, skew_factor, fullread,
                32, num_epochs_, 100000, split, "false", num_records,
                "cpu", "cpu_only", "false", repeat)
            with open(os.path.join(output_path, output_filename), "w") as f:
                f.write(command_output.stdout)


def _run_one_benchmark_sweep(benchmark, fullread, split, exec_mode, overlap_flush,
                             num_records, num_epochs_, repeats=1, skews=None,
                             use_numa=False):
    """Run one benchmark across skews, optionally with NUMA pinning (like doc config)."""
    database = "epic"
    if skews is None:
        skews = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
    for skew_factor in skews:
        for repeat in range(repeats):
            print_experiment_count()
            cmd = cmd_template.format(
                epic_driver_path, benchmark, database, 1,
                skew_factor, fullread, 32, num_epochs_, 100000, split,
                "false", num_records, "gpu", exec_mode, overlap_flush)
            print(cmd)
            if use_numa:
                command_output = run_with_numa(cmd, node=0, cores_only=False, threads=12)
            else:
                command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            output_filename = output_file_template.format(
                benchmark, database, 1, skew_factor, fullread,
                32, num_epochs_, 100000, split, "false", num_records,
                "gpu", exec_mode, overlap_flush, repeat)
            with open(os.path.join(output_path, output_filename), "w") as f:
                f.write(command_output.stdout)


def _run_ycsbf_sweep(fullread, split, exec_mode, overlap_flush, repeats=3,
                     skews=None, num_records_override=5_000_000):
    """Run ycsbf across skews with a specific (fullread, split, exec_mode, overlap_flush) config."""
    database = "epic"
    benchmark = "ycsbf"
    if skews is None:
        skews = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
    for skew_factor in skews:
        for repeat in range(repeats):
            print_experiment_count()
            cmd = cmd_template.format(
                epic_driver_path, benchmark, database, 1,
                skew_factor, fullread, 32, 5, 100000, split,
                "false", num_records_override, "gpu", exec_mode, overlap_flush)
            print(cmd)
            command_output = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            output_filename = output_file_template.format(
                benchmark, database, 1, skew_factor, fullread,
                32, 5, 100000, split, "false", num_records_override,
                "gpu", exec_mode, overlap_flush, repeat)
            with open(os.path.join(output_path, output_filename), "w") as f:
                f.write(command_output.stdout)
            err_filename = err_file_template.format(
                benchmark, database, 1, skew_factor, fullread,
                32, 5, 100000, split, "false", num_records_override,
                "gpu", exec_mode, overlap_flush, repeat)
            with open(os.path.join(output_path, err_filename), "w") as f:
                f.write(command_output.stderr)


if __name__ == "__main__":
    # Check if the directory exists
    if not os.path.exists(output_path):
        # If not, create the directory
        os.makedirs(output_path)
    #epic_tpcc_full_experiment()
    #epic_cpu_tpcc_full_experiment()
    #epic_tpcc_experiment()
    #epic_cpu_tpcc_experiment()
    #epic_ycsb_experiment()
    #epic_ycsb_hybrid_stager_experiment()
    #epic_ycsb_hybrid_stager_experiment_split()
    #epic_ycsb_hybrid_stager_experiment_split_numa3()
    #epic_ycsb_hybrid_stager_experiment_split_numa3_overlap()
    #epic_ycsb_hybrid_stager_experiment_split_numa3()
    #epic_ycsb_scatter_delay_sweep()
    #epic_ycsb_hybrid_stager_experiment_split_numa3_overlap()
    #epic_ycsb_hybrid_stager_sync_vs_async_all_skews()
    #epic_ycsb_hybrid_stager_experiment_split_numa2()
    #epic_ycsb_hybrid_stager_experiment_split_numa3()
    #epic_cpu_ycsb_experiment()
    #gacco_ycsb_experiment()
    #gacco_tpcc_experiment()
    #epic_ycsb_epoch_size_experiment()
    #epic_tpcc_epoch_size_experiment()
    #epic_microbenchmark()
    #gacco_ycsb_epoch_size_experiment()
    #gacco_tpcc_epoch_size_experiment()
    # Figure 7 (OSDI '24) YCSB-F replication — set FIG7_MODE=smoke for quick 2-skew test, FIG7_MODE=full for complete sweep
    mode = os.environ.get("FIG7_MODE")
    if mode == "smoke":
        epic_ycsb_figure7_ycsbf(skews=[0.0, 0.99], repeats=1)
    elif mode == "full":
        epic_ycsb_figure7_ycsbf()
    elif mode == "full_nosplit":
        # Only split_fields=false (Epic baseline line in Figure 7). Skip field-split
        # because gpuPiecewiseExecKernel is broken on this branch.
        epic_ycsb_figure7_ycsbf(splits=["false"], repeats=3)
    elif mode == "figure7_all":
        # All 4 YCSB workloads × 8 skews × 3 repeats, non-split only
        epic_ycsb_figure7(splits=["false"], repeats=3)
    elif mode == "figure7_split":
        # All 4 YCSB workloads × 8 skews × 3 repeats, split_fields=true ONLY
        # (non-split data already exists from figure7_all run)
        epic_ycsb_figure7(splits=["true"], repeats=3)
    elif mode == "figure7_split_ycsbf":
        # YCSB-F only × 8 skews × 3 repeats, split_fields=true
        epic_ycsb_figure7_ycsbf(splits=["true"], repeats=3)
    elif mode == "figure7_everything":
        # Full re-run: 4 workloads × 8 skews × 2 splits × 3 repeats = 192 runs
        epic_ycsb_figure7(splits=["false", "true"], repeats=3)
    elif mode == "ycsbf_rfalse_ftrue_gpu":
        # YCSB-F, -r false -f true, gpu_only, 5M records, 8 skews × 3 repeats
        _run_ycsbf_sweep(fullread="false", split="true", exec_mode="gpu_only",
                         overlap_flush="false", repeats=3)
    elif mode == "ycsbf_rfalse_ftrue_hybrid":
        # YCSB-F, -r false -f true, hybrid_staging, 5M records, 8 skews × 3 repeats
        _run_ycsbf_sweep(fullread="false", split="true", exec_mode="hybrid_staging",
                         overlap_flush="true", repeats=3)
    elif mode == "gpu_rfalse_ftrue_abc":
        # YCSB-A/B/C at gpu_only -r false -f true 5M 5 epochs (YCSB-F already done)
        for bench in ["ycsba", "ycsbb", "ycsbc"]:
            _run_one_benchmark_sweep(bench, fullread="false", split="true",
                                     exec_mode="gpu_only", overlap_flush="false",
                                     num_records=5_000_000, num_epochs_=5, repeats=1)
    elif mode == "cpu_only_all_configs":
        # All 4 YCSB × all 4 (r,f) combos × 8 skews × 1 repeat = 128 runs on CPU
        # 5M records, 5 epochs, no NUMA pinning (to match gpu_only runs)
        for bench in ["ycsba", "ycsbb", "ycsbc", "ycsbf"]:
            for r_flag in ["true", "false"]:
                for f_flag in ["false", "true"]:
                    _run_one_cpu_sweep(bench, fullread=r_flag, split=f_flag,
                                       num_records=5_000_000, num_epochs_=5, repeats=1)
    elif mode == "hybrid_async_20m_all":
        # All 4 YCSB benchmarks at hybrid_staging, 20M records, 300 epochs, -r false -f true
        # Matches the cpu-baseline campaign config for apples-to-apples with existing numbers.
        # Uses NUMA pinning --physcpubind=0-11,24-35 --membind=0, OMP_NUM_THREADS=12
        for bench in ["ycsba", "ycsbb", "ycsbc", "ycsbf"]:
            _run_one_benchmark_sweep(bench, fullread="false", split="true",
                                     exec_mode="hybrid_staging", overlap_flush="true",
                                     num_records=20_000_000, num_epochs_=300, repeats=1,
                                     use_numa=True)
    elif mode == "figure7_split_abc":
        # YCSB-A, B, C × 8 skews × 3 repeats, split_fields=true only
        # (YCSB-F already done in figure7_split_ycsbf)
        epic_ycsb_figure7(benchmarks=["ycsba", "ycsbb", "ycsbc"],
                          splits=["true"], repeats=3)
