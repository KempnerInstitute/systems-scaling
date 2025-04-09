#! /bin/bash
#SBATCH --job-name=tokenize-fineweb
#SBATCH --output=/n/holylfs06/LABS/kempner_dev/Lab/nikhilanand/systems-scaling/olmo/logs/%A_%a.log
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH -c 112
#SBATCH --time=48:30:00
#SBATCH --mem-per-cpu=1024
#SBATCH --account=kempner_dev
#SBATCH --partition=sapphire

module load python

conda activate tatm_env

if [[ -z $SLURM_GPUS_ON_NODE ]]; then
    RAY_NUM_GPUS=0
else
    RAY_NUM_GPUS=$SLURM_GPUS_ON_NODE
fi

# choose available port on the head node
head_port=`comm -23 <(seq 15000 20000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
nodes=`scontrol show hostnames $SLURM_JOB_NODELIST`
nodes_array=( $nodes )
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head node: $head_node"
echo "Head node ip: $head_node_ip"
echo "Head port: $head_port"
export RAY_HEAD_ADDR="$head_node_ip:$head_port"
echo "Head address: $RAY_HEAD_ADDR"

echo "Starting Ray head on $head_node"
srun -N 1 -n 1 -w "$head_node"  ray start --head --node-ip-address="$head_node_ip" --temp-dir /tmp/$USER/$SLURM_JOB_ID/ray \
    --port=$head_port --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $RAY_NUM_GPUS --min-worker-port 20001 --max-worker-port 30000 --block &

# wait for head node to start
sleep 5

# start ray on the rest of the nodes
worker_num=$((SLURM_NNODES - 1))
for (( i = 1; i <= worker_num; i++ )); do
    node=${nodes_array[$i]}
    echo "Starting Ray worker on $node"
    srun -N 1 -n 1 -w "$node"  ray start --address="$RAY_HEAD_ADDR" \
        --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $RAY_NUM_GPUS --min-worker-port 20001 --max-worker-port 30000 --block &
    sleep 5
done

export RAY_ADDRESS="$RAY_HEAD_ADDR"

 tatm tokenize --tokenizer KoboldAI/llama2-tokenizer --output-dir /n/holylfs06/LABS/kempner_shared/Everyone/testbed/text/fineweb-edu/tokenized/meta-llama-2/default /n/holylfs06/LABS/kempner_shared/Everyone/testbed/text/fineweb-edu