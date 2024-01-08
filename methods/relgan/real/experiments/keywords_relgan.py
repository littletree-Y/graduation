import os
from subprocess import call
import subprocess
import sys, time
from datetime import datetime

# Job id and gpu_id
if len(sys.argv) > 2:
    job_id = int(sys.argv[1])
    gpu_id = str(sys.argv[2])
    print('job_id: {}, gpu_id: {}'.format(job_id, gpu_id))
elif len(sys.argv) > 1:
    job_id = int(sys.argv[1])
    gpu_id = '0'
    print('job_id: {}, missing gpu_id (use default {})'.format(job_id, gpu_id))
else:
    print('Missing argument: job_id and gpu_id.')
    quit()

# Executables
executable = 'python3'

# Arguments
architecture = ['rmc_keywords', 'rmc_vanilla', 'rmc_vanilla', 'rmc_vanilla', 'rmc_vanilla', 'rmc_vanilla', 'rmc_vanilla', 'rmc_vanilla', 'rmc_vanilla']
gantype =      ['RSGAN', 'RSGAN', 'RSGAN', 'RSGAN', 'RSGAN', 'RSGAN', 'RSGAN', 'RSGAN', 'RSGAN']
opt_type =     ['adam', 'adam', 'adam', 'adam', 'adam', 'adam', 'adam', 'adam', 'adam']
temperature =  ['100', '100', '100', '100', '100', '1000', '1000', '1000', '1000']
d_lr =         ['1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4']
gadv_lr =      ['1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4']
mem_slots =    ['1', '1', '1', '1', '1', '1', '1', '1', '1']
head_size =    ['256', '256', '256', '256', '256', '256', '256', '256', '256']
num_heads =    ['2', '2', '2', '2', '2', '2', '2', '2', '2']
seed =         ['171', '171', '172', '173', '174', '179', '176', '177', '178']

# bs = '64'
bs = '16'
gpre_lr = '1e-2'
hidden_dim = '32'
seq_len = '20'
dataset = 'four'

gsteps = '1'
dsteps = '5'
gen_emb_dim = '32'
dis_emb_dim = '64'
num_rep = '64'
sn = False
decay = False
adapt = 'exp'
# npre_epochs = '150'
# nadv_steps = '2000'
npre_epochs = '30'
nadv_steps = '40'
ntest = '10'
pre_keywords_weight = '5.0'
adv_keywords_weight = '1.0'
# npre_epochs = '1'
# nadv_steps = '1'
# ntest = '1'


# Paths
rootdir = '../..'
scriptname = 'run.py'
cwd = os.path.dirname(os.path.abspath(__file__))

# 获取当前时间
current_time = datetime.now()

# 格式化时间到分钟，格式为 YYYY-MM-DD HH:MM
formatted_time = current_time.strftime('%Y-%m-%d %H:%M')

outdir = os.path.join(cwd, 'out', time.strftime("%Y%m%d"), dataset,
                      '{}_{}_{}_{}_bs{}_sl{}_sn{}_dec{}_ad-{}_npre{}_nadv{}_ms{}_hs{}_nh{}_ds{}_dlr{}_glr{}_tem{}_demb{}_nrep{}_hdim{}_sd{}_preweight{}_advweight{}_{}'.
                      format(dataset, architecture[job_id], gantype[job_id], opt_type[job_id], bs, seq_len, int(sn),
                             int(decay), adapt, npre_epochs, nadv_steps, mem_slots[job_id], head_size[job_id],
                             num_heads[job_id], dsteps, d_lr[job_id], gadv_lr[job_id], temperature[job_id], dis_emb_dim,
                             num_rep, hidden_dim, seed[job_id], pre_keywords_weight, adv_keywords_weight, formatted_time))

args = [
    # Architecture
    '--gf-dim', '64',
    '--df-dim', '64',
    '--g-architecture', architecture[job_id],
    '--d-architecture', architecture[job_id],
    '--gan-type', gantype[job_id],
    '--hidden-dim', hidden_dim,

    # Training
    '--gsteps', gsteps,
    '--dsteps', dsteps,
    '--npre-epochs', npre_epochs,
    '--nadv-steps', nadv_steps,
    '--ntest', ntest,
    '--d-lr', d_lr[job_id],
    '--gpre-lr', gpre_lr,
    '--gadv-lr', gadv_lr[job_id],
    '--batch-size', bs,
    '--log-dir', os.path.join(outdir, 'tf_logs'),
    '--sample-dir', os.path.join(outdir, 'samples'),
    '--optimizer', opt_type[job_id],
    '--seed', seed[job_id],
    '--temperature', temperature[job_id],
    '--adapt', adapt,
    '--pre-keywords-weight', pre_keywords_weight,
    '--adv-keywords-weight', adv_keywords_weight,

    # evaluation
    '--nll-gen',
    '--bleu',
    # '--selfbleu',
    # '--doc-embsim',
    '--singlebleu',

    # relational memory
    '--mem-slots', mem_slots[job_id],
    '--head-size', head_size[job_id],
    '--num-heads', num_heads[job_id],

    # dataset
    '--dataset', dataset,
    '--vocab-size', '5000',
    '--start-token', '0',
    '--seq-len', seq_len,
    '--num-sentences', '10',  # how many generated sentences to use per item
    '--gen-emb-dim', gen_emb_dim,
    '--dis-emb-dim', dis_emb_dim,
    '--num-rep', num_rep,
    '--data-dir', './data'
]

if sn:
    args += ['--sn']
if decay:
    args += ['--decay']

# Run
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
my_env = os.environ.copy()

# call([executable, scriptname] + args, env=my_env, cwd=rootdir)

call_args = [executable, '-u', scriptname] + args

# 通过Popen更好地控制进程执行和输出处理
with subprocess.Popen(
    call_args,
    stdout=subprocess.PIPE,  # 将stdout重定向到管道
    stderr=subprocess.STDOUT,  # 将stderr也重定向到stdout
    bufsize=0,  # 设置缓冲区大小为0，无缓冲
    env=my_env,
    cwd=rootdir,
    universal_newlines=True  # 也可以用universal_newlines=True，它在 Python 3.7+ 已经被废弃
) as proc:

    # 实时打印输出
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()  # 正确使用 flush 来确保输出被写出