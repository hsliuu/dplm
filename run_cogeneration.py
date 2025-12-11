import torch
import sys

sys.modules["openfold"] = None

from byprot.models.dplm2 import DPLM2
from generate_dplm2 import initialize_generation, save_results

dplm2 = DPLM2.from_pretrained("checkpoints/dplm2.pt").cuda()

input_tokens = initialize_generation(
    task="co_generation",
    length=200,
    num_seqs=1,
    tokenizer=dplm2.tokenizer,
    device=next(dplm2.parameters()).device
)[0]

samples = dplm2.generate(input_tokens=input_tokens, max_iter=200)

# 不使用 struct_tokenizer、不保存pdb
save_results(
    outputs=samples,
    task="co_generation",
    save_dir="./generation-results/dplm2_generation_no_pdb",
    tokenizer=dplm2.tokenizer,
    struct_tokenizer=None,
    save_pdb=False
)
