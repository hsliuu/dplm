from byprot.models.dplm2 import MultimodalDiffusionProteinLanguageModel as DPLM2
dplm2 = DPLM2.from_pretrained("airkingbd/dplm2_650m").cuda()

#co-generation
from generate_dplm2 import initialize_generation, save_results

input_tokens = initialize_generation(
  task="co_generation",
  length=200,
  num_seqs=5,
  tokenizer=dplm2.tokenizer,
  device=next(dplm2.parameters()).device
)[0]

samples = dplm2.generate(
  input_tokens=input_tokens,
  max_iter=500,
)
save_results(
    outputs=samples,
    task="co_generation",
    save_dir="./generation-results/dplm2_generation",
    tokenizer=dplm2.tokenizer,
    struct_tokenizer=dplm2.struct_tokenizer, save_pdb=True
)

samples = dplm2_bit.generate(
  input_tokens=input_tokens,
  max_iter=500,
)
save_results(
    outputs=samples,
    task="co_generation",
    save_dir="./generation-results/dplm2_bit_generation",
    tokenizer=dplm2_bit.tokenizer,
    struct_tokenizer=dplm2_bit.struct_tokenizer
)