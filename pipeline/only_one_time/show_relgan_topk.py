"""
对relgan生成的数据
选择k个和测试集最相似的
2023.12.24
"""
from pipeline.util import *
from pipeline.tfidf import *

if __name__ == "__main__":
    test_file = "../../methods/RelGAN/data/testdata/test_four.txt"
    output_file = "../../methods/RelGAN/real/experiments/out/20231223/four/four_rmc_vanilla_RSGAN_adam_bs32_sl20_sn0_dec0_ad-exp_npre30_nadv2000_ms1_hs256_nh2_ds5_dlr1e-4_glr1e-4_tem1000_demb64_nrep64_hdim32_sd171/samples/generator_text.txt"

    answers = read_txt2list(test_file)
    hyps = read_txt2list(output_file)

    top_k_pairs = get_top_k_pairs(answers, hyps, 10, max_threshold=0.8)
    for item in top_k_pairs:
        print_items = []
        print_items.append(item[0][0])
        print_items.append(item[0][1])
        print_items.append(str(item[1]))
        print("\t".join(print_items))