
from nupack import *
import re
import numpy as np

bases = np.array(list("ATCG"))
def DNA_complement1(sequence):
    # 构建互补字典
    comp_dict = {
        "A": "T",
        "T": "A",
        "G": "C",
        "C": "G",
        "a": "t",
        "t": "a",
        "g": "c",
        "c": "g",
    }
    # 求互补序列
    sequence_list = list(sequence)
    sequence_list = [comp_dict[base] for base in sequence_list]
    string = ''.join(sequence_list)
    return string
def DNA_reverse(sequence):
    return sequence[::-1]

def simulator(seq_pairs):
    #设置参数
    #浓度
    DFT_CONCERNTRATION = 1e-9
    #初始化模型
    model = Model(material='dna', celsius=21)

    yields = []
    strands = []
    for index,seq_pair in enumerate(seq_pairs.values):
        strand_pair = []
        name1 = '1'
        name2 = '2'
        strand_pair.append(Strand(seq_pair[0],name=name1))
        strand_pair.append(Strand(DNA_reverse(DNA_complement1(seq_pair[1])),name=name2))
        strands.append(strand_pair)
    tubes = []
    tube_names = []
    for index,strand in enumerate(strands):
        name_tube = 'tube'+str(index)
        tube_names.append(name_tube)
        tubes.append(Tube(strands={strand[0]:DFT_CONCERNTRATION,strand[1]:DFT_CONCERNTRATION},complexes=SetSpec(max_size=2),name=name_tube))
    results = tube_analysis(tubes,model=model,compute=['pfunc', 'pairs', 'mfe', 'sample', 'subopt'],
                              options={'num_sample': 2, 'energy_gap': 0.5})

    reg = r"(1\+2)|(2\+1)"
    for tube_name in tube_names:
        for my_complex, conc in results[tube_name].complex_concentrations.items():
            is_matched = re.search(reg, my_complex.name)
            if not is_matched is None:
                # print(conc)
                # conc = float('%.9f'%conc)
                # result = conc / DFT_CONCERNTRATION
                # result = float('%.9f'%result)
                yields.append(conc / DFT_CONCERNTRATION)
    return yields

# def simulator_2(seq_pair):
#     #设置参数
#     #浓度
#     DFT_CONCERNTRATION = 1e-9
#     #初始化模型
#     model = Model(material='dna', celsius=21)
#     #定义两条链
#     strand1 = Strand(seq_pair[0],name='1')
#     strand2 = Strand(DNA_reverse(DNA_complement1(seq_pair[1])),name='2')
#     #定义试管
#     tube1 = Tube(strands={strand1: DFT_CONCERNTRATION, strand2: DFT_CONCERNTRATION}, complexes=SetSpec(max_size=2),name='tube1')
#     #分析得到结果
#     result = tube_analysis([tube1],model=model,compute=['pfunc', 'pairs', 'mfe', 'sample', 'subopt'],
#                               options={'num_sample': 2, 'energy_gap': 0.5})
#
#     reg = r"(1\+2)|(2\+1)"
#     for my_complex, conc in result['tube1'].complex_concentrations.items():
#         print(my_complex.name)
#         is_matched = re.search(reg, my_complex.name)
#         if not is_matched is None:
#             yields = conc / DFT_CONCERNTRATION
#             return yields
#
#
# def multi_processing_simu()





