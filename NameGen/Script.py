from visual import *

names  = open("Indian_Names.csv","r").read().splitlines()

dict_name = {}

for name in names:
    name = ["."] + list(name) + ["."]
    for ch1, ch2 in zip(name,name[1:]):
        bigram = (ch1,ch2)
        dict_name[bigram] = dict_name.get(bigram,0) + 1

# print(sorted(dict_name.items(), key = lambda key_value : -key_value[1]))

import torch

Model = torch.zeros((27,27),dtype=torch.int32)
# print(Model)

all_chars_az = sorted(list(set(''.join(names))))
print(len(all_chars_az))


string_to_number  = {char: index+1 for index, char in enumerate(all_chars_az)}
string_to_number["."] = 0
print(string_to_number)



for name in names:
    name = ["."] + list(name) + ["."]
    for ch1, ch2 in zip(name,name[1:]):
        row = string_to_number[ch1] 
        col = string_to_number[ch2]
        Model[row,col] += 1


visualize_data(Model,string_to_number)