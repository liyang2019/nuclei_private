import pandas as pd
import os


gray_keys = [name.strip() for name in open('../data/2018-4-12_dataset/image_sets/test2_gray_key.txt')]
blue_keys = [name.strip() for name in open('../data/2018-4-12_dataset/image_sets/stage2_blue')]
purple_keys = [name.strip() for name in open('../data/2018-4-12_dataset/image_sets/stage2_purple')]
external_keys = [name.strip() for name in open('../data/2018-4-12_dataset/image_sets/stage2_external')]
pe_keys = purple_keys + external_keys

# gray_csv = pd.read_csv('ensemble_total/ensemble_submit.csv')
gray_csv = pd.read_csv('ensemble_total_fewer/ensemble_submit.csv')
blue_csv = pd.read_csv('ensemble_total_on_blue_fewer/ensemble_submit.csv')
pe_csv = pd.read_csv('ensemble_test_color_purple_external_wxw/combined_external_purple_submit.csv')

pe_names = set()
pe_name_list = []
pe_rle_list = []

print('reading purple and external')
for index, row in pe_csv.iterrows():
    pe_names.add(row['ImageId'])
    pe_name_list.append(row['ImageId'])
    pe_rle_list.append(row['EncodedPixels'])


for name in pe_keys:
    if name not in pe_names:
        print(name)
        pe_name_list.append(name)
        pe_rle_list.append('')


gray_names = set()
gray_name_list = []
gray_rle_list = []

print('reading gray')
for index, row in gray_csv.iterrows():
    gray_names.add(row['ImageId'])
    gray_name_list.append(row['ImageId'])
    gray_rle_list.append(row['EncodedPixels'])

for name in gray_keys:
    if name not in gray_names:
        print(name)
        gray_name_list.append(name)
        gray_rle_list.append('')

blue_names = set()
blue_name_list = []
blue_rle_list = []

print('reading blue')
for index, row in blue_csv.iterrows():
    blue_names.add(row['ImageId'])
    blue_name_list.append(row['ImageId'])
    blue_rle_list.append(row['EncodedPixels'])

for name in blue_keys:
    if name not in blue_names:
        print(name)
        blue_name_list.append(name)
        blue_rle_list.append('')

print(len(gray_name_list) == len(gray_rle_list))
print(set(gray_keys) == set(gray_name_list))
print(len(blue_name_list) == len(blue_rle_list))
print(set(blue_keys) == set(blue_name_list))

print(len(gray_name_list))
print(len(blue_name_list))

name_list = []
rle_list = []
for name, rle in zip(gray_name_list, gray_rle_list):
    name_list.append(name)
    rle_list.append(rle)
for name, rle in zip(blue_name_list, blue_rle_list):
    name_list.append(name)
    rle_list.append(rle)
for name, rle in zip(pe_name_list, pe_rle_list):
    name_list.append(name)
    rle_list.append(rle)

print(name_list == gray_name_list + blue_name_list + pe_name_list)
print(rle_list == gray_rle_list + blue_rle_list + pe_rle_list)

df = pd.DataFrame({'ImageId': name_list, 'EncodedPixels': rle_list})
os.makedirs('final_combined_csv_fewer', exist_ok=True)
df.to_csv('final_combined_csv_fewer/final_combined_fewer.csv', index=False, columns=['ImageId', 'EncodedPixels'])
