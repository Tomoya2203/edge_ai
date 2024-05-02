import torch
from torch.utils.data import Dataset
import glob
from PIL import Image
from torchvision import transforms

a = glob.glob("./images/*.jpg")
b = glob.glob("./labels/*.txt")
a.sort()
b.sort()
# for i in range(len(a)):
#     c = a[i].find("_")
#     d = b[i].find("_")
#     e = a[i].rfind(".")
#     f = b[i].rfind(".")
#     if a[i][c:e] != b[i][d:f]:
#         print(a[i], b[i])
#         break
#     else:
#         print("OK")


csv_name = "experiment.csv"
IMG_SIZE = (300, 100)
with open (csv_name, 'w') as f:
    f.write(f"x:image, y_0:0, y_1:1, y_2:2, y_3:3, y_4:4, y_5:5, y_6:6, y_7:7, y_8:8, y_9:9\n")
    for i in range(len(a) // 4):
        a = Image.open(f"./images/number_{i * 4}.jpg")
        b = Image.open(f"./images/number_{i * 4 + 1}.jpg")
        c = Image.open(f"./images/number_{i * 4 + 2}.jpg")
        d = Image.open(f"./images/number_{i * 4 + 3}.jpg")
        a.resize(IMG_SIZE)
        b.resize(IMG_SIZE)
        c.resize(IMG_SIZE)
        d.resize(IMG_SIZE)
        new_image = Image.new("RGB", (IMG_SIZE[0] * 2, IMG_SIZE[1] * 2))
        new_image.paste(a, (0, 0))
        new_image.paste(b, (IMG_SIZE[0], 0))
        new_image.paste(c, (0, IMG_SIZE[1]))
        new_image.paste(d, (IMG_SIZE[0], IMG_SIZE[1]))
        path = f"./connect/connect_{i}.jpg"
        new_image.save(path)
        se = set()
        for j in range(4):
            with open(f"./labels/number_{i * 4 + j}.txt", 'r') as file:
                for line in file:
                    A, B, C, E, F = map(float, line.strip().split())
                    se.add(int(A))
        lis = [0] * 10
        se = list(se)
        for j in range(len(se)):
            lis[se[j]] = 1
        f.write(f"connect_{i}.jpg, {lis[0]}, {lis[1]}, {lis[2]},"
                f" {lis[3]}, {lis[4]}, {lis[5]}, {lis[6]}, {lis[7]}, {lis[8]}, {lis[9]}\n")

