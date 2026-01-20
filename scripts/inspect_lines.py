import io
p = r"E:\\code\\EfficientSAM-main\\EfficientSAM-main\\train_sirst.py"
with io.open(p,'r',encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if 260 <= i <= 269:
            print(i, repr(line))
