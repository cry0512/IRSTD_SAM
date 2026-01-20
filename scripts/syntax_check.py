import ast, sys
p = r"E:\\code\\EfficientSAM-main\\EfficientSAM-main\\train_sirst.py"
with open(p,'r',encoding='utf-8') as f:
    src=f.read()
ast.parse(src)
print('train_sirst.py: syntax OK, length', len(src))

sys.path.insert(0, r"E:\\code\\EfficientSAM-main\\EfficientSAM-main")
import efficient_sam.efficient_sam_encoder as enc
import efficient_sam.freq_modules as fm
print('encoder loaded:', hasattr(enc,'ImageEncoderViT'))
print('freq modules:', all(hasattr(fm, n) for n in ['FreqGate','RadialFreqGate','SpectralTransformLite','FFTformerDFFNLite']))
