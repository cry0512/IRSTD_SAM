import io
p = r"E:\\code\\EfficientSAM-main\\EfficientSAM-main\\train_sirst.py"
with io.open(p,'r',encoding='utf-8') as f:
    s = f.read()
s = s.replace('\\"freq_gate\\"', '"freq_gate"').replace('\\"radial_gate\\"', '"radial_gate"')
with io.open(p,'w',encoding='utf-8',newline='\n') as f:
    f.write(s)
print('patched')
