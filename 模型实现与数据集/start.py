import os
import codecs

for i in range(100000):
    cmd='python DMVST_Net_backward.py'
    os.system(cmd)
    with codecs.open('./verifying_resault1','r','utf-8') as r:
        content=[s.strip() for s in r.readlines()]
    stop=int(content[-1].split()[-1])
    if stop==1:
        print(
            'The model has already been not very suitable for the verification set, and the model is close to the over-fitting state, the verification process is terminated.'
        )
        break