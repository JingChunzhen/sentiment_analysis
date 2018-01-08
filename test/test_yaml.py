
import yaml

with open('../config.yaml', 'rb') as f:
    param_all = yaml.load(f)
    

params = param_all["CNN"]
for k, v in params.items():
    print("{} {} {}".format(k, v, type(v)))

print("----------------------")
params = param_all["Global"]

for k, v in params.items():
    print("{} {} {}".format(k, v, type(v)))

