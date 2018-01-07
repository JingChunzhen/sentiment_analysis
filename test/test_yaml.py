import yaml

# how to add an element into the yaml file  

with open('config.txt', 'rb') as f:
    param_all = yaml.load(f)

print(param_all)

params = param_all['cnn']
a = params['filter_sizes']
b = params['embedding_size']
c = params['learning_rate']
d = params['benchmark']

print(a)
print(type(a)) # str

print(b)
print(type(b)) # int

print(c)
print(type(c)) # str

print(d)
print(type(d)) # float 




