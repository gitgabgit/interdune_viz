import yaml

with open("config.yaml", "r") as file:
    yaml_data = yaml.safe_load(file)



# get absolute filepaths on ISAAC
# reference local filepaths using config.yaml