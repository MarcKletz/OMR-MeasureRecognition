#%%
for i in range(0,5000):
    print("\r " + str(i), end="", flush=True)
# %%
name = "model_0001799.pth"
iteration = int(name.split("_")[1].split(".")[0].strip("0"))
print(int(iteration))
print(type(iteration))
# %%
if os.path.exists(save_file):
    with open(save_file, 'w') as f:
        f.write(self._saved_model_name)