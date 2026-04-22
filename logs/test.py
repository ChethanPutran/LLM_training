with open("logs/stage_2_0-480.log", "r") as f:
    lines = f.readlines()

count = 0
for i, line in enumerate(lines):
    if 'fwd_microstep' in line:
        print(i)
        count += 1
    if 'fwd:' in line:
        if count == 8:
            pass
        else:   
            raise Exception(f"Error at line {i}")
        print(i)
        count = 0