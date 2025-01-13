

def print_data_distribution(data1, data1_name, data2, data2_name, full_data):
    print(f"{data1_name}: {len(data1)}; {data2_name}: {len(data2)}; {data1_name} ratio: {len(data1)/ float(len(full_data)):.2f}")