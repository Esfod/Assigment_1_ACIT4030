import numpy as np

def read_binvox(file_path):
    with open(file_path, 'rb') as f:
        # header
        line = f.readline().decode().strip()
        if line != '#binvox 1':
            raise IOError('Not a binvox file')

        dims = None
        while True:
            line = f.readline().decode().strip()
            if line.startswith('dim'):
                dims = list(map(int, line.split()[1:]))  # [X, Y, Z]
            elif line == 'data':
                break

        raw = np.frombuffer(f.read(), dtype=np.uint8)
        values, counts = raw[::2], raw[1::2]
        data = np.repeat(values, counts).astype(np.float32)
        data = data.reshape(dims)
        return data

